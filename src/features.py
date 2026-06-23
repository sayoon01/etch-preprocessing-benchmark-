"""전처리 전략 A~E feature 빌더.

window(192행) 단위로 feature 벡터를 만든다. 전략은 누적 토글:
  A: ACTUAL 센서 window 평균 (raw)                  — 단순 수치 변환
  B: + step(seg) one-hot                            — ARC/ME1 분리
  C: + 동적특성(std/slope/min/max) + SP-실측 편차    — 변화량 feature
  D: + 도메인 비율(가스/압력/파워)                   — 비율 feature
  E: 행 수준 이상치/결측 처리 후 재집계               — 데이터 품질

ablation 용으로 개별 토글(use_b/c/d/e)을 독립적으로 켤 수 있다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---- 센서 컬럼 그룹 ----
ACTUAL = [
    "aiRFBIAS_ForwardPwr", "aiRFBIAS_ReflectPwr",
    "aiRFSOURCE_ForwardPwr", "aiRFSOURCE_ReflectPwr",
    *[f"aiMFC_FLOW_{i:02d}" for i in range(1, 13)],
    "aiESC_VOLTAGE_MONITORING",
    "avHE_1_LeakRateFlow", "aiHE_1_FLOW_SIGNAL", "aiHE_1_PRESSURE_SIGNAL",
    "avHE_2_LeakRateFlow", "aiHE_2_FLOW_SIGNAL", "aiHE_2_PRESSURE_SIGNAL",
    "aiAPC_CurrentPressure",
    "aiTEMP_PV1ch1", "aiTEMP_PV1ch2", "aiTEMP_PV1ch3",
]
SETPOINT = {  # 실측 ↔ 설정 짝 (SP-실측 편차용)
    "aiRFBIAS_ForwardPwr": "aoRFBIAS_Setpoint",
    "aiRFSOURCE_ForwardPwr": "aoRFSOURCE_Setpoint",
    **{f"aiMFC_FLOW_{i:02d}": f"aoMFC_FLOW_{i:02d}" for i in range(1, 13)},
}
SEGS = ["ARC", "ME1"]
# 가스 채널 매핑 (parse_labels 에서 자동도출된 결과; 3 run 공통)
GAS_CH = {"HBr": "aiMFC_FLOW_01", "Cl2": "aiMFC_FLOW_04",
          "CF4": "aiMFC_FLOW_07", "O2/He": "aiMFC_FLOW_11"}


def _slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    return float(np.polyfit(x, y, 1)[0])


def clean_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """전략 E: 행 수준 이상치/결측 처리.

    - RF off 구간 제거(소스 forward 가 거의 0 인 행)
    - reflect power 스파이크 clip(분위수 기반)
    - 센서 0-dropout 은 직전값으로 보간(MFC 실측)
    """
    r = rows.copy()
    src = r["aiRFSOURCE_ForwardPwr"]
    if (src > 5).any():  # 점화된 구간이 있으면 off 행 제거
        r = r[src > 5]
    if len(r) == 0:
        r = rows.copy()
    for col in ("aiRFBIAS_ReflectPwr", "aiRFSOURCE_ReflectPwr"):
        hi = r[col].quantile(0.99)
        if hi > 0:
            r[col] = r[col].clip(upper=hi)
    for ch in GAS_CH.values():
        s = r[ch].replace(0.0, np.nan).ffill().bfill()
        r[ch] = s.fillna(0.0)
    return r.reset_index(drop=True)


def window_features(rows: pd.DataFrame, seg_name: str,
                    use_b=True, use_c=True, use_d=True, use_e=True,
                    segs=None) -> dict:
    """단일 window → feature dict."""
    if use_e:
        rows = clean_rows(rows)
    seg_list = segs if segs is not None else SEGS
    f = {}

    # A: ACTUAL 평균
    for c in ACTUAL:
        f[f"{c}__mean"] = float(rows[c].mean())

    # B: step one-hot
    if use_b:
        for s in seg_list:
            f[f"seg_{s}"] = 1.0 if seg_name == s else 0.0

    # C: 동적특성 + SP-실측 편차
    if use_c:
        for c in ACTUAL:
            v = rows[c].to_numpy(dtype=float)
            f[f"{c}__std"] = float(np.std(v))
            f[f"{c}__min"] = float(np.min(v))
            f[f"{c}__max"] = float(np.max(v))
            f[f"{c}__slope"] = _slope(v)
        for ai, ao in SETPOINT.items():
            f[f"{ai}__sp_err"] = float((rows[ai] - rows[ao]).mean())

    # D: 도메인 비율
    if use_d:
        m = {c: float(rows[c].mean()) for c in ACTUAL}
        eps = 1e-6
        hbr, cl2 = m[GAS_CH["HBr"]], m[GAS_CH["Cl2"]]
        cf4, o2 = m[GAS_CH["CF4"]], m[GAS_CH["O2/He"]]
        total_gas = sum(m[f"aiMFC_FLOW_{i:02d}"] for i in range(1, 13))
        bias_fwd, src_fwd = m["aiRFBIAS_ForwardPwr"], m["aiRFSOURCE_ForwardPwr"]
        apc = m["aiAPC_CurrentPressure"]
        f["ratio_Cl2_HBr"] = cl2 / (hbr + eps)
        f["ratio_CF4_total"] = cf4 / (total_gas + eps)
        f["ratio_O2_total"] = o2 / (total_gas + eps)
        f["total_gas"] = total_gas
        f["ratio_bias_source"] = bias_fwd / (src_fwd + eps)
        f["source_per_pressure"] = src_fwd / (apc + eps)
        f["bias_per_pressure"] = bias_fwd / (apc + eps)
        f["he_backside_per_apc"] = m["aiHE_1_PRESSURE_SIGNAL"] / (apc + eps)
    return f


def build_matrix(runs_data: dict, use_b=True, use_c=True, use_d=True, use_e=True,
                 segs=None):
    """runs_data: {run_label: (windows, targets)} → (X, y_dict, groups)."""
    rows_feat, groups, ys = [], [], {k: [] for k in ("Remain_PR", "Poly_Depth", "Poly_ER")}
    for label, (windows, targets) in runs_data.items():
        for w in windows:
            feat = window_features(w["rows"], w["seg_name"],
                                   use_b=use_b, use_c=use_c, use_d=use_d, use_e=use_e,
                                   segs=segs)
            rows_feat.append(feat)
            groups.append(label)
            for t in ys:
                ys[t].append(targets.get(t))
    X = pd.DataFrame(rows_feat).fillna(0.0)
    y = {t: np.array(v, dtype=float) for t, v in ys.items()}
    return X, y, np.array(groups)


# 전처리 세트 정의: (이름, b, c, d, e)
PREP_SETS = {
    "A":        (False, False, False, False),
    "A+B":      (True,  False, False, False),
    "A+B+C":    (True,  True,  False, False),
    "A+B+C+D":  (True,  True,  True,  False),
    "A+B+C+D+E": (True, True,  True,  True),
    # 단독 ablation (A 대비)
    "A+B(only)": (True,  False, False, False),
    "A+C(only)": (False, True,  False, False),
    "A+D(only)": (False, False, True,  False),
    "A+E(only)": (False, False, False, True),
}


if __name__ == "__main__":
    import parse_process as pp, parse_labels as pl, windowing as wd
    labels = pl.load_labels()
    runs = pp.discover_runs()
    runs_data = {}
    for blk, d in labels.items():
        path = next((p for lbl, p in runs.items() if lbl.startswith(blk)), None)
        ws = wd.make_windows(pp.load_process(path))
        runs_data[blk] = (ws, d["targets"])
    for name, (b, c, e_d, e) in PREP_SETS.items():
        X, y, g = build_matrix(runs_data, b, c, e_d, e)
        print(f"{name:12s} X={X.shape}  groups={dict(zip(*np.unique(g, return_counts=True)))}")
