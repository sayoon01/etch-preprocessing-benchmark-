"""Excel 라벨 파서 + MFC↔가스 매핑 자동 도출.

- AIetch-process-data.xlsx 에서 run별 (ARC/ME1) 레시피 설정값과
  예측 대상(Remain PR / Poly Depth / Poly E/R)을 추출한다.
- RECIPE_*_PM2 파일의 step별 MFC 설정값을 Excel 가스 컬럼값과 매칭해
  MFC 채널 ↔ 가스 이름 매핑을 자동 도출한다.
"""
from __future__ import annotations
import glob
import os
import re
import pandas as pd

XLSX_DEFAULT = "AIetch-process-data.xlsx"

# Excel 레시피 행의 컬럼 인덱스(0-base, 원본 시트 기준)
RECIPE_COLS = {
    "Pressure": 5, "RFT": 6, "RFB": 7, "Cl2": 8, "HBr": 9, "He": 10,
    "CF4": 11, "O2/He": 12, "BHe": 13, "Time": 14, "Temp": 15,
}
GAS_COLS = ["Cl2", "HBr", "He", "CF4", "O2/He"]
# '(' 앞 접두사로 매칭 (㎛/㎚/Å 등 유니코드 차이 회피)
TARGETS = {
    "Remain PR": "Remain_PR",
    "Poly Depth": "Poly_Depth",
    "Poly E/R": "Poly_ER",
}
CENTER_COL = 4  # 결과값이 들어있는 'C'(center) 컬럼 인덱스

# Excel 블록명 → 장비 run 라벨 매칭 키
BLOCK_NAMES = ["Standard", "Source power", "Bias power"]


def load_labels(xlsx: str = XLSX_DEFAULT) -> dict[str, dict]:
    """블록명 → {recipe: {ARC:{}, ME1:{}}, targets: {}} 반환."""
    df = pd.read_excel(xlsx, header=None)
    out = {}
    for i in range(len(df)):
        name = df.iat[i, 1] if df.shape[1] > 1 else None
        if isinstance(name, str) and name.strip() in BLOCK_NAMES:
            out[name.strip()] = _parse_block(df, i)
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_block(df: pd.DataFrame, start: int) -> dict:
    # start 이후 'Date' 헤더 → 그 다음 두 행이 ARC, ME1
    hdr = None
    for r in range(start, min(start + 6, len(df))):
        if str(df.iat[r, 1]).strip() == "Date":
            hdr = r
            break
    arc, me1 = hdr + 1, hdr + 2
    recipe = {
        "ARC": {k: _num(df.iat[arc, c]) for k, c in RECIPE_COLS.items()},
        "ME1": {k: _num(df.iat[me1, c]) for k, c in RECIPE_COLS.items()},
    }
    # 결과값: 다음 블록 전까지 스캔
    targets = {}
    for r in range(hdr, min(hdr + 15, len(df))):
        key = str(df.iat[r, 1]).strip().split("(")[0].strip()
        if key in TARGETS:
            targets[TARGETS[key]] = _num(df.iat[r, CENTER_COL])
    return {"recipe": recipe, "targets": targets}


# ---------- MFC ↔ 가스 매핑 (RECIPE 파일) ----------

def _parse_recipe_steps(recipe_path: str) -> dict[str, dict]:
    """RECIPE 파일 → {step_name: {'MFC1..12':v, 'source':v, 'bias':v}}."""
    steps, cur, name = {}, None, None
    for ln in open(recipe_path, encoding="latin1"):
        parts = ln.strip().split("\t")
        if len(parts) < 2:
            continue
        k, v = parts[0], parts[1]
        if k == "Step_Name":
            name = v.strip()
            cur = steps.setdefault(name, {})
        elif cur is not None:
            m = re.fullmatch(r"MFC(\d+)", k)
            if m:
                cur[f"MFC{int(m.group(1))}"] = _num(v)
            elif k == "RF_source_power_setpoint":
                cur["source"] = _num(v)
            elif k == "RF_bias_power_setpoint":
                cur["bias"] = _num(v)
    return steps


def derive_mfc_gas_map(run_dir: str, labels_block: dict) -> dict[str, str]:
    """RECIPE 의 step별 MFC 설정값을 Excel 가스값과 매칭 → {'MFC7':'CF4', ...}."""
    rec_files = glob.glob(os.path.join(run_dir, "RECIPE_*_PM2"))
    if not rec_files:
        return {}
    steps = _parse_recipe_steps(rec_files[0])
    mapping = {}
    for step in ("ARC", "ME1"):
        if step not in steps:
            continue
        excel_gas = labels_block["recipe"][step]
        mfc_vals = {k: v for k, v in steps[step].items() if k.startswith("MFC")}
        # 이 step 에서 흘린 가스(>0) 중 아직 매핑 안 된 것, nonzero MFC 채널
        already = set(mapping.values())
        active_gas = [g for g in GAS_COLS
                      if (excel_gas.get(g) or 0) > 0 and g not in already]
        active_mfc = [m for m, v in mfc_vals.items() if (v or 0) > 0]
        # 1차: 값 정확 매칭
        for gas in list(active_gas):
            tgt = excel_gas[gas]
            for mfc in active_mfc:
                if mfc not in mapping and abs(mfc_vals[mfc] - tgt) < 1e-6:
                    mapping[mfc] = gas
                    active_gas.remove(gas)
                    break
        # 2차: 남은 가스 ↔ 남은 nonzero 채널 1:1 소거법 (예: Cl2)
        rem_mfc = [m for m in active_mfc if m not in mapping]
        if len(active_gas) == 1 and len(rem_mfc) == 1:
            mapping[rem_mfc[0]] = active_gas[0]
    return mapping


if __name__ == "__main__":
    import parse_process as pp

    labels = load_labels()
    runs = pp.discover_runs()
    print("=== Excel 라벨 ===")
    for blk, d in labels.items():
        print(f"\n[{blk}] targets={d['targets']}")
        print(f"  ARC recipe={d['recipe']['ARC']}")
        print(f"  ME1 recipe={d['recipe']['ME1']}")

    print("\n=== MFC↔가스 매핑 (run별 RECIPE 자동도출) ===")
    for blk in labels:
        # 블록명으로 run 폴더 찾기 (prefix 매칭)
        match = next((p for lbl, p in runs.items() if lbl.startswith(blk)), None)
        if not match:
            print(f"[{blk}] 매칭 run 폴더 없음")
            continue
        m = derive_mfc_gas_map(match, labels[blk])
        print(f"[{blk}] {m}")
