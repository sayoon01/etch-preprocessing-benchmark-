"""추가 분석: ② 하이퍼파라미터 튜닝, ③ segment 민감도.

- 튜닝: 모델별 소규모 grid 를 LORO 로 평가(전처리=A+B, 타깃=Poly_ER) → best 보고
- 민감도: window 에 포함할 step 구간 조합을 바꿔가며 성능 변화 측정
결과 → outputs/tuning.csv, outputs/sensitivity.csv
"""
from __future__ import annotations
import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import parse_process as pp
import parse_labels as pl
import windowing as wd
import features as ft
import evaluate as ev

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "outputs")
TARGET = "Poly_ER"


def _runs_data(use_segs):
    labels = pl.load_labels()
    runs = pp.discover_runs()
    data = {}
    for blk, d in labels.items():
        path = next((p for lbl, p in runs.items() if lbl.startswith(blk)), None)
        ws = wd.make_windows(pp.load_process(path), use_segs=use_segs)
        data[blk] = (ws, d["targets"])
    return data


# ---------- ② 하이퍼파라미터 튜닝 ----------
def tune():
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    data = _runs_data(wd.USE_SEGS)
    X, y, g = ft.build_matrix(data, use_b=True, use_c=False, use_d=False, use_e=False)
    X = X.to_numpy(); yt = y[TARGET]

    grids = {
        "RandomForest": (RandomForestRegressor, dict(
            n_estimators=[200, 500], max_depth=[3, 5, None],
            max_features=["sqrt", 1.0], random_state=[0], n_jobs=[-1])),
        "XGBoost": (XGBRegressor, dict(
            n_estimators=[200, 400], max_depth=[2, 3, 4],
            learning_rate=[0.03, 0.05, 0.1], random_state=[0], n_jobs=[-1])),
        "LightGBM": (LGBMRegressor, dict(
            n_estimators=[200, 400], num_leaves=[7, 15, 31],
            learning_rate=[0.03, 0.05, 0.1], random_state=[0],
            n_jobs=[-1], verbose=[-1])),
    }
    rows = []
    for mname, (cls, grid) in grids.items():
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            res = ev.loro_eval(X, yt, g, cls(**params))
            disp = {k: v for k, v in params.items()
                    if k not in ("random_state", "n_jobs", "verbose")}
            rows.append({"model": mname, "params": str(disp),
                         "run_MAPE": res["run_MAPE"], "base_MAPE": res["base_MAPE"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "tuning.csv"), index=False)
    best = df.sort_values("run_MAPE").groupby("model").first().reset_index()
    print("=== 튜닝 best (전처리=A+B, target=Poly_ER, baseline=64.7) ===")
    print(best[["model", "run_MAPE", "params"]].to_string(index=False))
    return df, best


# ---------- ③ segment 민감도 ----------
def sensitivity():
    from xgboost import XGBRegressor
    seg_sets = {
        "ME1 only": ("ME1",),
        "ARC+ME1 (default)": ("ARC", "ME1"),
        "ARC+ME1+overetch": ("ARC", "ME1", "overetch"),
        "all 4 segs": ("stable", "ARC", "ME1", "overetch"),
    }
    rows = []
    for name, segs in seg_sets.items():
        data = _runs_data(segs)
        X, y, g = ft.build_matrix(data, use_b=True, use_c=False, use_d=False,
                                  use_e=False, segs=list(segs))
        n_win = len(g)
        res = ev.loro_eval(X.to_numpy(), y[TARGET], g,
                           XGBRegressor(n_estimators=300, max_depth=4,
                                        learning_rate=0.05, random_state=0, n_jobs=-1))
        rows.append({"seg_set": name, "n_windows": n_win,
                     "run_MAPE": res["run_MAPE"], "base_MAPE": res["base_MAPE"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)
    print("\n=== segment 민감도 (A+B, XGBoost, target=Poly_ER) ===")
    print(df.round(2).to_string(index=False))
    return df


# ---------- 최종 베스트 파이프라인 (ME1 단독 + 튜닝 XGBoost) ----------
def final_pipeline():
    from xgboost import XGBRegressor
    data = _runs_data(("ME1",))
    X, y, g = ft.build_matrix(data, use_b=True, use_c=False, use_d=False,
                              use_e=False, segs=["ME1"])
    rows = []
    for tgt in ("Poly_ER", "Poly_Depth", "Remain_PR"):
        res = ev.loro_eval(X.to_numpy(), y[tgt], g,
                           XGBRegressor(n_estimators=400, max_depth=3,
                                        learning_rate=0.1, random_state=0, n_jobs=-1))
        rows.append({"target": tgt, "run_MAPE": res["run_MAPE"],
                     "base_MAPE": res["base_MAPE"],
                     "improve_%": (res["base_MAPE"] - res["run_MAPE"])
                     / res["base_MAPE"] * 100})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "final.csv"), index=False)
    print("\n=== 최종 베스트: ME1단독 + 튜닝XGBoost(n400,d3,lr0.1) ===")
    print(df.round(2).to_string(index=False))
    return df


# ---------- leakage 실증: random split vs LORO ----------
def leakage_demo():
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from xgboost import XGBRegressor
    data = _runs_data(wd.USE_SEGS)
    X, y, g = ft.build_matrix(data, True, True, True, False)  # A+B+C+D (heavy)
    X = X.to_numpy(); yt = y[TARGET]
    kf = KFold(5, shuffle=True, random_state=0)
    rp = np.zeros_like(yt)
    for tr, te in kf.split(X):
        m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         random_state=0, n_jobs=-1)
        m.fit(X[tr], yt[tr]); rp[te] = m.predict(X[te])
    rand = {"split": "random window (leak)",
            "R2": r2_score(yt, rp),
            "MAPE": float(np.mean(np.abs((yt - rp) / yt)) * 100)}
    res = ev.loro_eval(X, yt, g, XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0, n_jobs=-1))
    loro = {"split": "LORO (run group)", "R2": res["win_R2"], "MAPE": res["run_MAPE"]}
    df = pd.DataFrame([rand, loro])
    df.to_csv(os.path.join(OUT, "leakage.csv"), index=False)
    print("\n=== leakage 실증 (동일 모델·A+B+C+D, target=Poly_ER) ===")
    print(df.round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tune()
    sensitivity()
    final_pipeline()
    leakage_demo()
