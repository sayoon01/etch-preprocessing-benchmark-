"""전처리 9세트 × 모델 3 × 타깃 3 = 81 실험 일괄 실행.

결과를 outputs/results.csv 로 저장.
"""
from __future__ import annotations
import os
import sys
import warnings
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
TARGETS = ["Poly_ER", "Poly_Depth", "Remain_PR"]


def load_runs_data():
    labels = pl.load_labels()
    runs = pp.discover_runs()
    data = {}
    for blk, d in labels.items():
        path = next((p for lbl, p in runs.items() if lbl.startswith(blk)), None)
        if path is None:
            raise RuntimeError(f"run 폴더 매칭 실패: {blk}")
        ws = wd.make_windows(pp.load_process(path))
        data[blk] = (ws, d["targets"])
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    runs_data = load_runs_data()
    models = ev.get_models()
    rows = []

    # feature 행렬은 전처리 세트마다 1번만 생성 (모델/타깃 간 공유)
    for prep, (b, c, d, e) in ft.PREP_SETS.items():
        X, y, groups = ft.build_matrix(runs_data, b, c, d, e)
        for mname, model in models.items():
            for tgt in TARGETS:
                res = ev.loro_eval(X.to_numpy(), y[tgt], groups, model)
                rows.append({
                    "prep": prep, "model": mname, "target": tgt,
                    "n_features": X.shape[1],
                    "run_MAE": res["run_MAE"], "run_MAPE": res["run_MAPE"],
                    "base_MAE": res["base_MAE"], "base_MAPE": res["base_MAPE"],
                    "win_MAE": res["win_MAE"], "win_R2": res["win_R2"],
                    "improve_vs_base_%": (res["base_MAE"] - res["run_MAE"])
                    / res["base_MAE"] * 100 if res["base_MAE"] else np.nan,
                })
    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "results.csv")
    df.to_csv(path, index=False)
    print(f"저장: {path}  ({len(df)} 행)")
    # 요약 출력: 타깃별 prep 평균 win_R2 (모델 평균)
    print("\n=== 타깃별 · 전처리별 window R² (3모델 평균) ===")
    piv = (df[df.prep.str.contains(r"\+", regex=True) | (df.prep == "A")]
           .pivot_table(index="prep", columns="target", values="win_R2", aggfunc="mean"))
    cum = ["A", "A+B", "A+B+C", "A+B+C+D", "A+B+C+D+E"]
    print(piv.reindex([p for p in cum if p in piv.index]).round(3))
    return df


if __name__ == "__main__":
    main()
