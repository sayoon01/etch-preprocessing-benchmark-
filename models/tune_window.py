"""모델 × window × target × seed grid 실행 → model_results.csv.

각 모델의 최적 window 크기를 LORO MAPE로 탐색.
"""
from __future__ import annotations
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import dataset as D
import model_eval as E

OUT = os.path.join(os.path.dirname(__file__), "outputs")
WINDOWS = [48, 64, 96, 128, 192, 256]
MODELS = ["XGBoost", "LSTM", "PatchTST"]
TARGETS = ["Poly_ER", "Remain_PR"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for W in WINDOWS:
        stride = max(1, W // 4)
        ds = D.make_dataset(W, stride)
        n = len(ds["groups"])
        print(f"[W={W}] windows={n}")
        for model in MODELS:
            for target in TARGETS:
                seeds = SEEDS if model != "XGBoost" else [0]  # XGB 결정적
                for seed in seeds:
                    r = E.eval_model(ds, model, target, seed=seed)
                    rows.append({"model": model, "window": W, "stride": stride,
                                 "n_windows": n, "target": target, "seed": seed,
                                 "MAPE": r["MAPE"], "MAE": r["MAE"],
                                 "base_MAPE": r["base_MAPE"]})
        # 중간 저장
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_results.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "model_results.csv"), index=False)
    print("saved:", os.path.join(OUT, "model_results.csv"), len(df), "rows")

    # 요약: 모델별 최적 window (Poly_ER, seed 평균 MAPE)
    print("\n=== Poly_ER: 모델 × window 평균 MAPE ===")
    piv = (df[df.target == "Poly_ER"].groupby(["model", "window"])["MAPE"]
           .mean().unstack("window").round(1))
    print(piv)
    print("\n=== 모델별 최적 window (Poly_ER) ===")
    for model in MODELS:
        sub = df[(df.target == "Poly_ER") & (df.model == model)]
        g = sub.groupby("window")["MAPE"].mean()
        best = g.idxmin()
        print(f"{model:9s}: best W={best}  MAPE={g.min():.1f}  (base={sub.base_MAPE.mean():.1f})")


if __name__ == "__main__":
    main()
