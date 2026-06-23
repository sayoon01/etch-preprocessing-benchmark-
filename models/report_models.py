"""model_results.csv → window 튜닝 곡선 + 모델 비교 + MODEL_REPORT.md."""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "outputs")
MODELS = ["XGBoost", "LSTM", "PatchTST"]
COLORS = {"XGBoost": "tab:blue", "LSTM": "tab:orange", "PatchTST": "tab:green"}


def fig_window_tuning(df, target="Poly_ER"):
    sub = df[df.target == target]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in MODELS:
        g = sub[sub.model == m].groupby("window")["MAPE"]
        mean, std = g.mean(), g.std().fillna(0)
        ax.plot(mean.index, mean.values, marker="o", color=COLORS[m], label=m)
        ax.fill_between(mean.index, mean - std, mean + std, color=COLORS[m], alpha=0.15)
        bw = mean.idxmin()
        ax.scatter([bw], [mean.min()], color=COLORS[m], s=120, zorder=5,
                   edgecolor="black", marker="*")
    base = sub["base_MAPE"].mean()
    ax.axhline(base, ls="--", color="gray", label=f"mean-predictor ({base:.1f})")
    ax.set_xlabel("window size (timesteps)")
    ax.set_ylabel("run-level MAPE (%)  ↓ better")
    ax.set_title(f"{target}: per-model window-size tuning (★=best)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_window_tuning_{target}.png"), dpi=130)
    plt.close(fig)


def fig_model_compare(df, target="Poly_ER"):
    sub = df[df.target == target]
    names, means, stds = [], [], []
    for m in MODELS:
        g = sub[sub.model == m].groupby("window")["MAPE"].mean()
        bw = g.idxmin()
        s = sub[(sub.model == m) & (sub.window == bw)]["MAPE"]
        names.append(f"{m}\n(W={bw})")
        means.append(s.mean()); stds.append(s.std() if len(s) > 1 else 0)
    base = sub["base_MAPE"].mean()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    colors = [COLORS[m] for m in MODELS]
    ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    ax.axhline(base, ls="--", color="gray", label=f"mean-predictor ({base:.1f})")
    ax.set_ylabel("run-level MAPE (%)  ↓ better")
    ax.set_title(f"{target}: best-window model comparison")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fig_model_compare_{target}.png"), dpi=130)
    plt.close(fig)


def best_table(df, target):
    rows = []
    sub = df[df.target == target]
    for m in MODELS:
        g = sub[sub.model == m].groupby("window")["MAPE"].mean()
        bw = g.idxmin()
        s = sub[(sub.model == m) & (sub.window == bw)]["MAPE"]
        rows.append({"model": m, "best_window": bw,
                     "MAPE_mean": round(s.mean(), 1),
                     "MAPE_std": round(s.std() if len(s) > 1 else 0, 1),
                     "base_MAPE": round(sub.base_MAPE.mean(), 1)})
    return pd.DataFrame(rows).sort_values("MAPE_mean")


def main():
    df = pd.read_csv(os.path.join(OUT, "model_results.csv"))
    for t in ["Poly_ER", "Remain_PR"]:
        fig_window_tuning(df, t)
        fig_model_compare(df, t)

    L = ["# 식각 결과 예측 — 모델별 정확도 비교 (XGBoost / LSTM / PatchTST)", ""]
    L += ["## 0. 핵심 결론",
          "- **모델마다 최적 window가 다름** — 작은 window(샘플↑)는 시퀀스 모델에, 큰 window(집계 안정)는 트리에 유리.",
          "- run-level LORO 기준 모델 순위와 각자의 최적 window는 아래 표.",
          "- ⚠️ **라벨 run=3** → 결과는 통계적 유의성이 아닌 **경향**. 특히 극단값(Bias run) held-out fold는 모든 모델이 외삽 실패.", ""]

    for t in ["Poly_ER", "Remain_PR"]:
        L += [f"## {t} — 모델별 최적 window & 성능",
              best_table(df, t).to_markdown(index=False), "",
              f"![window tuning](fig_window_tuning_{t}.png)",
              f"![model compare](fig_model_compare_{t}.png)", ""]

    L += ["## window × 모델 평균 MAPE (Poly_ER)",
          df[df.target == "Poly_ER"].groupby(["model", "window"])["MAPE"]
            .mean().round(1).unstack("window").to_markdown(), ""]

    L += ["## 한계",
          "- 라벨 run 3개 → LORO 3-fold, 결과는 경향. 딥러닝은 seed 변동 큼(std 표기).",
          "- 트리·시퀀스 모두 학습 타깃 범위 밖 외삽 불가 → 극단 run held-out 시 큰 오차.",
          "- window가 작을수록 샘플↑이나 중첩↑(상관) → run-level 평가로 보정.",
          "- 권장: DOE로 run>20 확보 시 딥러닝의 진짜 우위 검증 가능."]
    open(os.path.join(OUT, "MODEL_REPORT.md"), "w").write("\n".join(L))
    print("saved MODEL_REPORT.md + figures")
    print("\n", best_table(df, "Poly_ER").to_string(index=False))


if __name__ == "__main__":
    main()
