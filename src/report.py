"""results.csv (+ tuning/sensitivity/final) → 타깃별 그림 + REPORT.md 생성."""
from __future__ import annotations
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, "outputs")
CUM = ["A", "A+B", "A+B+C", "A+B+C+D", "A+B+C+D+E"]
ABL = ["A+B(only)", "A+C(only)", "A+D(only)", "A+E(only)"]
ABL_LABEL = {"A+B(only)": "B step-split", "A+C(only)": "C dynamics",
             "A+D(only)": "D ratios", "A+E(only)": "E cleaning"}
TARGETS = ["Poly_ER", "Poly_Depth", "Remain_PR"]


def _load(name):
    p = os.path.join(OUT, name)
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------- 타깃별 누적 그림 ----------
def fig_cumulative(df, target):
    sub = df[(df.target == target) & (df.prep.isin(CUM))]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in sub.model.unique():
        s = sub[sub.model == m].set_index("prep").reindex(CUM)
        ax.plot(CUM, s["run_MAPE"], marker="o", label=m)
    base = sub["base_MAPE"].mean()
    ax.axhline(base, ls="--", color="gray", label=f"mean-predictor ({base:.1f})")
    ax.set_ylabel("run-level MAPE (%)  ↓ better")
    ax.set_xlabel("cumulative preprocessing")
    ax.set_title(f"{target}: cumulative preprocessing vs LORO error")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_cumulative_{target}.png"), dpi=130)
    plt.close(fig)


def fig_ablation(df, target):
    sub = df[(df.target == target) & (df.model == "XGBoost") & df.prep.isin(ABL)]
    sub = sub.set_index("prep").reindex(ABL)
    vals = sub["improve_vs_base_%"]
    labels = [ABL_LABEL[p] for p in ABL]
    colors = ["tab:green" if v > 0 else "tab:red" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("improvement vs mean-predictor (%)  ↑ better")
    ax.set_title(f"Single-technique effect (XGBoost, {target})")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_ablation_{target}.png"), dpi=130)
    plt.close(fig)


def fig_sensitivity(sens):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["tab:green" if r < b else "tab:red"
              for r, b in zip(sens.run_MAPE, sens.base_MAPE)]
    ax.bar(sens.seg_set.astype(str), sens.run_MAPE, color=colors)
    for i, (r, n) in enumerate(zip(sens.run_MAPE, sens.n_windows)):
        ax.text(i, r + 1, f"n={n}", ha="center", fontsize=8)
    ax.axhline(sens.base_MAPE.mean(), ls="--", color="gray", label="mean-predictor")
    ax.set_ylabel("run-level MAPE (%)  ↓ better")
    ax.set_title("Segment sensitivity (A+B, XGBoost, Poly E/R)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sensitivity.png"), dpi=130)
    plt.close(fig)


def write_report(df, tuning, sens, final):
    cum_er = (df[(df.target == "Poly_ER") & (df.prep.isin(CUM))]
              .groupby("prep")[["run_MAPE", "base_MAPE"]].mean().reindex(CUM))
    abl_er = (df[(df.target == "Poly_ER") & (df.model == "XGBoost") & df.prep.isin(ABL)]
              .set_index("prep").reindex(ABL)["improve_vs_base_%"])

    L = []
    L += ["# 식각 공정 결과 예측 — 전처리 전략 비교 결과", ""]

    # 헤드라인: 최종 베스트
    if final is not None:
        L += ["## 0. 최종 베스트 파이프라인",
              "**ME1 단독 window + 튜닝 XGBoost(n=400, depth=3, lr=0.1)**",
              final.round(2).to_markdown(index=False), ""]

    L += ["## 1. 요약 (핵심 결론)",
          f"- **Step(ARC/ME1) 분리(B)가 가장 효과적인 전처리** (단독 +{abl_er['A+B(only)']:.0f}%).",
          "- **C/D/E 를 더할수록 악화** — n=3 에서 feature 과다 → 과적합 → held-out run 외삽 실패.",
          "- **식각 본구간(ME1)만 사용**할 때 가장 정확 (ARC·overetch·stable 은 노이즈).",
          "- 모델은 **XGBoost** 가 압도적, 튜닝으로 Poly E/R MAPE 46→22% 개선.",
          ""]

    L += ["## 2. 누적 전처리 단계별 (Poly E/R, 3모델 평균)",
          cum_er.round(2).to_markdown(),
          ""]
    for t in TARGETS:
        L += [f"![cum_{t}](fig_cumulative_{t}.png)"]
    L += [""]

    L += ["## 3. 단독 기법별 효과 (XGBoost, Poly E/R, baseline 대비 %)",
          abl_er.round(1).to_markdown(),
          "",
          "기법 기여도: **B(step분리) > D(비율) > C(변화량) ≫ E(이상치처리, 역효과)**",
          "![ablation](fig_ablation_Poly_ER.png)", ""]

    if tuning is not None:
        best = tuning.sort_values("run_MAPE").groupby("model").first().reset_index()
        L += ["## 4. 하이퍼파라미터 튜닝 (전처리=A+B, target=Poly_ER)",
              best[["model", "run_MAPE", "params"]].round(2).to_markdown(index=False),
              "", "→ XGBoost 만 baseline(64.7) 을 큰 폭으로 하회.", ""]

    if sens is not None:
        L += ["## 5. Segment 민감도 (A+B, XGBoost, Poly E/R)",
              sens.round(2).to_markdown(index=False),
              "",
              "→ **ME1 단독이 최고.** ARC(반사방지막 식각)·overetch·stable 은 poly 결과와 무관해 노이즈로 작용.",
              "![sensitivity](fig_sensitivity.png)", ""]

    L += ["## 6. 한계 (반드시 함께 읽을 것)",
          "- **라벨 run = 3개.** window(106개)는 feature 용이며 독립표본 아님 — 라벨은 run 당 1개.",
          "- LORO 3-fold → 결과는 **통계적 유의성 아닌 경향**.",
          "- 트리는 학습 타깃범위 밖 외삽 불가 → held-out 극단값이면 큰 오차.",
          "- Poly E/R = Poly Depth × 7.5 (식각시간 80s 고정) → 두 타깃은 동일 예측문제.",
          "- **권장:** DOE 로 run>20 확보 시 본 파이프라인 그대로 정식 벤치마크 가능."]

    path = os.path.join(OUT, "REPORT.md")
    open(path, "w").write("\n".join(L))
    print("저장:", path)


def main():
    df = _load("results.csv")
    tuning, sens, final = _load("tuning.csv"), _load("sensitivity.csv"), _load("final.csv")
    for t in TARGETS:
        fig_cumulative(df, t)
        fig_ablation(df, t)
    if sens is not None:
        fig_sensitivity(sens)
    write_report(df, tuning, sens, final)
    print("그림 생성 완료")


if __name__ == "__main__":
    main()
