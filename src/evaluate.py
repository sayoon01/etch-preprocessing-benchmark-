"""LORO(Leave-One-Run-Out) 교차검증 + 모델 정의.

n=3 이므로 LORO = 3 fold. 각 fold 에서 2 run 학습 → 남은 run 의 window 예측.
 - run-level (주지표): 남은 run window 예측을 median 집계 → 1 예측 vs 정답 1
 - window-level (보조): 모든 held-out window 예측을 pool → run 구분 능력(R²/MAE)
 - baseline: mean-predictor (학습 run 들의 타깃 평균으로 예측)

⚠️ 트리 모델은 학습 타깃 범위 밖을 외삽하지 못한다. n=3 LORO 에서 held-out run 이
   극단값이면 큰 오차가 불가피 — 이는 데이터 부족의 본질적 한계로 보고서에 명시.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def get_models(seed: int = 0) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, verbose=-1),
    }


def loro_eval(X, y, groups, model):
    """단일 (X, y, model) → 지표 dict."""
    uniq = list(dict.fromkeys(groups))  # run 순서 보존
    run_true, run_pred, base_pred = [], [], []
    win_true_all, win_pred_all = [], []

    for held in uniq:
        tr = groups != held
        te = groups == held
        m = _clone(model)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        win_pred_all.extend(p)
        win_true_all.extend(y[te])
        run_pred.append(float(np.median(p)))
        run_true.append(float(y[te][0]))           # run 내 동일 타깃
        base_pred.append(float(np.mean(y[tr])))    # mean-predictor (LOO)

    run_true = np.array(run_true); run_pred = np.array(run_pred)
    base_pred = np.array(base_pred)
    wt = np.array(win_true_all); wp = np.array(win_pred_all)
    return {
        "run_MAE": mean_absolute_error(run_true, run_pred),
        "run_MAPE": float(np.mean(np.abs((run_true - run_pred) / run_true)) * 100),
        "base_MAE": mean_absolute_error(run_true, base_pred),
        "base_MAPE": float(np.mean(np.abs((run_true - base_pred) / run_true)) * 100),
        "win_MAE": mean_absolute_error(wt, wp),
        "win_R2": r2_score(wt, wp) if len(set(wt)) > 1 else float("nan"),
        "run_true": run_true.tolist(),
        "run_pred": run_pred.tolist(),
    }


def _clone(model):
    from sklearn.base import clone
    return clone(model)
