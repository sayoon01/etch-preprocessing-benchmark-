"""XGBoost 회귀 (윈도우 집계 feature 입력)."""
from __future__ import annotations
import numpy as np
from xgboost import XGBRegressor


def train_predict(feat_tr, y_tr, feat_te, seed=0):
    m = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9,
                     random_state=seed, n_jobs=-1)
    m.fit(feat_tr, y_tr)
    return np.asarray(m.predict(feat_te), dtype=float)
