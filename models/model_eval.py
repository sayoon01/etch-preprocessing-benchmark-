"""LORO 평가 (모델 공통). run-level MAPE/MAE + mean-baseline."""
from __future__ import annotations
import numpy as np
import dataset as D
import xgb_model


def _import_dl(name):
    if name == "LSTM":
        import lstm_model as m
    elif name == "PatchTST":
        import patchtst as m
    else:
        raise ValueError(name)
    return m


def eval_model(ds, model_name: str, target: str, seed: int = 0) -> dict:
    seq, seg, groups = ds["seq"], ds["seg"], ds["groups"]
    y = ds["y"][target]
    uniq = list(dict.fromkeys(groups.tolist()))

    run_true, run_pred, base_pred = [], [], []
    for held in uniq:
        tr = groups != held
        te = groups == held
        if model_name == "XGBoost":
            ftr = D.agg_features(seq[tr], seg[tr])
            fte = D.agg_features(seq[te], seg[te])
            pred = xgb_model.train_predict(ftr, y[tr], fte, seed=seed)
        else:
            mu, sd = D.channel_norm_stats(seq[tr])
            ntr = (seq[tr] - mu) / sd
            nte = (seq[te] - mu) / sd
            pred = _import_dl(model_name).train_predict(ntr, y[tr], nte, seed=seed)
        run_pred.append(float(np.median(pred)))
        run_true.append(float(y[te][0]))
        base_pred.append(float(y[tr].mean()))

    rt = np.array(run_true); rp = np.array(run_pred); bp = np.array(base_pred)
    mape = float(np.mean(np.abs((rt - rp) / rt)) * 100)
    mae = float(np.mean(np.abs(rt - rp)))
    base_mape = float(np.mean(np.abs((rt - bp) / rt)) * 100)
    return {"MAPE": mape, "MAE": mae, "base_MAPE": base_mape,
            "run_true": rt.tolist(), "run_pred": rp.tolist()}


if __name__ == "__main__":
    ds = D.make_dataset(96, 24)
    print("XGBoost Poly_ER:", eval_model(ds, "XGBoost", "Poly_ER", 0))
