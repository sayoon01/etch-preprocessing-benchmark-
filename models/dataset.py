"""윈도우 → 모델 입력 데이터셋.

- DL용: 원시 시퀀스 텐서 [N, T, C]
- XGBoost용: 윈도우 집계 feature [N, 5C(+step)]
- 정규화는 평가 루프에서 학습 run 통계로만 수행(누설 차단).
"""
from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
import parse_process as pp
import parse_labels as pl
import windowing as wd

# 딥러닝 과적합 억제 위해 핵심 12채널만 (3모델 공통 입력)
CHANNELS = [
    "aiRFBIAS_ForwardPwr", "aiRFBIAS_ReflectPwr",
    "aiRFSOURCE_ForwardPwr", "aiRFSOURCE_ReflectPwr",
    "aiAPC_CurrentPressure",
    "aiMFC_FLOW_01", "aiMFC_FLOW_04", "aiMFC_FLOW_07", "aiMFC_FLOW_11",
    "aiTEMP_PV1ch1", "aiTEMP_PV1ch2", "aiTEMP_PV1ch3",
]
SEGS = ("ARC", "ME1")
TARGETS = ("Poly_ER", "Remain_PR")


def make_dataset(window: int, stride: int):
    """→ dict(seq[N,T,C], seg[N], groups[N], y={target:[N]})."""
    labels = pl.load_labels(os.path.join(ROOT, "AIetch-process-data.xlsx"))
    runs = pp.discover_runs(os.path.join(ROOT, "Equipment Data"))
    seqs, segs, groups, ys = [], [], [], {t: [] for t in TARGETS}
    for blk, d in labels.items():
        path = next((p for lbl, p in runs.items() if lbl.startswith(blk)), None)
        df = pp.load_process(path)
        for w in wd.make_windows(df, win=window, stride=stride, use_segs=SEGS):
            arr = w["rows"][CHANNELS].to_numpy(dtype=np.float32)  # [T, C]
            if arr.shape[0] != window:                            # 짧은 구간 패딩 방지: skip
                if arr.shape[0] < window:
                    continue
                arr = arr[:window]
            seqs.append(arr)
            segs.append(w["seg_name"])
            groups.append(blk)
            for t in TARGETS:
                ys[t].append(float(d["targets"][t]))
    return {
        "seq": np.stack(seqs),                       # [N, T, C]
        "seg": np.array(segs),
        "groups": np.array(groups),
        "y": {t: np.array(v, np.float32) for t, v in ys.items()},
    }


def agg_features(seq: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """XGBoost용 집계: 채널별 mean/std/min/max/slope + step one-hot."""
    N, T, C = seq.shape
    x = np.arange(T)
    feats = []
    mean = seq.mean(1)
    std = seq.std(1)
    mn = seq.min(1)
    mx = seq.max(1)
    # slope (선형회귀 기울기) 벡터화
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    slope = (seq * xc[None, :, None]).sum(1) / denom        # [N, C]
    feats = np.concatenate([mean, std, mn, mx, slope], axis=1)  # [N, 5C]
    me1 = (seg == "ME1").astype(np.float32)[:, None]
    return np.concatenate([feats, me1], axis=1)


def channel_norm_stats(seq_train: np.ndarray):
    """학습 시퀀스 [Ntr,T,C] → 채널별 mean/std (시간·샘플 축 통합)."""
    flat = seq_train.reshape(-1, seq_train.shape[-1])
    mu = flat.mean(0)
    sd = flat.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


if __name__ == "__main__":
    ds = make_dataset(96, 24)
    print("seq", ds["seq"].shape, "channels", len(CHANNELS))
    import collections
    print("groups", dict(collections.Counter(ds["groups"])))
    print("agg feature dim", agg_features(ds["seq"], ds["seg"]).shape[1])
    for t in TARGETS:
        print(t, "unique targets:", sorted(set(ds["y"][t].tolist())))
