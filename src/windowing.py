"""Sliding window 생성.

각 run 시계열을 step 구간(seg) 내부에서 window=192행, stride=32행으로 자른다.
window 가 ARC↔ME1 경계를 넘지 않도록 step 내부에서만 슬라이딩(물리적으로 타당).
strategy A 는 step 정보를 feature 로 쓰지 않을 뿐, window 자체는 동일 기반을 공유한다.
"""
from __future__ import annotations
import pandas as pd

WIN = 192
STRIDE = 16  # ARC+ME1 만 사용 시 표본 확보 위해 16 (window 중첩 → run-level LORO 로 평가)
# stable/overetch 는 식각 본구간이 아니므로 기본 제외 (ARC, ME1 만 사용)
USE_SEGS = ("ARC", "ME1")


def make_windows(df: pd.DataFrame, win: int = WIN, stride: int = STRIDE,
                 use_segs=USE_SEGS) -> list[dict]:
    """run DataFrame → window 리스트.

    각 원소: {'seg_name', 'start', 'rows'(window 행 DataFrame)}.
    """
    windows = []
    for seg_name, g in df.groupby("seg_name", sort=False):
        if seg_name not in use_segs:
            continue
        g = g.reset_index(drop=True)
        n = len(g)
        if n < win:
            # 구간이 window 보다 짧으면 구간 전체를 1 window 로
            windows.append({"seg_name": seg_name, "start": 0, "rows": g})
            continue
        for s in range(0, n - win + 1, stride):
            windows.append({"seg_name": seg_name, "start": s,
                            "rows": g.iloc[s:s + win].reset_index(drop=True)})
    return windows


if __name__ == "__main__":
    import parse_process as pp
    for label, path in pp.discover_runs().items():
        df = pp.load_process(path)
        w = make_windows(df)
        by_seg = {}
        for x in w:
            by_seg[x["seg_name"]] = by_seg.get(x["seg_name"], 0) + 1
        print(f"[{label}] windows={len(w)} {by_seg}")
