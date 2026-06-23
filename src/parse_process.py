"""PROCESS 로그 파서.

장비 시계열 로그(`Equipment Data/*/PROCESS_*_PM2`)를 읽어
- 43개 센서 컬럼 + timestamp 를 DataFrame 으로 반환
- 로그 내부의 inline `$STEP` 마커로 step 구간(seg index)을 라벨링

run 폴더명에 한글/공백이 있으므로 glob 로 안전하게 처리.
"""
from __future__ import annotations
import glob
import os
import re
import datetime as dt
import pandas as pd

EQUIP_DIR_DEFAULT = "Equipment Data"

# 4개 step 구간의 의미 (inline $STEP 마커 기준, 0=첫 마커 이전)
SEG_NAMES = {0: "stable", 1: "ARC", 2: "ME1", 3: "overetch"}


def _find_process_file(run_dir: str) -> str:
    cands = [p for p in glob.glob(os.path.join(run_dir, "PROCESS_*_PM2"))
             if not p.endswith("_ALL")]
    if not cands:
        raise FileNotFoundError(f"PROCESS_*_PM2 없음: {run_dir}")
    return cands[0]


def _parse_ts(s: str) -> dt.datetime:
    return dt.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S")


def load_process(run_dir: str) -> pd.DataFrame:
    """run 폴더 → 시계열 DataFrame (컬럼: timestamp, seg, <43 센서>)."""
    path = _find_process_file(run_dir)
    lines = open(path, encoding="latin1").read().splitlines()
    header = lines[0].split("\t")
    sensor_cols = header[1:]  # 첫 컬럼은 timestamp

    rows, seg = [], 0
    for ln in lines[1:]:
        if ln.startswith("$STEP"):
            seg += 1
            continue
        if not ln or not ln[0].isdigit():
            continue
        parts = ln.split("\t")
        ts = _parse_ts(parts[0])
        vals = [float(x) for x in parts[1:1 + len(sensor_cols)]]
        rows.append((ts, seg, *vals))

    df = pd.DataFrame(rows, columns=["timestamp", "seg", *sensor_cols])
    df["seg_name"] = df["seg"].map(lambda s: SEG_NAMES.get(s, f"seg{s}"))
    return df


def discover_runs(equip_dir: str = EQUIP_DIR_DEFAULT) -> dict[str, str]:
    """run 식별자(영문 라벨) → 폴더 경로.

    폴더명 예: '15시46분29초 Source power 300w' → 'Source power 300w'
    """
    runs = {}
    for d in sorted(os.listdir(equip_dir)):
        full = os.path.join(equip_dir, d)
        if not os.path.isdir(full):
            continue
        # 앞의 'HH시MM분SS초 ' 시각 프리픽스 제거
        label = re.sub(r"^\d+시\d+분\d+초\s*", "", d).strip()
        runs[label] = full
    return runs


SENSOR_COLS = None  # 첫 로드시 채움


if __name__ == "__main__":
    runs = discover_runs()
    print("발견된 run:", list(runs))
    for label, path in runs.items():
        df = load_process(path)
        seg_counts = df.groupby("seg_name", sort=False).size().to_dict()
        print(f"\n[{label}] rows={len(df)}, 센서컬럼={len(df.columns) - 3}")
        print("  seg별 행수:", seg_counts)
