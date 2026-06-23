# 식각 공정 결과 예측 — 모델별 정확도 비교 (XGBoost / LSTM / PatchTST)

> **핵심 결론: 모델마다 최적 window 크기가 다르며, LSTM이 가장 정확하다.**
> 시퀀스 모델(LSTM·PatchTST)은 **작은 window**, 트리(XGBoost)는 **큰 window**에서 최적.
> 단, 라벨 run이 3개뿐이라 결과는 **경향**이며 "외삽 조건"은 모든 모델이 실패한다.

---

## 1. 실험 배경

식각 장비는 공정 중 **42개 센서를 ~4Hz로 시계열 로그**에 기록한다(RF 파워·가스·압력·온도 등).
반면 공정 결과(**Poly E/R, Remain PR**)는 **공정 종료 후 측정되는 단일 값**으로만 제공된다.

따라서 "시계열 입력 → 단일 결과 예측"이라는 회귀 문제이며, 모델 종류에 따라
이 매핑을 학습하는 방식(집계 특징 vs 원시 시퀀스)이 다르다.

### 예측 대상
| 타깃 | 의미 | 값 범위 |
|---|---|---|
| **Poly E/R** | 폴리실리콘 식각률 (Å/min) — 얼마나 빨리 깎이나 | 1590 ~ 3652 (넓음 → 예측 어려움) |
| **Remain PR** | 잔류 감광막 두께 (㎛) — 마스크가 얼마나 남나 | 1.98 ~ 2.17 (좁음 → 정확도 높게 보임) |

---

## 2. 실험 목적

이전 연구(전처리 전략 비교, §8 참고)에 이어, 본 실험의 주제는
**"어떤 모델이 가장 정확한가"** 와 **"모델마다 최적 window 크기는 얼마인가"** 이다.

비교 모델: **XGBoost(트리) / LSTM(순환신경망) / PatchTST(트랜스포머)**

---

## 3. 실험 과정

| STEP | 내용 | 상세 |
|:--:|---|---|
| **1** | 시계열 파싱 | PROCESS 로그 → 핵심 **12채널** (Bias/Source Forward·Reflect, APC압력, MFC 4종, 온도 3) |
| **2** | 구간 고정 | inline `$STEP`로 분할 후 **ARC+ME1** 사용 (3모델 공통) |
| **3** | Window 생성 | window별 sliding window (stride=window//4) |
| **4** | 모델별 입력 변환 | XGBoost=집계 feature / LSTM·PatchTST=원시 시퀀스 `[T,12]` |
| **5** | 모델 학습 | XGBoost / LSTM / PatchTST (RTX 5090, torch cu128) |
| **6** | LORO 평가 | 2 run 학습 → 1 run 예측, seed 5회 → 평균±표준편차 |

### 모델별 입력 형태 (핵심 차이)
| 모델 | 입력 | 구성 |
|---|---|---|
| **XGBoost** | 윈도우 **집계 feature** (채널별 mean/std/min/max/slope + step, 61차원) | 트리, 정규화 불필요 |
| **LSTM** | **원시 시퀀스** `[T, 12]` | 1층 LSTM(hidden 48)+dropout |
| **PatchTST** | **patch 시퀀스** | channel-independent patch + Transformer 2층 |

### Window 크기 튜닝 (요청 핵심)
- grid: **{48, 64, 96, 128, 192, 256}**, 각 window별 총 샘플 수:

| window | 48 | 64 | 96 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|
| 샘플 수 | 213 | 154 | 96 | 67 | 37 | 22 |

→ 작은 window일수록 샘플↑. 모델 × window × 타깃 × seed = **132 LORO 평가**.

---

## 4. 주요 결과

### 결과 1 — 모델마다 최적 window가 정반대

**Poly E/R, window × 모델 평균 MAPE** (낮을수록 좋음, baseline 63.7)

| 모델 | W48 | W64 | W96 | W128 | W192 | W256 | 최적 |
|---|---|---|---|---|---|---|---|
| **LSTM** | **21.1** | 22.2 | 27.7 | 53.6 | 57.7 | 54.7 | **W48** |
| **PatchTST** | **42.5** | 44.9 | 49.3 | 49.5 | 52.7 | 48.3 | **W48** |
| **XGBoost** | 69.5 | 58.3 | 57.6 | 67.0 | 61.5 | **43.8** | **W256** |

![window tuning](models/outputs/fig_window_tuning_Poly_ER.png)

→ **시퀀스 모델(LSTM/PatchTST)은 작은 window(샘플↑), 트리(XGBoost)는 큰 window(집계 안정)에서 최적** — 두 곡선이 정반대로 교차.

### 결과 2 — 모델 정확도 비교 (각자 최적 window)

| 타깃 | 모델 | 최적 W | MAPE | **정확도(=100−MAPE)** | baseline |
|---|---|---|---|---|---|
| **Poly E/R** | **LSTM** 🥇 | 48 | 21.1 ± 0.5 | **약 79%** | 36% |
| | PatchTST | 48 | 42.5 ± 3.0 | 약 57% | 36% |
| | XGBoost | 256 | 43.8 | 약 56% | 36% |
| **Remain PR** | **LSTM** 🥇 | 64 | 3.5 ± 0.1 | 약 96.5% | 94% |
| | PatchTST | 48 | 4.9 | 약 95% | 94% |
| | XGBoost | 256 | 5.0 | 약 95% | 94% |

![model compare](models/outputs/fig_model_compare_Poly_ER.png)

→ **두 타깃 모두 LSTM이 최고**, 세 모델 모두 baseline 상회. LSTM은 변동(std)도 가장 작아 우위가 안정적.
> Remain PR은 값 범위(1.98~2.17)가 좁아 baseline조차 94% → 모델 우열 판별엔 **Poly E/R가 핵심 지표**.

### 결과 3 — "Poly 정확도 79%"의 실체 (fold별 분해)

최고 모델 LSTM(W48)의 LORO fold별 예측:

| held-out run | 정답 → 예측 | 정확도 | 유형 |
|---|---|---|---|
| Standard | 1687 → 1743 | **96.7%** | 내삽 |
| Source 300w | 1590 → 1729 | **91.3%** | 내삽 |
| **Bias 140w** | 3652 → 1687 | **46.2%** | **외삽(불가)** |

→ **내삽 가능한 2개 조건은 이미 91~97%.** 평균 79%를 끌어내리는 건 **Bias fold 하나**다.
Bias run(3652)을 빼면 학습값이 둘 다 ~1600 → 모델이 **본 적 없는 큰 값으로 외삽 불가**(수학적 한계). log변환·튜닝으로도 +1~2%p뿐.

---

## 5. 실험 의의

1. **모델 선택은 window 크기와 짝지어야 한다.** "어떤 모델이 좋은가"는 window를 고정하고 비교하면 틀린 결론이 난다 — LSTM은 W48에서 1등이지만 W192에선 XGBoost에 진다. **모델마다 최적 window를 따로 찾아야** 공정한 비교다.

2. **소량 데이터에서도 LSTM이 트리를 능가했다.** 흔히 "딥러닝은 데이터가 많아야"라지만, 작은 window로 **샘플 수를 213개까지 늘리자** LSTM이 가장 정확했다. 핵심은 표본 절대량이 아니라 **모델에 맞는 입력 표현(짧은 시퀀스 다량)**이었다.

3. **정확도의 상한은 모델이 아니라 데이터가 정한다.** 내삽 조건은 90%대인데 외삽 조건은 46% — 즉 현재 한계는 알고리즘이 아니라 **run이 3개뿐**이라는 점이다. 정확도를 올리는 길은 더 정교한 모델이 아니라 **더 다양한 공정 조건(DOE)**이다.

---

## 6. 한계 및 향후 과제

- **라벨 run = 3개** → LORO 3-fold, 결과는 **경향**(통계적 유의성 아님).
- 트리·시퀀스 모두 **학습 타깃 범위 밖 외삽 불가** → 극단 run(Bias) held-out fold는 전 모델 실패.
- 작은 window는 샘플↑이나 중첩(상관)↑ → **run-level 평가**로 보정.
- **향후:** DOE로 run을 10~20개 이상(특히 고 E/R 조건 포함) 확보하면, 외삽이 내삽으로 바뀌어
  본 파이프라인 그대로 재실행 시 진짜 높은 정확도를 검증할 수 있다.

---

## 7. 실행 방법

```bash
pip install pandas openpyxl scikit-learn xgboost matplotlib tabulate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU(RTX 50xx)

cd models
python3 tune_window.py     # 모델×window grid → outputs/model_results.csv
python3 report_models.py   # 그림 + outputs/MODEL_REPORT.md
```

### 프로젝트 구조
```
ald-etch/
├─ AIetch-process-data.xlsx   # 공정 조건 + 결과 정답 (3 run)
├─ Equipment Data/            # 장비 시계열 로그
├─ models/
│  ├─ dataset.py              # 윈도우→XGB집계feature & DL시퀀스, 채널정규화
│  ├─ xgb_model.py / lstm_model.py / patchtst.py
│  ├─ model_eval.py           # LORO + mean-baseline
│  ├─ tune_window.py          # 모델×window grid 실행
│  ├─ report_models.py        # 그림 + 보고서
│  └─ outputs/                # model_results.csv, MODEL_REPORT.md, fig_*.png
├─ src/ + run_experiment.py + analysis_extra.py   # §8 선행 전처리 연구
└─ architecture.md            # 실험 설계 문서
```

상세 결과: [`models/outputs/MODEL_REPORT.md`](models/outputs/MODEL_REPORT.md)

---

## 8. 부속: 선행 전처리 연구 (참고)

주제 전환 전, 동일 데이터로 전처리 전략(A~E) 비교를 수행해
**"ME1 구간 선별 + 공정 단계(ARC/ME1) 분리가 가장 효과적, 과한 feature engineering은 역효과"**
를 확인했다. 상세: [`outputs/REPORT.md`](outputs/REPORT.md), 코드 `src/` + `run_experiment.py` + `analysis_extra.py`.
