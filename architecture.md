# 식각 공정 결과 예측 — 모델별 정확도 비교 실험 설계

## 1. 목표
식각(Etch) 공정 결과(**Poly E/R, Remain PR**)를 예측할 때
**어떤 모델이 가장 정확한가**를 비교한다. 비교 모델: **XGBoost / LSTM / PatchTST**.
각 모델의 **최적 window 크기를 따로 탐색**한다.
("어떤 전처리가 좋은가"는 선행 연구로 정리됨 → §9 참고. 본 실험의 주제는 모델 비교.)

> 이전 버전은 "전처리 전략 비교"가 주제였으나, **더 중요한 결론은 모델별 정확도**라는 판단에 따라 주제를 전환함.
> 모델: XGBoost/RandomForest/LightGBM → **XGBoost/LSTM/PatchTST**.

## 2. 데이터 실태 (설계의 전제)
| 항목 | 내용 |
|---|---|
| 라벨 run 수 | **3개**: Standard / Source power 300w / Bias power 140w |
| 정답(Excel) | run별 Poly E/R(Å/min), Remain PR(㎛) — center 1점만 (Poly Depth = E/R÷7.5 동일문제라 생략) |
| Feature 소스 | `Equipment Data/*/PROCESS_*_PM2` 장비 시계열 로그 (~4Hz) |
| 사용 구간 | inline `$STEP` 마커로 분할 후 **ARC+ME1** 고정 (3모델 공통, 샘플 확보) |

### 정답값
| run | Poly E/R(Å/min) | Remain PR(㎛) |
|---|---|---|
| Standard | 1687.5 | 2.17 |
| Source 300w | 1590 | 2.16 |
| Bias 140w | 3652.5 | 1.98 |

> ⚠️ **n=3.** 라벨 run이 3개뿐 → 결과는 통계적 유의성이 아닌 **경향**. 특히 극단값(Bias run)을
> held-out 하는 fold는 **모든 모델이 외삽 실패**(트리·시퀀스 공통). window 확장은 신호용이며 라벨은 run당 1개.

## 3. 입력 채널 (12)
딥러닝 과적합 억제 위해 핵심 12채널만 (3모델 공통):
- 파워: `aiRFBIAS_ForwardPwr/ReflectPwr`, `aiRFSOURCE_ForwardPwr/ReflectPwr`
- 압력: `aiAPC_CurrentPressure`
- 가스: `aiMFC_FLOW_01(HBr)/04(Cl2)/07(CF4)/11(O2He)`
- 온도: `aiTEMP_PV1ch1~3`

## 4. 모델별 입력 형태 (핵심 차이)
| 모델 | 입력 | 구성 |
|---|---|---|
| **XGBoost** | 윈도우 **집계 feature** (채널별 mean/std/min/max/slope + step one-hot, 61차원) | 트리, 정규화 불필요 |
| **LSTM** | **원시 시퀀스** `[T, 12]` | 1층 LSTM(hidden 48)+dropout→회귀헤드, y표준화 |
| **PatchTST** | **patch 시퀀스** `[T, 12]` | channel-independent patch(len16/stride8)+Transformer(2층)→평균풀→헤드 |

- 정규화: 채널 z-score를 **학습 run에서만** 산출(누설 차단). y는 DL에서 학습 통계로 표준화 후 역변환.

## 5. window 크기 튜닝 (요청사항)
- grid: **{48, 64, 96, 128, 192, 256}**, stride = window//4
- window별 총 샘플 수(ARC+ME1): 48→213 / 64→154 / 96→96 / 128→67 / 192→37 / 256→22
- **모델 × window × 타깃 × seed**로 LORO MAPE 측정 → **모델별 argmin window** 선택
- 산출: 모델별 "window↔MAPE" 곡선(최적점 ★ 표시)

## 6. 평가 프로토콜
- **LORO CV (3-fold)**: 2 run 학습 → 남은 run window 예측 → run별 **median** 집계 → run-level MAPE/MAE
- baseline: mean-predictor (학습 run 타깃 평균)
- 딥러닝 변동 대응: **seed 5회** → mean±std (XGBoost는 결정적이라 1회)
- 환경: torch 2.11+cu128, **RTX 5090** GPU

## 7. 실제 결과 (요약)

### Poly E/R — 모델별 최적 window & 성능 (run-level MAPE, baseline 63.7)
| 모델 | 최적 window | MAPE | 비고 |
|---|---|---|---|
| **LSTM** 🥇 | **48** | **21.1 ± 0.5** | 작은 window(샘플 213) 최적, 변동 작음 |
| PatchTST | 48 | 42.5 ± 3.0 | 작은 window 선호 |
| XGBoost | 256 | 43.8 | 큰 window(집계 안정) 최적 |

### Remain PR (baseline 6.3)
| 모델 | 최적 window | MAPE |
|---|---|---|
| **LSTM** | 64 | **3.5 ± 0.1** |
| PatchTST | 48 | 4.9 |
| XGBoost | 256 | 5.0 |

**핵심 발견:**
- **모델마다 최적 window가 다름** — 시퀀스 모델(LSTM/PatchTST)은 작은 window(샘플↑), 트리(XGBoost)는 큰 window(집계 안정).
- 두 타깃 모두 **LSTM이 최고 정확도**, 모든 모델이 baseline은 상회.
- 단, LSTM의 낮은 MAPE는 외삽 가능한 2개 fold를 잘 맞힌 결과이며, 극단 run fold는 전 모델 실패(n=3 한계).

## 8. 코드 구조
```
models/
├─ dataset.py        # 윈도우→XGB집계feature & DL시퀀스[N,T,C], 채널정규화
├─ xgb_model.py      # XGBoost 회귀
├─ lstm_model.py     # PyTorch LSTM 회귀
├─ patchtst.py       # compact PatchTST 회귀
├─ model_eval.py     # LORO + mean-baseline
├─ tune_window.py    # 모델×window grid 실행 → outputs/model_results.csv
├─ report_models.py  # window 튜닝 곡선 + 모델 비교 + MODEL_REPORT.md
└─ outputs/          # model_results.csv, MODEL_REPORT.md, fig_*.png
```
실행: `cd models && python3 tune_window.py && python3 report_models.py`

## 9. 부속: 선행 전처리 연구 (참고)
주제 전환 전, 동일 데이터로 전처리 전략(A~E) 비교를 수행해
"**ME1 구간 선별 + step 분리**가 가장 효과적, 과한 feature engineering은 역효과"를 확인함.
상세: `outputs/REPORT.md`, 코드 `src/` + `run_experiment.py` + `analysis_extra.py`.

## 10. 한계 & 향후
- 라벨 run 3개 → LORO 3-fold, 경향 수준. 외삽 불가로 극단 run fold는 전 모델 큰 오차.
- window 작을수록 샘플↑이나 중첩↑(상관) → run-level 평가로 보정.
- **권장:** DOE로 run>20 확보 시 딥러닝(LSTM/PatchTST)의 진짜 우위를 정식 검증 가능.
