# 식각 공정 결과 예측 — 전처리 전략 비교 실험

> **핵심 결론: 모델보다 전처리가 중요하다.**
> 복잡한 Feature Engineering보다 **공정 단계(ARC/ME1) 분리**라는
> 물리 특성 기반 전처리가 예측 성능에 가장 크게 기여했다.

---

## 1. 실험 배경

기존 식각 공정 데이터는 **Stable → ARC → ME1 → Over-Etch** 등 서로 다른
공정 단계가 **하나의 시계열 로그에 혼재**되어 있다.

또한 공정 결과 데이터는 **Remain PR / Poly Depth / Poly E/R** 와 같이
**공정 종료 후 측정되는 단일 결과값** 형태로 제공된다.

따라서 단순히 전체 로그를 학습하면 공정과 직접 관련 없는 구간까지 함께
학습되어 **예측 성능 저하**가 발생할 수 있다.

---

## 2. 실험 목적

본 실험은 *모델 성능 비교*가 아니라
**전처리 전략에 따라 예측 성능이 얼마나 달라지는지**를 확인하는 것이 목표다.

특히 다음 전처리 기법이 식각 결과 예측에 어떤 영향을 미치는지 분석했다.

- 공정 단계 분리 (ARC / ME1)
- 변화량 Feature (std, slope, max−min)
- 비율 Feature (Bias/Source, Gas/Pressure)
- 이상치 · 결측 처리

---

## 3. 실험 과정

| STEP | 내용 | 상세 |
|:--:|---|---|
| **1** | PROCESS 로그 파싱 | 42개 센서 추출 — RF Power / Gas Flow / Pressure / Temperature / ESC / He Cooling |
| **2** | 공정 단계 분리 | inline `$STEP` 마커로 Stable / ARC / ME1 / Over-Etch 구간 식별 |
| **3** | Window 생성 | Window = 192행, Stride = 16행 → **총 106개 Window** (ARC+ME1) |
| **4** | 전처리 전략 적용 | A~E (아래 표) |
| **5** | 모델 학습 | XGBoost / RandomForest / LightGBM |
| **6** | LORO 평가 | 2개 Run 학습 → 1개 Run 검증, 반복 (Leave-One-Run-Out) |

### 전처리 전략

| 전략 | 내용 |
|:--:|---|
| **A** | 기본 수치 변환 (센서 평균) |
| **B** | ARC / ME1 구간 분리 |
| **C** | 변화량 Feature (std, slope, max−min) |
| **D** | 비율 Feature (Bias/Source, Gas/Pressure) |
| **E** | 이상치 및 결측 처리 |

> **실험 규모:** 전처리 9세트 × 모델 3종 × 타깃 3종 = **81 실험** + 튜닝 + 민감도 분석

---

## 4. 실험 조건 (상세)

### 4.1 공정 조건 (DOE) — 3개 Run

ARC 단계는 3개 Run 모두 동일하며, **ME1 단계의 파워만 단일 인자로 변경**한 one-factor 실험이다.

**ARC 단계 (3 Run 공통)**

| Pressure | Source(RFT) | Bias(RFB) | CF4 | O2/He | Time | Temp |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 16 mT | 550 W | 40 W | 100 sccm | 20 sccm | 35 s | 65 ℃ |

**ME1 단계 (Run별 — 굵게 표시한 값이 변경 인자)**

| 항목 | Standard | Source power 300w | Bias power 140w |
|:--|:--:|:--:|:--:|
| Pressure (mT) | 4 | 4 | 4 |
| **Source RFT (W)** | 600 | **300** | 600 |
| **Bias RFB (W)** | 70 | 70 | **140** |
| Cl2 (sccm) | 50 | 50 | 50 |
| HBr (sccm) | 225 | 225 | 225 |
| O2/He (sccm) | 15 | 15 | 15 |
| Time (s) | 80 | 80 | 80 |

### 4.2 측정 결과 (정답 라벨, 공정 종료 후 측정)

| 타깃 | Standard | Source power 300w | Bias power 140w |
|:--|:--:|:--:|:--:|
| Remain PR (㎛) | 2.17 | 2.16 | 1.98 |
| Poly Depth (㎚) | 225 | 212 | 487 |
| Poly E/R (Å/min) | 1687.5 | 1590 | 3652.5 |

> Bias 파워를 70→140 W로 올리자 Poly Depth가 225→487 ㎚로 급증(E/R 1687→3652) —
> 이온 충격(bias) 증가가 식각률을 지배함을 보여준다.

### 4.3 데이터 규모

| Run | 로그 행수 | 공정 시간 | 샘플링 | Window 수 (ARC+ME1) |
|:--|--:|--:|:--:|--:|
| Standard | 1,786 | 446 s | ~4 Hz | 50 |
| Source power 300w | 1,250 | 312 s | ~4 Hz | 17 |
| Bias power 140w | 1,680 | 419 s | ~4 Hz | 39 |
| **합계** | | | | **106** |

### 4.4 입력 센서 (42 채널)

| 그룹 | 채널 수 | 내용 |
|:--|:--:|---|
| RF Power | 6 | Bias / Source 각 Setpoint · Forward · Reflect |
| Gas Flow | 24 | MFC 12채널 실측(ai) + 설정(ao) |
| Pressure | 1 | APC 챔버 압력 |
| Temperature | 3 | TEMP ch1~3 |
| ESC | 2 | 전압 모니터 · 설정 |
| He Cooling | 6 | 백사이드 He flow / pressure ×2 |

> **MFC↔가스 매핑(자동 도출):** MFC1=HBr, MFC4=Cl2, MFC7=CF4, MFC11=O2/He (3 Run 공통)

### 4.5 Window · 전처리 파라미터

- Window = **192행 (~48 s)**, Stride = **16행 (~4 s)**, 사용 구간 = **ARC + ME1**
- 전처리 세트: 누적 5종(A → A+B+C+D+E) + 단독 ablation 4종 = **9종**

### 4.6 모델 하이퍼파라미터

| 모델 | 기본값 (81실험 공통) | 튜닝 best |
|:--|---|---|
| XGBoost | n=300, depth=4, lr=0.05, subsample/colsample=0.9 | **n=400, depth=3, lr=0.1** |
| RandomForest | n=300 | n=200, depth=None, max_features=sqrt |
| LightGBM | n=300, depth=4, lr=0.05 | n=200, num_leaves=15, lr=0.1 |

- seed=0 고정. 전처리 효과를 분리하기 위해 81실험은 기본값으로 통일.

### 4.7 평가 프로토콜 (LORO)

- **Leave-One-Run-Out (3-fold):** 2개 Run 학습 → 남은 1개 Run의 Window 예측 →
  Run별 **median** 집계 → 정답 1개와 비교.
- **Baseline:** mean-predictor (학습 Run들의 타깃 평균).
- **지표:** run-level **MAPE / MAE** (주), window-level **R²** (보조).
- random split 금지 — Window 누설(leakage) 차단 (§6-(1) 참고).

---

## 5. 주요 결과

### 결과 1 — 누적 전처리: 어디까지 쌓는 게 좋은가 (Poly E/R, 3모델 평균)

전처리를 A → A+B → … → A+B+C+D+E로 누적하며 run-level MAPE를 측정.

| 전처리 | run-level MAPE | baseline | 판정 |
|:--|--:|--:|:--:|
| A (기본) | 71.0% | 64.7% | baseline 미달 |
| **A+B (step 분리)** | **69.7%** | 64.7% | 누적 중 최저 |
| A+B+C (+변화량) | 81.0% | 64.7% | 악화 |
| A+B+C+D (+비율) | 83.1% | 64.7% | 악화 |
| A+B+C+D+E (+이상치) | 97.4% | 64.7% | 최악 |

![누적 전처리 — Poly E/R](outputs/fig_cumulative_Poly_ER.png)

→ **B에서 최저점을 찍고 C·D·E를 더할수록 단조 악화.** (그림에서 XGBoost가 A+B에서 baseline 점선을 유일하게 하회)

---

### 결과 2 — 단독 기법별 기여도 (XGBoost, Poly E/R, baseline 대비)

각 기법을 A에 **하나씩만** 더했을 때의 순수 효과.

| 단독 기법 | baseline 대비 개선율 | 판정 |
|:--|--:|:--:|
| **A+B (step 분리)** | **+20.7%** | ✅ 최대 기여 |
| A+D (비율 Feature) | +9.5% | ◯ 일부 효과 |
| A+C (변화량 Feature) | +0.4% | △ 미미 |
| A+E (이상치 처리) | **−40.2%** | ⚠️ 역효과 |

![단독 기법 효과 — Poly E/R](outputs/fig_ablation_Poly_ER.png)

> 기여도 순위: **B(step분리) ≫ D(비율) > C(변화량) ≫ E(이상치처리)**

---

### 결과 3 — 사용 공정 구간 민감도 (A+B, XGBoost, Poly E/R)

Window를 어느 단계에서 뽑느냐에 따른 변화.

| 사용 구간 | Window 수 | run-level MAPE | 판정 |
|:--|--:|--:|:--:|
| **ME1 only** | 33 | **21.9%** | ✅ 최적 |
| ARC+ME1 (기본) | 106 | 29.4% | ◯ |
| ARC+ME1+Over-Etch | 146 | 99.9% | ✕ 붕괴 |
| 4구간 전체 | 157 | 100.8% | ✕ 붕괴 |

![구간 민감도](outputs/fig_sensitivity.png)

→ ME1(본 식각)에 집중할수록 정확. ARC(반사방지막 식각)·Over-Etch·Stable은 **노이즈로 작용**.

---

### 결과 4 — 모델 비교 (전처리 A+B, 튜닝 후, Poly E/R)

| 모델 | run-level MAPE | baseline(64.7%) 대비 |
|:--|--:|:--:|
| **XGBoost** | **29.0%** | ✅ 유일하게 하회 |
| RandomForest | 72.0% | ✕ |
| LightGBM | 84.0% | ✕ |

→ 동일 전처리에서도 **XGBoost만 baseline을 넘었다.** 튜닝으로 XGBoost는 A+B 기본 45.9% → 29.0%로 개선.

---

### 결과 5 — 타깃별 결과 (최종 베스트 파이프라인 적용)

**ME1 단독 Window + 튜닝 XGBoost** (n=400, depth=3, lr=0.1)

| 예측 대상 | MAPE | baseline | 개선율 |
|:--|--:|--:|--:|
| Poly E/R | **21.9%** | 59.5% | **+63%** |
| Poly Depth | **21.9%** | 59.5% | **+63%** |
| Remain PR | **3.5%** | 6.0% | **+41%** |

타깃별 누적 전처리 곡선:

| Poly Depth | Remain PR |
|:--:|:--:|
| ![Poly Depth](outputs/fig_cumulative_Poly_Depth.png) | ![Remain PR](outputs/fig_cumulative_Remain_PR.png) |

> Poly E/R = Poly Depth × 7.5 (식각시간 80초 고정)의 완전 선형 관계 → 두 타깃은 동일 예측문제(곡선도 동일).
> Remain PR은 값 범위가 좁아(1.98~2.17) 절대 MAPE는 낮지만 개선율은 +41%.

---

## 6. 실험 의의

### (1) "성능이 좋아 보이는 것"과 "실제로 맞히는 것"은 다르다 — 평가 설계가 결론을 만든다

같은 모델, 같은 무거운 전처리(A+B+C+D)를 쓰고 **평가 방식만 바꾸면** 결과가 정반대로 나온다.

| 검증 방식 | win R² | MAPE | 해석 |
|:--|--:|--:|---|
| Random window 분할 | **0.94** | **1.2%** | 거의 완벽해 보임 (착시) |
| LORO (run 단위 분할) | **−2.67** | **58.2%** | 사실상 무용 |

같은 Run에서 잘라낸 Window들은 **결과값(라벨)이 동일**하기 때문에, Window를 무작위로
섞어 나누면 모델은 식각 물리를 배우는 게 아니라 **"이 Window가 어느 Run에서 왔는지"를
암기**한다. 즉 흔히 쓰는 random split은 이 데이터에서 **누설(leakage)** 그 자체이며,
1.2% MAPE라는 화려한 숫자는 전부 허상이다.

> **이 실험의 모든 결론은 leakage를 차단한 LORO 평가 위에서만 성립한다.**
> random split로 평가했다면 "Feature를 많이 넣을수록 좋다"는 *틀린* 결론을 내렸을 것이다.
> 즉 전처리 전략보다 먼저 검증된 것은 **"검증 방식을 틀리면 어떤 전처리도 의미가 없다"** 는 점이다.

### (2) 정보는 데이터의 "양"이 아니라 "위치"에 있다

성능을 끌어올린 것은 Feature 개수가 아니라 **결과를 결정하는 물리 구간(ME1)에
신호를 집중**시킨 것이었다.

- ME1은 실제 Poly-Si를 깎는 **본(本) 식각 단계** → 결과값(Depth, E/R)을 직접 결정.
- ARC(반사방지막 식각) · Over-Etch · Stable은 Poly 결과와 **인과적으로 무관**.

따라서 이 구간들을 함께 넣으면 표본만 늘 뿐(33→157개) **조건과 무관한 분산(노이즈)이
주입**되어 Run 간 구별력이 무너진다(MAPE 21.9% → 100%). 이것은 단순한
"적을수록 좋다"가 아니라, **신호 대 잡음비(S/N)는 공정 인과구조를 반영해 구간을 고를 때
극대화된다**는 의미다.

### (3) "이상치 제거"가 해로웠던 진짜 이유 — 이상치가 곧 공정 지문(fingerprint)

통상 데이터 클리닝은 성능을 올리지만, 여기서는 −40%로 **가장 해로웠다**. 장비 trace에서
RF 점화 전후의 천이(transient), 순간 스파이크, 0-구간 같은 "이상치"는 잡음이 아니라
**각 Run의 공정 조건을 구별해 주는 신호**였기 때문이다. 통계 기준의 일괄 제거·보간은
바로 그 구별 정보를 지워 버린다.
→ **공정 데이터에서 클리닝은 통계가 아니라 공정 물리 기준으로 설계해야 한다.**

### (4) 이 데이터에서 "예측"의 본질 — 외삽이 아니라 조건 분별

트리 모델은 학습 타깃 범위 밖을 **외삽하지 못한다**. Run이 3개뿐이라 LORO의 매 fold는
"본 적 없는 조건"을 맞혀야 하므로, 절대 오차를 줄이는 문제라기보다 **Feature 공간에서
공정 조건 간 분리도(separability)를 얼마나 잘 보존하느냐**의 문제다. ME1 + 구간 분리가
이긴 이유도 바로 이 **조건 간 분리도를 극대화**했기 때문이다.

### (5) 실무 함의 — 노력을 어디에 쓸 것인가

- **고효율 투자:** ① 결과를 지배하는 단계로 trace를 분할(segmentation), ② DOE로 Run 수 확보.
- **저효율 투자:** 정교한 변화량·비율 Feature 파이프라인, 일괄 이상치 처리.
- **모델 선택은 그 다음 문제:** 동일 전처리에서 XGBoost만 baseline을 넘었고(RF·LGBM은 미달),
  전처리가 잘못되면 어떤 모델도 baseline을 못 넘었다.
- **타깃 결합 구조:** Poly E/R = Poly Depth × 7.5 (식각시간 고정) → 두 타깃은 사실상
  **하나의 자유도**. 측정/모델링 노력을 중복 투입할 필요가 없다.

> **결론:** "모델보다 전처리"라는 통념을 한 단계 구체화하면 —
> *식각 결과 예측의 성패는 **공정 인과구조에 맞춰 신호 구간을 고르고(ME1),
> 누설 없는 검증으로 그것을 증명하는 것**에 달려 있으며, Feature 양과 모델 선택은
> 그 다음 순위다.*

---

## 7. 한계 및 향후 과제

- **라벨 Run = 3개** (Standard / Source power 300w / Bias power 140w).
  Window(106개)는 센서 신호를 Feature로 쓰기 위한 것이며 독립 표본이 아니다 — 라벨은 Run당 1개.
- LORO 3-fold이므로 결과는 **통계적 유의성이 아닌 경향**으로 해석해야 한다.
- 트리 모델은 학습 타깃 범위 밖을 외삽하지 못해, held-out Run이 극단값이면 큰 오차가 불가피하다.
- **향후:** DOE로 Run 수를 20개 이상 확보하면 본 파이프라인을 그대로 재실행해
  정식 벤치마크로 승격할 수 있다.

---

## 8. 실행 방법

```bash
# 의존성 설치
pip install pandas openpyxl scikit-learn xgboost lightgbm matplotlib tabulate

# 81 실험 → 튜닝/민감도/누설실증 → 보고서·그림 생성
python3 run_experiment.py      # outputs/results.csv
python3 analysis_extra.py      # outputs/tuning.csv, sensitivity.csv, final.csv, leakage.csv
python3 src/report.py          # outputs/REPORT.md + 그림
```

### 프로젝트 구조
```
ald-etch/
├─ AIetch-process-data.xlsx   # 공정 조건 + 결과 정답 (3 run)
├─ Equipment Data/            # 장비 시계열 로그 (PROCESS_*_PM2)
├─ AI DATA/                   # SEM 단면 이미지 (측정 원본)
├─ src/
│  ├─ parse_process.py        # 로그 파싱 + step 분할
│  ├─ parse_labels.py         # 라벨 + MFC↔가스 자동매핑
│  ├─ windowing.py            # sliding window (192/16)
│  ├─ features.py             # 전처리 A~E 빌더
│  ├─ evaluate.py             # LORO CV + 모델 3종
│  └─ report.py               # 표·그림 생성
├─ run_experiment.py          # 81 실험
├─ analysis_extra.py          # 튜닝 + 민감도 + 최종베스트 + 누설실증
├─ architecture.md            # 실험 설계 문서
└─ outputs/                   # 결과 (REPORT.md, *.csv, *.png)
```

### 결과 산출물 (outputs/)

| 파일 | 대응 결과 | 내용 |
|:--|:--|---|
| `results.csv` | 결과 1·2 | 81실험 전체 수치 (전처리×모델×타깃) |
| `tuning.csv` | 결과 4 | 모델별 하이퍼파라미터 grid 결과 |
| `sensitivity.csv` | 결과 3 | 공정 구간별 성능 |
| `final.csv` | 결과 5 | 최종 베스트 파이프라인 타깃별 성능 |
| `leakage.csv` | 의의 (1) | random split vs LORO 누설 실증 |
| `fig_cumulative_*.png` | 결과 1·5 | 타깃별 누적 전처리 곡선 |
| `fig_ablation_*.png` | 결과 2 | 타깃별 단독 기법 기여도 |
| `fig_sensitivity.png` | 결과 3 | 공정 구간 민감도 |
| `REPORT.md` | 전체 | 위 결과 통합 보고서 |
