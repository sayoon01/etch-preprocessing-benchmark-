# 식각 공정 결과 예측 — 전처리 전략 비교 실험 설계

## 1. 목표
식각(Etch) 공정 결과(**Poly E/R, Poly Depth, Remain PR**)를 예측할 때
**어떤 전처리 전략이 가장 효과적인가**를 비교한다.
("어떤 모델이 좋은가"는 부차적 — 전처리 효과 분석이 주제.)

## 2. 데이터 실태 (설계의 전제)
| 항목 | 내용 |
|---|---|
| 라벨 run 수 | **3개**: Standard / Source power 300w / Bias power 140w |
| 정답(Excel) | run별 Remain PR(㎛), Poly Depth(㎚), Poly E/R(Å/min) — center 1점만 |
| Feature 소스 | `Equipment Data/*/PROCESS_*_PM2` 장비 시계열 로그 (**43컬럼, ~4Hz**) |
| step 구조 | 로그 내 inline `$STEP` 마커로 4구간(stable/ARC/ME1/over-etch) 분할 |
| AI DATA | SEM 단면 이미지(.bmp) — Excel 측정의 원본, 본 실험 feature엔 미사용 |

### 정답값
| run | Remain PR(㎛) | Poly Depth(㎚) | Poly E/R(Å/min) |
|---|---|---|---|
| Standard | 2.17 | 225 | 1687.5 |
| Source 300w | 2.16 | 212 | 1590 |
| Bias 140w | 1.98 | 487 | 3652.5 |

> ⚠️ **n=3.** 본 연구는 벤치마크가 아니라 **파일럿(feasibility) + 전처리 효과 경향 분석**이다.
> window 확장은 센서 신호를 feature로 쓰기 위한 것이며, 라벨은 여전히 run당 1개임을 보고서에 명시한다.

## 3. 센서 컬럼 그룹 (43)
- 파워: `aoRFBIAS_Setpoint, aiRFBIAS_ForwardPwr, aiRFBIAS_ReflectPwr, aoRFSOURCE_Setpoint, aiRFSOURCE_ForwardPwr, aiRFSOURCE_ReflectPwr`
- 가스: `aiMFC_FLOW_01~12`(실측), `aoMFC_FLOW_01~12`(설정)
- 척/쿨링: `aiESC_VOLTAGE_MONITORING, aoESC_VOLTAGE_SETPOINT, avHE_*/aiHE_*`(백사이드 He flow/pressure ×2)
- 압력/온도: `aiAPC_CurrentPressure, aiTEMP_PV1ch1~3`

### MFC ↔ 가스 매핑 (RECIPE 설정값 ↔ Excel 가스 컬럼 자동 도출) — **확정**
`MFC1=HBr, MFC4=Cl2, MFC7=CF4, MFC11=O2/He` (3 run 공통, parse_labels.py 에서 자동 도출)

## 4. 파이프라인 (실제 실행 기준)
```
[1] PROCESS 로그 파싱 (timestamp + 42센서, inline $STEP 구간 라벨)
[2] step 분할 (stable/ARC/ME1/over-etch) — 기본은 ARC+ME1 만 사용
[3] Excel 라벨 결합 (run별 타깃) + 가스 매핑 자동 도출
[4] sliding window 생성: window=192행(~48s), stride=16행(~4s)
[5] window 집계 → feature (전략 A~E)
[6] LORO CV 평가
```
- window/stride: **192 / 16**, 사용구간 **ARC+ME1** → **총 106 window**
  (Standard 50 / Source 17 / Bias 39), 모든 구간 ≥1 window.
- window 중첩으로 window들은 상관됨 → **run-level LORO만 진짜 평가**(window-level R²은 참고).

## 5. 전처리 전략 A~E (시계열 window 버전, 누적식)
| 전략 | 추가 내용 | feature 예시 |
|---|---|---|
| **A** 단순 수치 변환 | step 무시, run 전체 평균 1벡터 | 43센서 mean |
| **B** Step 분리 | ARC/ME1 구간 feature 분리 | `ARC_*`, `ME1_*` |
| **C** 변화량 feature | 동적 특성 | std, slope, min/max, ramp-up time, SP대비 편차(Forward−Reflect, ao−ai) |
| **D** 비율 feature | 식각화학 도메인 | Cl2/HBr, 총가스량, Bias/Source 파워비, 파워/압력비, He백사이드/APC비 |
| **E** 이상치·결측 | 데이터 품질 | RF-off·stabilization(0값) 제거, steady-state만 선별, Reflect 스파이크 제거, dropout 보간 |

- 메인: `A → A+B → A+B+C → A+B+C+D → A+B+C+D+E` 누적 5세트
- 보조: `A+B만/A+C만/A+D만/A+E만` 단독 ablation 4세트

## 6. 실험 매트릭스
```
전처리 9세트 × 모델 3(XGB/RF/LGBM) × 타깃 3(E/R, Depth, Remain PR) = 81 실험
+ baseline: mean-predictor(LOO 평균)
```
- 하이퍼파라미터는 전 실험 고정(전처리 효과만 분리), seed 고정.
- MLP 제외: 소량 tabular에서 트리계가 정석, MLP는 데이터 부족으로 비교 불공정.

## 7. 평가
- **LORO CV (3 fold)**: 2 run 학습 → 남은 run window 예측 → run별 median 집계 → run-level 1예측 vs 정답.
- 주지표: run-level **MAE / MAPE** (mean-predictor 대비 상대 개선율). 보조: window-level R².
- random split 금지 (GroupKFold, group=run) — leakage 방지가 핵심 기여.

## 8. 보고서 구조
1. 데이터·실험 개요  2. 방법론(window+LORO, leakage 방지)
3. 전처리 누적 단계별 성능 곡선(메인)  4. 단독 ablation 기여도 순위
5. 모델 간 차이(부차)  6. 해석: 식각 예측에 핵심인 전처리
7. 한계(n=3 경향) & 향후(DOE로 run 확보)

## 9. 코드 구조
```
ald-etch/
├─ src/
│  ├─ parse_process.py   # PROCESS 로그 → DataFrame + step 분할
│  ├─ parse_labels.py    # Excel → run별 타깃 + 가스 매핑 자동도출
│  ├─ windowing.py       # step 내 sliding window (192/16, ARC+ME1)
│  ├─ features.py        # 전처리 A~E feature 빌더 (토글)
│  ├─ evaluate.py        # LORO CV + 모델 3종 + mean-baseline
│  └─ report.py          # 결과 표/그림 생성
├─ outputs/              # 지표 csv, 그림, REPORT.md
├─ run_experiment.py     # 81 실험 일괄 실행 → outputs/results.csv
└─ analysis_extra.py     # 튜닝 + segment 민감도 + 최종베스트
```
실행 순서: `python3 run_experiment.py && python3 analysis_extra.py && python3 src/report.py`

## 10. 실제 수행 내역 & 핵심 결과 (요약)
실행한 실험:
1. **기본 81실험** — 전처리 9세트 × 모델 3 × 타깃 3 (`run_experiment.py`)
2. **하이퍼파라미터 튜닝** — 모델별 grid, 전처리 A+B, target Poly_ER (`analysis_extra.py`)
3. **segment 민감도** — ME1 / ARC+ME1 / +overetch / 4구간 전체
4. **최종 베스트 파이프라인** — ME1 단독 + 튜닝 XGBoost

핵심 결과 (자세한 표·그림은 `outputs/REPORT.md`):
| 발견 | 내용 |
|---|---|
| 가장 효과적 전처리 | **B(step 분리)** 단독 +21% > D(비율) +9.5% > C(변화량) +0.4% ≫ **E(이상치처리) −40%** |
| 누적 효과 | C/D/E 더할수록 악화 (n=3 과적합 → 외삽 실패) |
| 구간 선택 | **ME1 단독이 최고**(MAPE 21.9), ARC·overetch·stable 은 노이즈 |
| 모델 | **XGBoost 압도적**, 튜닝으로 Poly E/R MAPE 46→22% |
| 최종 베스트 | ME1단독+튜닝XGBoost: Poly E/R **21.9%**(base 59.5, +63%), Remain PR **3.5%**(+41%) |
| 부수 발견 | Poly E/R = Poly Depth × 7.5 (식각시간 80s 고정) → 동일 예측문제 |
