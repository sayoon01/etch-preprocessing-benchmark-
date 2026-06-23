# 데이터 기술서 (Data Description)

본 문서는 식각(Etch) 공정 결과 예측 프로젝트에서 사용한 **3종 데이터 소스**의
구조 · 내용 · 특성 · 측정 기준을 정리한다.

| 데이터 소스 | 역할 | 형식 | 비고 |
|:--|:--|:--|:--|
| `AIetch-process-data.xlsx` | **공정 조건 + 결과 정답(라벨)** | Excel (표) | 모델의 정답값 |
| `Equipment Data/` | **공정 중 장비 센서 시계열** | 로그 (텍스트) | 모델의 입력(feature) 원천 |
| `AI DATA/` | **결과 측정 원본 이미지** | SEM 이미지(.bmp)+메타(.txt) | 정답값의 측정 근거 |

세 소스는 모두 **동일한 3개 실험 조건**(Standard / Source power 300w / Bias power 140w)으로
1:1 연결된다.

---

## 0. 실험 개요 (공통 배경)

- **공정:** Poly-Si(폴리실리콘) 게이트 식각, 레시피 `DB POLY(ARC-ME1)`, 챔버 PM2
- **시료:** 8″ Coupon (웨이퍼 쿠폰), Cathode/Wall 측정
- **공정 단계:** `Stable`(안정화) → `ARC`(반사방지막 식각) → `ME1`(주 Poly 식각) → `Over-Etch`
- **DOE(실험계획):** 단일 인자 변경 (one-factor)
  - **Standard** — 기준 조건
  - **Source power 300w** — ME1의 Source RF를 600→**300 W**로 변경
  - **Bias power 140w** — ME1의 Bias RF를 70→**140 W**로 변경
- **측정일:** 장비·SEM 로그 기준 2026-03-25 (xlsx의 Date 필드는 `260224`로 기재)

---

## 1. `AIetch-process-data.xlsx` — 공정 조건 & 결과 정답

### 1.1 개요
- 시트 1개: **`Poly-Si WF`**, 크기 66 × 18 (대부분 빈 셀, 3개 블록에 데이터 집중)
- 구조: 상단 `Before Etch` 표제 + **실험 조건별 블록 3개**
  (`Standard`, `Source power`, `Bias power`)
- 각 블록 = ① 레시피 조건 표 + ② 결과 측정 표

### 1.2 레시피 조건 항목 (각 블록의 ARC / ME1 두 행)

| 컬럼 | 의미 | 단위 |
|:--|:--|:--|
| Date | 공정 일자 | YYMMDD |
| Step | 공정명 (Poly) / 세부 단계 (ARC, ME1) | - |
| WF | 시료 형태 (Coupon) | - |
| Pressure | 챔버 압력 | mT |
| RFT | **Source(Top) RF 파워** | W |
| RFB | **Bias(Bottom) RF 파워** | W |
| Cl2 / HBr / He / CF4 / O2/He | 가스 유량 | sccm |
| B He | 백사이드 He 압력(웨이퍼 쿨링) | Torr |
| Time | 단계 시간 | sec |
| Temp | 온도 | ℃ |
| Chuck | 척 크기 (8″) | - |

### 1.3 결과 측정 항목 (예측 대상)

| 항목 | 의미 | 단위 | 측정 여부 |
|:--|:--|:--|:--|
| **Remain PR** | 식각 후 잔류 PhotoResist 두께 | ㎛ | ✅ |
| **Poly Depth** | Poly 식각 깊이 | ㎚ | ✅ |
| **Poly E/R** | Poly 식각률 (Etch Rate) | Å/min | ✅ |
| Taper Angle | 식각 측벽 각도 | ˚ | ✗ (미측정) |
| Sel to PR | PR 대비 선택비 | - | ✗ (미측정) |
| Uniformity | 균일도 | % | ✗ (0) |

- **측정 위치:** 결과 표에 T/C/B/L/R(상/중/하/좌/우) 컬럼이 있으나 **C(center)만 측정**.
- **타깃 결합 관계:** `Poly E/R = Poly Depth × 7.5` (ME1 식각시간 80초 고정의 산술 결과)
  → E/R과 Depth는 사실상 **동일한 예측 문제(1 자유도)**.

### 1.4 실제 값

| 항목 | Standard | Source 300w | Bias 140w |
|:--|--:|--:|--:|
| Remain PR (㎛) | 2.17 | 2.16 | 1.98 |
| Poly Depth (㎚) | 225 | 212 | 487 |
| Poly E/R (Å/min) | 1687.5 | 1590 | 3652.5 |

> Bias 파워 ↑(70→140W) 시 Depth가 2배 이상 급증 — 이온 충격이 식각률을 지배.

---

## 2. `Equipment Data/` — 장비 센서 시계열 로그

### 2.1 폴더 구성
실험 조건별 폴더 3개 (`HH시MM분SS초 <조건명>` 형식):
- `15시28분07초 Standard`
- `15시46분29초 Source power 300w`
- `15시59분30초 Bias power 140w`

### 2.2 폴더 내 파일 (조건당)

| 파일 | 내용 | 핵심 활용 |
|:--|:--|:--|
| **`PROCESS_xx_PM2`** | **공정 중 센서 시계열 (메인 데이터)** | ✅ feature 원천 |
| `PROCESS_xx_PM2_ALL` | 단계(STEP) 전환 타임스탬프 요약 | step 분할 |
| `RECIPE_xx_PM2` | 레시피 원본 (단계별 setpoint 전체) | 가스 매핑·조건 확인 |
| `Status.log` | Job 메타 (시작/종료/소요시간) | 메타 |
| `WaferData.log` | 웨이퍼 처리 이력 (챔버·시간) | 메타 |
| `lotdata.log` | 스케줄러 이벤트 로그 | 메타 |
| `Info.log` | 장비 리비전·모듈 정보 | 메타 |
| `AlarmCount.log` | 알람 발생 수 | 품질 점검 |

### 2.3 PROCESS 파일 구조
- **형식:** 탭 구분 텍스트, 1행 = 헤더, 이후 각 행 = 1 시점
- **샘플링:** **~4 Hz** (초당 약 4행)
- **행 수:** Standard 1,786 / Source 1,250 / Bias 1,680
- **컬럼:** 1열 timestamp + **42개 센서** = 43열
- **단계 마커:** 파일 중간에 `$STEP n` 라인이 삽입되어 stable/ARC/ME1/over-etch 구간을 구분

### 2.4 센서 42채널 (6 그룹)

| 그룹 | 채널 | 의미 |
|:--|:--|:--|
| **RF Power** (6) | `ao/aiRFBIAS_*`, `ao/aiRFSOURCE_*` | Bias·Source 각 Setpoint / Forward / Reflect 파워 |
| **Gas Flow** (24) | `aiMFC_FLOW_01~12`, `aoMFC_FLOW_01~12` | MFC 12채널 실측(ai) + 설정(ao) |
| **ESC** (2) | `ai/aoESC_VOLTAGE_*` | 정전척(E-Chuck) 전압 모니터/설정 |
| **He Cooling** (6) | `av/aiHE_1_*`, `av/aiHE_2_*` | 백사이드 He 누설률·유량·압력 ×2존 |
| **Pressure** (1) | `aiAPC_CurrentPressure` | APC 챔버 압력(실측) |
| **Temperature** (3) | `aiTEMP_PV1ch1~3` | 챔버 온도 3채널 |

- **접두어 규약:** `ai`=실측(actual input), `ao`=설정(actual output/setpoint), `av`=계산값.
- **MFC ↔ 가스 매핑 (RECIPE setpoint ↔ xlsx 가스값 자동 도출, 3 Run 공통):**
  `MFC1 = HBr`, `MFC4 = Cl2`, `MFC7 = CF4`, `MFC11 = O2/He`

### 2.5 신호 해석 기준
- **센서 0값 구간:** RF off / 안정화(stabilization) 구간 → 식각 본구간 아님.
- **Reflect Power:** RF 매칭 손실분(낮을수록 정상).
- **Forward − Reflect:** 실효 투입 파워.
- **백사이드 He(`HE_*`):** 웨이퍼 척킹·냉각 상태 모니터링.

---

## 3. `AI DATA/` — SEM 측정 이미지

### 3.1 폴더 구성
실험 조건별 폴더 3개 (`Standard`, `Source power 300w`, `Bias power 140w`),
각 폴더에 이미지 3쌍(`.bmp` + 메타 `.txt`).

### 3.2 파일 종류

| 파일 | 측정 대상 | 배율 | 용도 |
|:--|:--|:--|:--|
| `PR.bmp` / `PR.txt` | 잔류 PR 단면 | ×15k | **Remain PR** 측정 근거 |
| `Depth.bmp` / `Depth.txt` | Poly 식각 깊이 단면 | ×15k | **Poly Depth / E/R** 측정 근거 |
| `1.bmp` / `1.txt` | 전경(overview) | ×5k | 형상 참고 |

### 3.3 SEM 촬영 사양 (메타 .txt)
- **장비:** Hitachi **S-4700** (FE-SEM)
- **이미지:** 640 × 480, 8-bit Grayscale, BMP
- **가속전압:** 25 kV, **WD:** 7.6 mm, Signal: SE(M)
- **Scale bar(MicronMarker):** 3000 nm
- **측정일:** 2026-03-25

### 3.4 특성
- 이미지는 **표 형태 feature가 아니라 측정 원본**이다.
  xlsx의 Remain PR / Poly Depth 수치는 이 단면 이미지에서 수동 측정한 값.
- 본 프로젝트 모델 입력에는 직접 사용하지 않음(향후 이미지 기반 측정 자동화 시 활용 가능).

---

## 4. 데이터 간 관계 (연결 구조)

```
                     실험 조건 3종 (Standard / Source 300w / Bias 140w)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  Equipment Data            AIetch-process-data.xlsx          AI DATA
  (공정 중 센서 입력)   ──▶   (조건 + 결과 정답 라벨)   ◀──    (정답 측정 원본)
   PROCESS 시계열               Remain PR / Depth / E/R          SEM 단면 이미지
```

- **모델 학습 흐름:** `Equipment Data`(입력 X) → `xlsx`(정답 y) 로 매핑 학습.
- `AI DATA`는 y의 신뢰성 근거(이미지)로, 모델 파이프라인에는 미사용.

---

## 5. 데이터 특성 · 한계 · 주의사항

| 구분 | 내용 |
|:--|:--|
| **표본 규모** | 라벨 = **3 Run (각 1 측정점)**. 통계적 일반화엔 부족, 파일럿 수준. |
| **DOE 구조** | Source / Bias 단일 인자 변경. 교호작용·연속 범위 미탐색. |
| **측정 위치** | Coupon center 1점만 (Edge T/B/L/R 미측정). |
| **타깃 종속성** | Poly E/R = Poly Depth × 7.5 → 독립 타깃 아님. |
| **시계열 활용** | 라벨이 Run당 1개이므로, 시계열을 window로 펼쳐도 **독립 표본 수는 늘지 않음**(누설 주의 → Run 단위 분리 평가 필요). |
| **단계 혼재** | 한 로그에 stable/ARC/ME1/over-etch 혼재 → **ME1(주 식각)만 선별**해야 신호 명확. |
| **날짜 표기** | xlsx Date(`260224`)와 장비·SEM 로그(2026-03-25) 표기 불일치 — 기재값 그대로 보존. |
| **단위 혼용** | Depth(㎚) vs E/R(Å/min) 등 단위 상이 — 전처리 시 주의. |

---

## 6. 단위 · 약어 정리

| 약어 | 의미 |
|:--|:--|
| ARC | Anti-Reflective Coating (반사방지막) 식각 단계 |
| ME1 | Main Etch 1 (주 Poly 식각 단계) |
| PR | PhotoResist (감광막) |
| E/R | Etch Rate (식각률) |
| MFC | Mass Flow Controller (가스 유량 제어기) |
| ESC | Electro-Static Chuck (정전척) |
| APC | Auto Pressure Controller (자동 압력 제어) |
| RFT / RFB | RF Top(Source) / RF Bottom(Bias) 파워 |
| sccm / mT / Torr | 표준 cc/min / milliTorr / Torr |
| Coupon | 웨이퍼 조각 시편 |
