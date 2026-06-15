# 🔥 칼로리 소모량 예측 — 역공학으로 데이터 생성 공식 복원

> **"RMSE 0.02, 리더보드 1위"는 표면일 뿐입니다.**
> 진짜 성과는 트리 모델이 못 넘는 벽 앞에서 *"이 데이터는 수식으로 생성된 것 아닌가?"*를 의심하고,
> **역공학으로 생성 공식(Keytel)과 숨은 단위 버그(lb)까지 복원**한 것입니다.

| 역할 | 기간 | 맥락 | 성과 |
|---|---|---|---|
| **팀장 (3명)** | 2026.02 | AI 헬스케어 ML 해커톤 · 13조 "13일의 칼로리 버닝조" | 기수 내 리더보드 **1위** *(비공식)* · RMSE 0.5 → **0.02** |

---

## 1. 문제 (Problem)

운동 기록(심박수·체온·키·체중·성별·나이·운동시간)으로 **소모 칼로리(`Calories_Burned`)를 예측**하는 회귀 문제입니다.

- 데이터: `train.csv` (7,500 × 11), `test.csv` (7,500 × 10)
- 피처: `Exercise_Duration`, `BPM`, `Body_Temperature(F)`, `Weight(lb)`, `Height`, `Gender`, `Age`, …
- 평가: RMSE (낮을수록 우수)

**막힌 지점**: CatBoost를 Feature Engineering + Optuna로 최적화해도 RMSE는 **~0.5에서 정체**. "다른 수강생들은 0.1 이하"라는 단서가, 모델이 아니라 **데이터 자체**를 의심하게 만든 출발점이었습니다.

---

## 2. 접근 (Approach) — 5단계 역공학

> 핵심 전환: *"모델을 더 돌리지 말고, 이 데이터가 어떻게 생성되었는지를 의심하자."*

| 단계 | 발견 | 방법 |
|---|---|---|
| **1** | 데이터가 **100% 결정론적** (같은 피처 → 항상 같은 칼로리) | `groupby(features).std()` → 불일치 **0개** / 고유 조합 7,499 |
| **2** | 칼로리가 **모두 정수** (1~300) | `Calories_Burned % 1 == 0` → 소수점 0개 |
| **3** | `Weight(lb)` 고유값 **간격이 2.2 lb** | 고유값 88개 중 81개 간격 = 2.2 lb = **1 kg 단위** |
| **4** | **Keytel(2005) 공식**의 BPM·Age 계수가 회귀 추정치와 일치 | `LinearRegression`으로 성별별 계수 추정 → Keytel/4.184와 비교 (차이 ~1e-4) |
| **5** | **Weight 계수만 어긋남 → 단위 버그 규명** | 체중이 정수 kg인데 **lb로 저장**됨을 역추적, 그리드 서치로 보정 |

**숨은 단위 버그의 정체**

```
Keytel 원본:  0.1988 / kg
이 데이터:    0.1988 / 4.184 ÷ 2.20462 ≈ 0.02160 / lb-unit   (정수 kg을 lb로 저장)
→ 복원:       W_kg_int = round(Weight(lb) / 2.20462)
```

**복원된 최종 공식**

```
Male:    Cal = round((0.15079×BPM + 0.02160×W_kg_int + 0.04821×Age − 13.168) × Duration)
Female:  Cal = round((0.10688×BPM − 0.01372×W_kg_int + 0.01769×Age −  4.876) × Duration)
```

---

## 3. 핵심 성과 (Key Results)

- **Train 정확률 100% (7,500/7,500)** · 반올림 RMSE **0.0000** — 생성 공식을 완벽 복원.
- **RMSE 0.5 → 0.1 → 0.02**로 단계적 돌파, 기수 내 프로젝트 리더보드 **1위** *(대외 공식 수상 아님)*.
- **first-principles 사고의 교과서적 사례** — *"CatBoost는 트리로 f(X)를 근사, 선형회귀는 직선으로 근사, 역공학은 데이터가 알려주는 진짜 f(X)를 그대로 쓴다."* 결론은 **"데이터 특성에 맞는 모델(접근) 선정"**.
- 트리 모델의 한계를 모델 튜닝이 아니라 **데이터 생성 메커니즘 진단**으로 돌파한 점이 면접 포인트.

---

## 4. 기술 스택 (Tech)

- **베이스라인(정체 구간)**: `CatBoost` · `XGBoost` · `LightGBM` + `Optuna` 튜닝 + `SHAP` 기반 Feature Selection
- **역공학(돌파)**: `LinearRegression`(계수 검증) · Grid Search(Weight 계수 탐색) · 단위 환산(lb→kg)·타깃 정수화
- **공통**: `pandas` · `numpy` · `matplotlib` · OOF(K-Fold) RMSE 평가

---

## 5. 재현 방법 (Reproduce)

```bash
# 1) 의존성
pip install pandas numpy scikit-learn matplotlib koreanize_matplotlib

# 2) 데이터 배치 (대회 제공 train.csv / test.csv)
#    노트북 상단 BASE 경로를 본인 환경에 맞게 수정

# 3) 실행: 노트북을 순차 실행
#    [Perfect]Regressor_260222_Keytel_IntKg.ipynb
#    → submit_260222_Perfect_Keytel_rounded.csv 생성 (제출 권장본)
```

| 파일 | 설명 |
|---|---|
| `[Perfect]Regressor_260222_Keytel_IntKg.ipynb` | 최종 역공학 솔루션 (5단계 발견 → 공식 복원 → 제출 생성) |
| `13조_13일의 칼로리 버닝조_260224_발표용.pptx` | 발표자료 |

> ⚠️ 공개 전 점검: 노트북의 `BASE` 절대경로(개인 폴더)는 상대경로로 교체, 대회 데이터는 재배포 규정 확인 후 다운로드 안내로 대체.

---

*이 프로젝트의 한 줄: 모델을 더 돌리는 대신 데이터의 생성 원리를 의심한다 — 데이터 사이언스의 first-principles.*
