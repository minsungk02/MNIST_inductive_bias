# 추가 진단지표 변경 내역 및 결과 1차 분석

> 2026-07-16. 재학습 없이 기존 32개 `best.pt`(full 2 + subset 30)와 history.json만으로
> 신규 시각 진단지표를 뽑고 전체 스윕을 실행한 결과의 1차 해석.
> 지표별 정의/실행법 상세는 [extra_metrics.md](extra_metrics.md) 참고.

---

## 1. 변경 내역

| 파일 | 변경 |
|---|---|
| `scripts/analyze.py` | full 2개 하드코딩 → `--all`(32개 전체 스윕)/`--run-dir` 지원. best.pt에 내장된 cfg로 yaml 없이 모델 재구성. per-run 산출물 추가: `pos_embed_sim.png`, `test_preds.npy`, analysis.json에 `pca2d`·`pos_embed_locality`·`test_acc`·식별키(arch/subset_size/subset_seed) |
| `src/analysis.py` | 신규 함수: `pos_embed_similarity`/`pos_embed_locality`(ViT 위치임베딩 구조), `penultimate_pca2d`(표현공간 2D PCA), `predict_all`(예측 캐시) |
| `scripts/analyze_extra.py` | **신규.** 2D translation 민감도 히트맵(`--which translation2d`), Fourier 민감도 히트맵(`--which fourier`, eps=4.0 캘리브레이션), 첫 레이어 필터 몽타주(`--which filters`) |
| `scripts/compare.py` | 전체 run 집계로 확장(기존 full 전용 플롯은 그대로 유지) + 신규 비교 플롯 10종 생성 |
| `docs/extra_metrics.md` | 지표별 해석 가이드 + 재학습 시 추가 로깅 제안 |

실행된 스윕: `analyze.py --all`(32 run) → `analyze_extra.py` translation2d 전체(2000장)
+ full 10k / fourier full(2000장, eps=4.0) / filters → `compare.py`.

---

## 2. 신규 그래프(png) 카탈로그 — 각각 뭘 뽑은 것이고, 뭐가 보였나

모두 `experiments/` 바로 아래에 생성. (per-run 그래프는 §끝 표 참고)

### 2.1 `compare_erf_grid.png` — ERF 히트맵 그리드 (arch 행 × 데이터크기 열)

- **뭘 한 것**: 중앙 유닛의 입력 |gradient| 히트맵(ERF)을 32개 run 전부에서 계산,
  각 맵을 sum=1 정규화 후 subset seed 3개 평균, 행별 공유 컬러스케일로 나열.
- **어떻게 읽나**: 밝은 영역 = 중앙 표현에 실제로 영향을 주는 입력 픽셀.
  뭉쳐 있으면 국소(local), 퍼져 있으면 전역(global).
- **관측**: CNN 행은 n=100부터 full까지 **항상 중앙에 뭉친 blob** — 데이터가
  10배씩 늘어도 모양이 거의 안 변한다. ViT 행은 전 구간에서 화면 전체에 퍼진
  speckle + 중앙 패치 주변의 약한 사각 구조 — **어느 n에서도 국소화되지 않는다.**

### 2.2 `compare_bias_vs_n.png` — 스칼라 바이어스 지표 4종 vs 데이터 크기 (2×2)

- **뭘 한 것**: run별 analysis.json에서 ①ERF ±7px 에너지 비율 ②평균 attention
  distance(ViT, uniform baseline 점선) ③4px shift equivariance error
  ④마지막 블록 고주파 비율을 뽑아 n축(log)·seed 평균±std로 그림.
- **어떻게 읽나**: **곡선이 평평하면 그 성질은 아키텍처 내장, n에 따라 움직이면
  데이터로부터 학습되는 것.**
- **관측**: ①CNN 0.93대에서 평평 vs ViT 0.38~0.46 배회 ②ViT attention distance는
  1.982→1.956, baseline 2.008 바로 밑에 붙어 있음(끝까지 전역) ③CNN 0.12~0.19 vs
  ViT 1.33~1.44 (7~11배 격차 유지) ④CNN만 0.285→0.453으로 뚜렷이 상승 —
  유일하게 'CNN이 데이터에 맞춰 조정하는' 지표.

### 2.3 `compare_pos_embed.png` — ViT 위치임베딩 유사도 (n별 맵 + locality 곡선)

- **뭘 한 것**: ViT 패치 pos_embed 16개 간 cosine 유사도 행렬 [16,16]을 n별로
  (seed 평균) 나열하고, locality 점수(= off-diagonal에서 corr(유사도, −패치거리);
  +1=2D locality 학습, 0=init 무구조)를 n축 곡선으로. **데이터 불필요, 가중치만 사용.**
- **어떻게 읽나**: 가까운 패치끼리 유사도가 높은 밴드 구조가 생기면 "2D 격자
  locality를 배웠다"(Raghu et al. 2021 스타일).
- **관측 (핵심 발견)**: locality 점수가 **+0.03(n=100, init 노이즈) → −0.27(full)로
  단조 감소.** 즉 이 소형 ViT는 locality가 아니라 **이웃 패치를 서로 구별하는
  (anti-local) 위치코드**를 학습한다. 구조가 n에 따라 체계적으로 변한다는 것
  자체가 "위치 표현은 학습되는 것"의 증거이되, 방향이 대형 ViT 문헌과 반대.

### 2.4 `compare_translation2d.png` — 2D shift 정확도 히트맵 그리드 (arch × n)

- **뭘 한 것**: test 이미지를 (dx,dy) ∈ [−4,4]² 81가지로 평행이동(배경 0 채움,
  기존 robustness와 동일 규약)시켜 각각 정확도 측정. 기존 1D 대각 shift 곡선의
  2D 확장. 전체 run은 2000장, full run은 10k 전체로 측정.
- **어떻게 읽나**: 중앙 = clean 정확도. 밝은 영역이 넓을수록 translation invariance.
- **관측**: CNN = **고원**(full: 81칸 평균 0.955, 최악 모서리 0.848),
  ViT = **bullseye 붕괴**(full: clean 0.976 → 최악 0.134). ViT는 n이 커져도
  최악값이 0.07→0.13으로 거의 안 오름 — '본 적 없는 위치'는 데이터로 해결 안 됨.
  부수 관찰: CNN 맵에 수평 밴딩(세로 이동에 더 민감) — MaxPool stride 효과 추정.

### 2.5 `compare_agreement.png` — CNN·ViT 오류 겹침 누적 막대 (n별)

- **뭘 한 것**: run별 test 10k 예측 캐시(`test_preds.npy`)를 (size, subset_seed)로
  짝지어 {둘 다 정답 / CNN만 / ViT만 / 둘 다 오답} 비율을 계산, seed 평균.
- **어떻게 읽나**: disagreement(CNN만+ViT만)가 크면 두 바이어스가 **서로 다른
  실수**를 만든다는 example 단위 증거 (앙상블 여지).
- **관측**: disagreement 35.1%(n=100) → 2.4%(full). n=100에서 테스트셋의
  **28.7%는 CNN만 맞힘** (ViT만 6.4%, 비율 4.5:1) — 데이터효율 격차의 출처가
  숫자로 분해된다.

### 2.6 `compare_pca_grid.png` — 분류 직전 표현의 2D PCA 산점도 (arch × n)

- **뭘 한 것**: head 입력 표현(CNN: GAP 뒤 160d, ViT: CLS 64d)을 동일한 test
  2000장에 대해 2D PCA 투영, 색=숫자 클래스. 부호 규약 고정으로 패널 간 직접 비교.
- **어떻게 읽나**: 클래스별 색이 갈라져 있으면 표현공간이 이미 분류에 맞게
  조직화된 것.
- **관측**: CNN은 n=100부터 클러스터 구조가 보이고(설명분산 83+10% — 저차원·구조적)
  n이 커져도 형태 유지. ViT는 n=100에서 완전히 뭉개진 구름(20+15%) →
  n=3k~54k에 걸쳐 서서히 클러스터로 조직화. **데이터효율 곡선의 표현공간 버전.**

### 2.7 `compare_confusion.png` — full run confusion matrix 쌍

- **뭘 한 것**: full 예측 캐시로 10×10 confusion(행 정규화). 대각선은 숫자로
  acc%, 색은 오류율만 (오류 패턴 강조).
- **관측**: 둘 다 4→9, 7→2가 주요 혼동쌍이지만 크기가 다름 — CNN 최악 셀은
  4→9(0.8%), ViT는 7→2(1.4%)와 9→4·5→3. ViT의 오류율 스케일 자체가 약 1.7배.

### 2.8 `compare_per_class_acc.png` — 클래스별 정확도 (class × n 히트맵 + 차이맵)

- **뭘 한 것**: 예측 캐시로 숫자 클래스별 정확도를 class × n 히트맵으로,
  세 번째 패널에 CNN−ViT 차이(빨강=CNN 우세).
- **관측**: 격차는 균일하지 않다 — n=100에서 **4·5·8에서 격차 최대**(4는 +0.5 수준),
  1은 전 구간 격차 거의 0 (직선이라 둘 다 쉬움). ViT는 곡선 획이 많은 숫자를
  늦게 배운다. full에서는 차이맵이 거의 백지(수렴).

### 2.9 `compare_fourier.png` — Fourier 민감도 히트맵 쌍 (full, 공유 스케일)

- **뭘 한 것**: 단일 Fourier 기저(단위 L2)를 eps=4로 더했을 때의 오류율을
  주파수별로 잰 28×28 맵 (중앙=DC, 가장자리=고주파; Yin et al. 2019).
  eps 캘리브레이션: eps=2는 CNN이 안 무너져(최대 2%) 신호 없음 → eps=4 채택.
- **어떻게 읽나**: 밝은 주파수 = 그 대역 섭동에 취약. `freq_response`(feature map
  관점)의 입력공간 대응물.
- **관측**: 같은 크기 섭동에서 **ViT가 전 대역에 걸쳐 훨씬 취약**(최대 0.63 vs
  0.29). 대역 구조는 반대 방향 — CNN은 저주파 평균 0.093 < 고주파 0.110
  (고주파 의존 = high-pass), ViT는 저주파 0.389 > 고주파 0.336 (저주파 의존 =
  low-pass). Park & Kim 2022 명제의 입력공간 재확인. CNN 맵은 매끈하고 DC·초고주파
  모서리는 깜깜(둔감), ViT 맵은 중저주파에 hot spot이 흩어져 있음.

### 2.10 `compare_first_layer_filters.png` — 첫 레이어 필터 몽타주

- **뭘 한 것**: full 체크포인트에서 CNN conv1(3×3 ×40)과 ViT patch-embed
  (7×7 ×64) 가중치를 필터별 정규화해 나란히 (가중치만, 데이터 불필요).
- **관측**: CNN은 방향성 있는 엣지/코너 검출기 모양(전 위치 공유), ViT 패치
  템플릿은 고주파 노이즈처럼 보이는 패치별 부호 — 첫 층부터 정보를 조직하는
  방식이 다름을 보여주는 발표용 garnish.

### per-run 그래프 (experiments/<run>/ 안, 32개 run 각각)

| 파일 | 내용 | 비고 |
|---|---|---|
| `erf.png`, `equivariance.png` | 기존 지표 | 이번에 **전 run으로 확장** |
| `attention_distance.png` | 블록별 attention 거리 | ViT run만 |
| `pos_embed_sim.png` | 위치임베딩 유사도 [16,16] | **신규**, ViT run만 |
| `translation2d.png`/`.json` | 2D shift 히트맵 | **신규**, 전 run |
| `fourier.png`/`.json` | Fourier 민감도 | **신규**, full run만 |
| `test_preds.npy` | test 10k 예측 캐시 (int8) | **신규**, 전 run |
| `analysis.json` | 위 지표 전부 + pca2d + test_acc | 전 run |

---

## 3. 종합 수치표 (subset seed 3개 평균)

| arch | n | test acc | ERF ±7px | equiv@4px | attn dist | posEmb loc | 2D-shift 평균 | 2D-shift 최악 |
|---|---|---|---|---|---|---|---|---|
| cnn | 100 | 0.714 | **0.933** | **0.124** | – | – | 0.475 | 0.280 |
| cnn | 300 | 0.787 | 0.934 | 0.148 | – | – | 0.553 | 0.291 |
| cnn | 1k | 0.918 | 0.930 | 0.178 | – | – | 0.694 | 0.401 |
| cnn | 3k | 0.961 | 0.923 | 0.187 | – | – | 0.790 | 0.500 |
| cnn | 10k | 0.981 | 0.921 | 0.190 | – | – | 0.902 | 0.707 |
| cnn | 54k | 0.988 | 0.918 | 0.194 | – | – | **0.955** | **0.848** |
| vit | 100 | 0.492 | 0.379 | 1.438 | 1.982 | +0.029 | 0.187 | 0.069 |
| vit | 300 | 0.601 | 0.394 | 1.415 | 1.983 | −0.027 | 0.209 | 0.075 |
| vit | 1k | 0.740 | 0.427 | 1.378 | 1.980 | −0.106 | 0.249 | 0.086 |
| vit | 3k | 0.843 | 0.418 | 1.342 | 1.978 | −0.174 | 0.297 | 0.099 |
| vit | 10k | 0.925 | 0.393 | 1.330 | 1.968 | −0.205 | 0.393 | 0.105 |
| vit | 54k | 0.976 | 0.462 | 1.341 | **1.956** | **−0.274** | 0.533 | 0.134 |

(attn dist의 uniform baseline = 2.008. 2D-shift = (dx,dy)∈[−4,4]² 81칸 평균/최솟값 정확도)

---

## 4. 1차 분석 — 관점별 종합

### 4.1 "내장된 바이어스 vs 학습해야 하는 바이어스" — 곡선의 평평함이 곧 결론

`compare_bias_vs_n.png`의 핵심은 **CNN 곡선이 전부 평평하다**는 것이다.

- **ERF locality (±7px 에너지)**: CNN은 n=100→54k에서 0.933→0.918로 사실상 불변.
  locality는 3×3 conv 커널 구조에서 오는 것이지 데이터에서 오는 게 아니다.
  ViT는 0.38~0.46 사이를 배회 — **540배 데이터를 부어도 국소성이 생기지 않는다.**
- **Equivariance error @4px**: CNN 0.12~0.19 vs ViT 1.33~1.44 (약 **7~11배** 격차,
  전 구간 유지). shift 불변성도 데이터로 좁혀지지 않는다.
- 흥미 포인트: CNN의 equivariance error가 n과 함께 미세하게 **증가**(0.124→0.194)
  하고 고주파 비율도 0.285→0.453으로 올라간다. 데이터가 많을수록 CNN도 위치
  의존적·고주파 세부특징을 더 활용한다는 신호 — "CNN=완전 불변"이 아니라
  "구조적으로 불변에 가깝게 **출발**한다"가 정확한 서술.

### 4.2 ViT는 locality를 배웠는가? — 아니오, 오히려 반대 방향 (핵심 발견)

attention distance는 전 구간 uniform baseline 근처(§2.2), pos_embed는 데이터가
커질수록 **anti-local 코드**로 조직화(§2.3, +0.03→−0.27). 대형 ViT 문헌의
"데이터가 많으면 locality가 창발한다"는 **16패치·MNIST 스케일에서 재현되지 않는다.**
이 ViT는 공간 이웃 구조 대신 패치 정체성을 구별하는 위치코드를 학습했고, 이는
4.3의 shift 취약성과 정합적이다(위치코드가 구별적일수록 내용이 한 패치 옆으로
이동하면 표현이 크게 변함).

### 4.3 Shift robustness의 2D 전모 — 고원 vs 원뿔

CNN full은 ±4px 어디로 밀어도 0.85 이상(고원), ViT full은 clean 0.976이 모서리에서
0.13대까지 붕괴(bullseye). 결정적인 것은 **ViT의 최악값이 n에 거의 무반응**
(0.07→0.13)이라는 점 — 증강 없이 데이터만으로는 위치 일반화가 안 생기며,
이것이 기존 1D shift 결과(97.6→16.8)의 완전한 그림이다.

### 4.4 두 모델은 '다른' 문제를 틀린다

§2.5 + §2.8: n=100에서 disagreement 35.1%, 그 대부분(28.7%p)이 'CNN만 정답'.
클래스별로는 4·5·8에서 격차 최대, 1은 격차 없음 — ViT는 곡선 획이 많은 숫자부터
못 푼다. full에서도 2.4%가 갈리므로 두 바이어스는 끝까지 완전히 수렴하지 않는다.

### 4.5 주파수 관점 — 서로 반대의 대역 의존성

feature map(`freq_response`)과 입력공간(`compare_fourier.png`) 양쪽에서
**Conv=high-pass / MSA=low-pass**가 일관 확인. 같은 eps=4 섭동에서 ViT가
전 대역 취약(최대 0.63 vs 0.29)한 것은 기존 noise robustness 결과의 입력공간 근거.

---

## 5. 한 문단 요약

MNIST 스케일에서 CNN의 locality·translation-equivariance는 **데이터와 무관하게
아키텍처에 내장**되어 있고(모든 바이어스 지표가 n축에서 평평), 그 현금 가치는
소량 데이터 성능(+22%p @ n=100, 테스트셋의 28.7%는 CNN만 정답)과 shift 강건성
(2D 고원 vs bullseye 붕괴)으로 나타난다. 반면 ViT는 위치 표현을 데이터로부터
학습하지만 그 방향이 locality가 아니라 **패치 구별(anti-local) 코드**였으며
(pos_embed locality +0.03→−0.27, attention은 끝까지 uniform 근처), 이것이
'본 적 없는 위치'에 일반화하지 못하는 이유(2D-shift 최악값 0.07→0.13 정체)를
내부에서 설명한다. 두 모델의 주파수 선호(Conv=high-pass/MSA=low-pass)는
feature map과 입력공간(Fourier 민감도) 양쪽에서 일관되게 재확인됐다.

## 6. 남은 확인거리 (후속)

- CNN 2D-shift 히트맵의 수평 밴딩 원인 (MaxPool stride vs 숫자 형태).
- CNN equivariance error의 완만한 증가(0.124→0.194)가 어느 층에서 오는지.
- disagreement 2.4%(full)의 실제 이미지 확인 — 두 바이어스가 갈리는 경계 사례.
- 재학습 기회가 있으면: log-spaced epoch 체크포인트로 위 지표들의 시간축 버전
  (자세한 목록은 extra_metrics.md 마지막 절).
