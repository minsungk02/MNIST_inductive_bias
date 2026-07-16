# 추가 진단 지표 가이드 — inductive bias를 '눈으로' 구분하기

재학습 없이, 이미 저장된 32개 `best.pt`(full 2 + subset 30)와 `history.json`만으로
계산하는 시각 진단 지표 모음. 핵심 관점 전환:

> "CNN은 국소적이다"(정적 사실) → **"locality/invariance가 데이터 크기에 따라
> 어떻게 발현되는가"**(이 실험의 주제 그 자체)

CNN의 inductive bias(locality, weight sharing, translation equivariance)는
**아키텍처에 내장**되어 있어 n=100에서도 이미 존재한다. ViT는 같은 성질을
**데이터로부터 학습**해야 하므로 n이 작으면 아예 없다. 아래 지표들은 전부
이 차이를 서로 다른 각도에서 시각화한다.

---

## 실행 방법

```bash
# 1) 모든 run에 대해 진단 지표 일괄 계산 (GPU ~10-15분)
python scripts/analyze.py --all

# 2) 2D translation 민감도: 전체 run(2000장) + full run은 10k 전체
python scripts/analyze_extra.py --which translation2d --all --max-images 2000
python scripts/analyze_extra.py --which translation2d --runs "*_full_*" --max-images 10000

# 3) Fourier 민감도 (full run만; eps=4.0은 캘리브레이션 결과)
python scripts/analyze_extra.py --which fourier --runs "*_full_*" --max-images 2000 --eps 4.0

# 4) 첫 레이어 필터 비교 (가중치만)
python scripts/analyze_extra.py --which filters

# 5) 비교 플롯 일괄 생성 -> experiments/compare_*.png
python scripts/compare.py
```

단일 run 스모크 테스트: `python scripts/analyze.py --run-dir experiments/vit_n100_s42_ss0`

---

## 지표별 해석 가이드

### C1. 기존 진단지표 × 데이터 크기 (`compare_erf_grid.png`, `compare_bias_vs_n.png`)

기존에 full 체크포인트 2개에만 돌리던 ERF·attention distance·equivariance·주파수응답을
32개 전체에 돌려 **데이터 크기 축**으로 폈다.

- **`compare_erf_grid.png`** — arch(행) × n(열)의 ERF 히트맵 그리드.
  읽는 법: CNN 행은 n=100부터 full까지 **항상 중앙에 뭉친 점**(내장된 locality).
  ViT 행은 어느 n에서도 넓게 퍼짐 → locality가 데이터로도 잘 안 생김(MNIST 스케일).
  각 맵은 sum=1 정규화 후 subset seed 3개 평균, 행별 공유 컬러스케일.
- **`compare_bias_vs_n.png`** — 스칼라 지표 4종 vs n (log축, seed 평균±std):
  ERF ±7px 에너지 비율 / 평균 attention distance(uniform baseline 점선) /
  4px shift equivariance error / 마지막 블록 고주파 비율.
  읽는 법: **CNN 곡선이 평평하면 "바이어스는 내장"**, ViT 곡선이 n에 따라
  움직이면 "바이어스는 학습됨", 안 움직이면 "이 데이터 규모로는 학습 안 됨".

관측 예 (n=100, subset seed 0): ERF 중앙 에너지 CNN 94% vs ViT 41%,
equivariance@4px CNN 0.13 vs ViT 1.42 — 소량 데이터에서 이미 극명한 대비.

### C2. ViT pos_embed 유사도 (`pos_embed_sim.png`, `compare_pos_embed.png`)

패치 위치임베딩 16개 간 cosine 유사도 [16,16] (CLS 제외) + locality 점수
(= off-diagonal 쌍에서 corr(유사도, −패치거리); +1 = 2D locality 학습, 0 = init 노이즈).

읽는 법: ViT가 "가까운 패치는 비슷한 위치코드"라는 **2D 격자 구조 자체를
배웠는가**를 가중치만으로 판정 (Raghu et al. 2021 스타일). n=100에서는
init(trunc_normal σ=0.02) 근처라 무구조(점수 ≈ 0.03) — **무구조 자체가 신호**다.

**실제 관측 (이 실험):** locality 점수가 n=100의 0.03에서 full의 **−0.27까지
단조 감소**한다. 즉 이 소형 ViT(4×4 패치 격자)는 대형 ViT처럼 2D locality를
배우는 게 아니라, 오히려 **이웃 패치를 서로 구별하는(anti-local) 위치코드**를
학습한다. 해석: (1) pos_embed 구조가 데이터량에 따라 체계적으로 변한다는 것
자체가 "위치 표현은 학습되는 것"의 증거이고, (2) 그 방향이 locality가 아니라는
점은 attention distance가 uniform baseline에 붙어 있는 결과와 함께 "MNIST +
16패치 스케일에서는 locality prior가 데이터로 유도되지 않는다"를 뒷받침한다.

### C3. 2D translation 민감도 (`translation2d.png`, `compare_translation2d.png`)

기존 robustness.json의 1D 대각 shift 곡선을 (dx,dy) ∈ [-4,4]² 전평면으로 확장한
정확도 히트맵. 섭동 규약은 기존과 동일([0,1]에서 fill=0 shift 후 정규화).

읽는 법: **CNN = 고원(plateau), ViT = 중앙만 밝은 원뿔(bullseye)**.
Conv+GAP의 translation invariance와 고정 pos_embed의 위치 과적합이
한 장에 대비된다. full ViT: clean 0.98 → 모서리 ~0.15.

### C4. 예측 캐시 기반 오류 분석 (`compare_confusion.png`, `compare_per_class_acc.png`, `compare_agreement.png`)

run마다 test 10k 예측을 `test_preds.npy`(int8)로 캐시 → 세 가지 플롯:

- **confusion** (full): 행 정규화, 대각선은 숫자(acc%), 색은 오류율만 —
  어떤 숫자쌍을 헷갈리는지 아키텍처별로 비교.
- **per-class acc** (class × n 히트맵 + CNN−ViT 차이): 어떤 클래스에서
  격차가 크고, ViT가 어떤 숫자를 늦게 배우는지.
- **agreement** (n별 누적 막대): {둘 다 정답 / CNN만 / ViT만 / 둘 다 오답}.
  읽는 법: disagreement가 크면 **서로 다른 bias가 서로 다른 실수를 만든다**는
  직접 증거 (두 모델이 상보적 → 앙상블 여지). n이 작을수록 갈라짐이 커야 정상.
  실제 관측: disagreement가 n=100의 35.1%(대부분 'CNN만 정답')에서 full의
  2.4%까지 감소 — 데이터효율 격차가 example 단위로 어디서 오는지 보여준다.

### C5. 표현공간 PCA (`compare_pca_grid.png`)

분류 직전 표현(CNN: GAP 뒤 160d, ViT: 최종 LN 뒤 CLS 64d)을 test 2000장에 대해
2D PCA로 투영, arch × n 그리드 산점도 (색 = 숫자 클래스).

읽는 법: 작은 n에서 **CNN은 이미 클래스별 클러스터가 갈라져 있고 ViT는 뭉개져 있으면**
데이터효율 격차를 '표현공간의 기하'로 본 것. n=100에서 설명분산도 참고
(CNN 83+10% vs ViT 20+15% — CNN 표현이 훨씬 저차원/구조적).
모든 run에서 같은 이미지·같은 부호규약을 쓰므로 패널 간 직접 비교 가능.

### C6. Fourier 민감도 히트맵 (`fourier.png`, `compare_fourier.png`)

Yin et al. 2019: 단일 Fourier 기저(단위 L2, eps=4.0)를 더했을 때의 오류율을
주파수 (i,j)별로 잰 28×28 맵 (중앙 = DC, 가장자리 = 고주파).

읽는 법: 기존 `freq_response`(feature map의 고주파 비율)의 **입력공간 버전**.
어느 주파수 대역 섭동에 무너지는가 — CNN(high-pass 특성)과 ViT(low-pass 특성)가
민감한 대역이 다르면 noise robustness 결과의 원인을 입력쪽에서 설명한다.
eps 캘리브레이션: eps=2는 CNN이 거의 안 무너져(max err 2%) 신호가 없고,
eps=4에서 CNN 30% / ViT 62%로 구조가 보인다.

### C7. 첫 레이어 필터 (`compare_first_layer_filters.png`)

CNN conv1 (3×3 × 40) vs ViT patch-embed (7×7 × 64) 가중치 몽타주.

읽는 법: CNN은 엣지/블롭 검출기(전 위치 공유), ViT는 패치 템플릿.
"같은 파라미터 수라도 첫 층부터 세상을 다르게 본다"는 발표용 garnish.

---

## 산출물 위치 정리

| 위치 | 파일 | 지표 |
|---|---|---|
| `experiments/<run>/` | `analysis.json` | 모든 스칼라/배열 지표 (+식별 키 arch/subset_size/subset_seed, test_acc) |
| | `erf.png`, `equivariance.png` | 기존 per-run 플롯 (전 run으로 확장) |
| | `attention_distance.png`, `pos_embed_sim.png` | ViT 전용 |
| | `test_preds.npy` | test 10k 예측 캐시 |
| | `translation2d.json/.png` | 2D shift 히트맵 |
| | `fourier.json/.png` | Fourier 민감도 (full만) |
| `experiments/` | `compare_erf_grid.png`, `compare_bias_vs_n.png` | C1 |
| | `compare_pos_embed.png` | C2 |
| | `compare_translation2d.png` | C3 |
| | `compare_confusion.png`, `compare_per_class_acc.png`, `compare_agreement.png` | C4 |
| | `compare_pca_grid.png` | C5 |
| | `compare_fourier.png` | C6 |
| | `compare_first_layer_filters.png` | C7 |

---

## (참고) 재학습을 다시 돌릴 일이 있다면 — 추가하면 좋은 로깅

지금 체크포인트는 best epoch 1개뿐이라 "**언제** 바이어스가 생기는가"(시간축)는 못 본다.
다음 재학습 때 아래를 추가하면 위 지표들의 epoch-동역학 버전이 공짜로 열린다:

1. **log-spaced 주기 체크포인트** (`epoch_{1,2,5,10,20,40,80,160}.pt`)
   → ERF/attention distance/pos_embed locality/PCA를 **epoch 축**으로:
   "ViT가 locality를 배우는 순간"이 있는가, 아니면 끝까지 안 배우는가.
2. **epoch별 경량 지표를 history.json에 추가** (둘 다 val 1배치로 <1s/epoch):
   - `pos_embed_locality` (가중치만, 데이터 불필요)
   - mean attention distance
3. **epoch별 grad norm(전체) + 모듈별 weight norm** (patch_embed / blocks / head)
   → 작은 n에서 ViT 암기(train-val gap)의 동역학적 원인.
4. **epoch별 클래스별 val acc** (10개 숫자) → ViT가 어떤 숫자를 늦게 배우는지.
5. `best.pt`의 cfg를 `cfg.yaml`로도 함께 덤프 (pickle 없이 도구가 읽게).
