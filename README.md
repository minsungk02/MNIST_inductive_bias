# MNIST: Inductive Bias 비교 실험 (CNN vs ViT)

동일 통제조건에서 **inductive bias가 강한 CNN**과 **약한 ViT**를 비교하여,
MNIST 태스크에서 "공간 prior의 가치"를 **4개 렌즈**로 정량화한다.

- **독립변인**: 아키텍처(inductive bias) — 이 한 축만 바뀐다
- **통제변인**: 데이터/분할/정규화/옵티마이저/스케줄/예산/정규화기법/평가규칙/seed
- **핵심 통제**: param band 1.05 · dropout 0 · **train 증강 off** · config=single source of truth

## 실험 4축

| 축 | 방법 | 예측 (bias 이론) |
|---|---|---|
| **1. 데이터 효율성 (심장)** | train 100/300/1k/3k/10k/full 제한 (로그 등간격), test acc 곡선 | 소량에서 CNN 압도, 데이터↑ 격차↓ |
| **2. 수렴·과적합** | epoch별 train/val 곡선, epochs-to-best, train-val gap | ViT가 더 느리게 수렴 |
| **3. shift robustness (가장 날카로운 데모)** | test-time shift, acc vs shift량 | GAP-CNN 완만 / pos-embed ViT 급락 |
| **4. 해석 (garnish)** | CNN Grad-CAM vs ViT attention rollout | "어디를 보나" 정성 대비 |

축 3의 rotation/noise는 "둘 다 회전 prior 없음"을 확인하는 **exploratory 대조군** — shift만이 translation-equivariance의 clean test다.

## 설계 원칙

1. **config = single source of truth**. `base.yaml`에 레시피 고정, `cnn.yaml`/`vit.yaml`은 architecture만 덮어씀.
2. **레시피는 합집합(superset)**. ViT에 필수인 warmup을 CNN에도 적용 — 동일하면서 누구도 불리하지 않게.
3. **param band matching이 학습 전 필수 게이트** (`count_params.py`). 현재 CNN 146,250 / ViT 139,018 (ratio 1.05).
4. **데이터 효율성 = train-to-convergence**. subset은 "N개로 도달 가능한 최고 성능"을 재므로 step 고정이 아니라 수렴까지 학습(넉넉한 max_epochs + early stopping). **수렴 속도 축은 step 수가 맞는 full 데이터에서만** 비교 → 두 축 분리.
5. **subset은 stratified(클래스 균등) + nested(100⊂300⊂…), val은 6k 고정**. 소규모 n 분산 통제를 위해 subset_seed 3개.

## 워크플로

```bash
pip install -r requirements.txt

# 0) ★ param band 게이트 (통과 전엔 학습 안 함)
python scripts/count_params.py

# 1) 전체 실험 (서버에서 한 번에)
bash scripts/run_all.sh
```

`run_all.sh`가 하는 일: param 게이트 → 각 모델 full 학습 → subset(100/300/1k/3k/10k)×seed(0/1/2) 학습 → robustness 스윕 → test 평가 → 해석 시각화 → `compare.py` 집계.

개별 실행:
```bash
python scripts/train.py --config configs/vit.yaml --seed 42                       # full
python scripts/train.py --config configs/vit.yaml --subset-size 100 --subset-seed 0
python scripts/eval_robustness.py --config configs/vit.yaml --seed 42
python scripts/evaluate.py --config configs/vit.yaml --seed 42                     # test 1회
python scripts/visualize.py --config configs/vit.yaml --seed 42 --n 8
python scripts/compare.py
```

## 서버 사용 (SSH → tmux → detach, 터미널 꺼도 계속 학습)

VSCode Remote-SSH 터미널이 꺼지면 학습 프로세스도 죽는다(SSH 세션의 자식이라서).
**tmux** 안에서 돌리면 SSH가 끊겨도 서버가 프로세스를 유지한다.

```bash
# [최초 1회] 클론 + 환경
ssh myserver
git clone <your-repo-url> mnist-inductive-bias && cd mnist-inductive-bias
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# [런치] tmux 세션 안에서 실행
tmux new -s mnist
source .venv/bin/activate
bash scripts/run_all.sh 2>&1 | tee "experiments/run_$(date +%Y%m%d_%H%M).log"
#   -> Ctrl-b 눌렀다 떼고 d : detach. 이제 노트북/SSH 꺼도 계속 돌아감.

# [확인, 선택] 나중에 아무 때나
tmux attach -t mnist            # 라이브 진행 (다시 Ctrl-b d 로 나옴)
tail -f experiments/run_*.log   # 접속 없이 로그만
nvidia-smi                      # GPU 사용 확인
```

**개입 지점은 사실상 2번뿐**: (1) 런치 + detach, (2) 끝난 뒤 `experiments/compare_*.png` 회수.
`run_all.sh`는 개별 run이 실패해도 전체는 계속 진행하고 로그에 `FAIL`로 남긴다.
Phase 1(full 2개)이 먼저 돌아 파이프라인이 몇 분 안에 검증되고, 긴 subset 스윕은 Phase 3으로 뒤에 배치.

device는 `src/utils.get_device()`가 `cuda > mps > cpu` 자동 선택 — Mac(MPS)에서 개발, 3090 Ti(CUDA)에서 학습해도 같은 코드가 그대로 돈다. `experiments/`·`data/`는 gitignore.

## 산출물 (experiments/)

- 각 run: `best.pt`, `history.json`, `curves.png`, (`robustness.json`, `interpret.png`)
- 집계: `compare_convergence.png`, `compare_data_efficiency.png`(error band), `compare_robustness.png`, 콘솔 비교표

## 디렉토리 구조

```
configs/      base.yaml(레시피+subset+robustness) + cnn.yaml/vit.yaml(architecture만)
src/
  config.py       _base_ 상속 + deep merge
  data.py         MNIST 분할, stratified+nested subset, 고정 val, 증강 없음
  models/         simple_cnn.py, vit.py(attn 반환 지원), registry.py
  engine.py       train/val 루프, warmup+cosine, early stopping, best=min val_loss
  robustness.py   test-time shift/rotation/noise perturbation
  utils.py        seed, device 자동선택, param count
  analysis.py     진단지표 4종 함수 (attention distance/ERF/equivariance/train-val gap)
scripts/
  count_params.py     ★ param band 게이트
  train.py            학습(full/subset) + history/곡선
  evaluate.py         최종 test (1회)
  eval_robustness.py  robustness 스윕
  visualize.py        Grad-CAM / attention rollout
  analyze.py          진단지표: attention distance / ERF / equivariance (재학습X)
  compare.py          집계 -> 비교 곡선/표 (+ train-val gap / 진단 overlay)
  run_all.sh          전체 오케스트레이션
experiments/  (gitignore) 체크포인트·곡선·json
```

## 진단 지표 (2차 분석 · 재학습 없음)

1차 결과(데이터효율·shift)를 **아키텍처로 설명**하기 위해, 저장된 full 체크포인트(`best.pt`)와
`history.json` 만으로 뽑는 4개 진단 지표. `scripts/analyze.py`(①②③) + `compare.py`(④ 집계).

| 지표 | 무엇을 재나 | 무엇을 말해주나 | 어디서 |
|---|---|---|---|
| **① Attention distance** | ViT 블록별 attention의 패치 공간거리 | 낮을수록 국소(CNN다움) — ViT가 locality를 배웠나 | `analyze.py` (ViT) |
| **② ERF** | 중앙 유닛의 입력 gradient 히트맵 | CNN 조밀 / ViT 광역 — 국소성 직접 시각화 | `analyze.py` (both) |
| **③ Equivariance error** | shift 시 분류직전 표현 변화율 ‖g(shift x)−g(x)‖/‖g(x)‖ | CNN≈불변, ViT 급증 — shift 붕괴의 내부근거 | `analyze.py` (both) |
| **④ Train-val gap** | `history.json`의 train_acc−val_acc | 클수록 암기 — 소량서 ViT 열세의 이유 | `compare.py` (history) |

```bash
# full 학습(Phase 1)과 robustness(Phase 2)가 끝난 뒤
python scripts/analyze.py --config configs/cnn.yaml --seed 42
python scripts/analyze.py --config configs/vit.yaml --seed 42
python scripts/compare.py      # ④ 집계 + ①②③ CNN/ViT overlay 그림
```

산출: 각 run에 `analysis.json`, `erf.png`, `equivariance.png`, (`attention_distance.png`, ViT),
집계로 `compare_{train_val_gap,equivariance,attention_distance,erf}.png`.
`run_all.sh` Phase 2에 `analyze.py`가 포함돼 전체 파이프라인에서 자동 실행된다.

## 해석 시 caveat

MNIST는 쉬워서 full에선 둘 다 고성능으로 붙는다(anchor 점). 결론의 무게는 **데이터 효율성 곡선의 왼쪽 끝(소량 데이터)**과 **shift acc 급락 대비**에 실린다. ViT의 열세는 실패가 아니라 **data efficiency / translation-invariance 차이의 재현**이며, 데이터 규모·사전학습·강증강이 있으면 격차가 좁혀지거나 역전될 수 있다.
