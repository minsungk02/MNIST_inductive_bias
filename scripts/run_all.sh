#!/usr/bin/env bash
# 전체 실험 오케스트레이션 (3090 Ti 서버). tmux 안에서 실행 권장:
#   tmux new -s mnist
#   source .venv/bin/activate
#   bash scripts/run_all.sh 2>&1 | tee "experiments/run_$(date +%Y%m%d_%H%M).log"
#   (Ctrl-b d 로 detach -> 노트북/SSH 꺼도 계속 돌아감)
#
# 개별 run이 실패해도 전체는 계속 진행하고 로그에 FAIL로 남긴다.

mkdir -p experiments
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run() { log "START: $*"; if python "$@"; then log "OK:   $*"; else log "FAIL: $*"; fi; }

CONFIGS=("configs/cnn.yaml" "configs/vit.yaml")
SUBSETS=(100 300 1000 3000 10000)   # 로그 등간격: saturation 전후를 걸침
SUBSET_SEEDS=(0 1 2)                 # 소규모 n 분산 통제
SEED=42

# 0) param band 게이트 (공정성 전제) — 실패 시 전체 중단
python scripts/count_params.py || { echo "param gate failed. abort."; exit 1; }

# --- Phase 1: full 학습 (빠름, 수렴/과적합 결과 + 파이프라인 검증) ---
log "===== Phase 1: full training ====="
for cfg in "${CONFIGS[@]}"; do
  run scripts/train.py --config "$cfg" --seed $SEED
done

# --- Phase 2: full 기반 후처리 (robustness / test / 해석 / 진단지표) ---
log "===== Phase 2: robustness / test / interpret / analyze ====="
for cfg in "${CONFIGS[@]}"; do
  run scripts/eval_robustness.py --config "$cfg" --seed $SEED
  run scripts/evaluate.py       --config "$cfg" --seed $SEED
  run scripts/visualize.py      --config "$cfg" --seed $SEED --n 8
  run scripts/analyze.py        --config "$cfg" --seed $SEED   # 진단지표: equivariance/ERF/attention distance
done

# --- Phase 3: 데이터 효율성 스윕 (가장 긴 구간) ---
log "===== Phase 3: data-efficiency sweep ====="
for cfg in "${CONFIGS[@]}"; do
  for n in "${SUBSETS[@]}"; do
    for ss in "${SUBSET_SEEDS[@]}"; do
      run scripts/train.py --config "$cfg" --seed $SEED --subset-size "$n" --subset-seed "$ss"
    done
  done
done

# --- Phase 4: 집계 (train-val gap 포함) ---
log "===== Phase 4: aggregate ====="
python scripts/compare.py
log "ALL DONE. see experiments/compare_*.png"
