"""★ 학습 전 필수 게이트: 두 모델의 파라미터 수를 나란히 출력한다.

워크플로:
    configs/cnn.yaml 의 channels / configs/vit.yaml 의 embed_dim·depth 조정
        -> python scripts/count_params.py
        -> 같은 band(비율 ~1.x) 착지 확인
        -> 그제서야 train

param을 안 맞추면 '용량이 커서 이긴 것'과 'prior가 좋아서 이긴 것'이 섞인다.
그래서 이 스크립트를 통과하기 전엔 학습하지 않는다.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.models.registry import build_model
from src.utils import count_parameters, seed_everything


def report(config_path: str):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    model = build_model(cfg)
    total, trainable = count_parameters(model)
    return cfg["name"], total, trainable


def main():
    configs = ["configs/cnn.yaml", "configs/vit.yaml"]
    results = [report(c) for c in configs]

    print(f"{'model':<10}{'total':>12}{'trainable':>12}")
    print("-" * 34)
    for name, total, trainable in results:
        print(f"{name:<10}{total:>12,}{trainable:>12,}")

    counts = [t for _, t, _ in results]
    ratio = max(counts) / min(counts)
    print("-" * 34)
    print(f"band ratio (max/min): {ratio:.3f}")
    if ratio <= 1.3:
        print("OK: 같은 band. 공정 비교 가능.")
    else:
        print("WARN: band 차이가 큼. channels / embed_dim·depth 조정 권장 (목표 <= 1.3).")


if __name__ == "__main__":
    main()
