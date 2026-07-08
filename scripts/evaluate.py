"""최종 test set 평가 — 개발이 다 끝난 뒤 딱 한 번만.

    python scripts/evaluate.py --config configs/cnn.yaml --seed 42

test set 은 학습/모델선택 과정에서 절대 보지 않는다 (leakage 방지).
full best.pt(val_loss 최소)를 로드해 test accuracy/loss 를 뽑는다.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn

from src.config import load_config
from src.data import get_dataloaders
from src.engine import evaluate
from src.models.registry import build_model
from src.utils import get_device, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(cfg["seed"])
    device = get_device()

    run = f"{cfg['name']}_full_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"
    ckpt = torch.load(Path(cfg["output"]["dir"]) / run / "best.pt", map_location=device)

    _, _, test_loader = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"{run} | best epoch {ckpt['epoch']+1} | val_loss {ckpt['val_loss']:.4f} "
          f"| TEST loss {test_loss:.4f} acc {test_acc:.4f}")


if __name__ == "__main__":
    main()
