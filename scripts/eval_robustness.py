"""Test-time robustness 스윕 — 학습 끝난 뒤 실행.

    python scripts/eval_robustness.py --config configs/cnn.yaml --seed 42

best.pt(val_loss 최소)를 로드해, base.yaml의 robustness 스윕값에 대해
clean 대비 perturbed test accuracy 를 뽑는다. 결과는 run 폴더에 robustness.json 으로 저장.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torchvision import datasets, transforms

from src.config import load_config
from src.models.registry import build_model
from src.robustness import make_perturb
from src.utils import get_device, seed_everything


@torch.no_grad()
def eval_with_perturb(model, raw_test, perturb_fn, device, batch_size=256):
    model.eval()
    correct, n = 0, 0
    xs, ys = [], []
    for img01, y in raw_test:
        xs.append(perturb_fn(img01)); ys.append(y)
        if len(xs) == batch_size:
            correct, n = _flush(model, xs, ys, device, correct, n); xs, ys = [], []
    if xs:
        correct, n = _flush(model, xs, ys, device, correct, n)
    return correct / n


def _flush(model, xs, ys, device, correct, n):
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys).to(device)
    pred = model(x).argmax(1)
    return correct + (pred == y).sum().item(), n + len(ys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(cfg["seed"])
    device = get_device()

    # full 학습 체크포인트 로드 (robustness는 full 모델로 평가)
    run = f"{cfg['name']}_full_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"
    ckpt_path = Path(cfg["output"]["dir"]) / run / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    # 정규화 이전 [0,1] test set (perturb는 여기에 적용)
    raw_test = datasets.MNIST(cfg["data"]["root"], train=False, download=True,
                              transform=transforms.ToTensor())
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]

    mag_key = {"shift": "shift_px", "rotation": "rotation_deg", "noise": "noise_std"}
    results = {}
    for kind in ["shift", "rotation", "noise"]:
        mags = cfg["robustness"][mag_key[kind]]
        results[kind] = []
        for m in mags:
            fn = make_perturb(kind, m, mean, std, seed=cfg["seed"])
            acc = eval_with_perturb(model, raw_test, fn, device)
            results[kind].append({"mag": m, "acc": acc})
            print(f"{run} | {kind:>8}={m:<5} acc={acc:.4f}")

    out = Path(cfg["output"]["dir"]) / run / "robustness.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
