"""저장된 full 체크포인트 기반 진단 지표 (재학습 없음).

    python scripts/analyze.py --config configs/cnn.yaml --seed 42
    python scripts/analyze.py --config configs/vit.yaml --seed 42

뽑는 지표 (best.pt 로딩만으로):
  - equivariance error (both)    shift 내부표현 변화율      -> shift 붕괴 근거
  - ERF (both)                   중앙유닛 입력 gradient      -> 국소성 시각화
  - attention distance (vit)     블록별 attention 공간거리   -> locality 학습 여부
결과: experiments/<run>/analysis.json + erf.png / equivariance.png / attention_distance.png
train-val gap(4)은 모델이 필요없어 compare.py 에서 history 로 집계한다.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.analysis import attention_distance, effective_receptive_field, equivariance_error
from src.config import load_config
from src.data import get_dataloaders
from src.models.registry import build_model
from src.utils import get_device, seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=20, help="attention/ERF 표본 배치 수")
    p.add_argument("--max-images", type=int, default=2000, help="equivariance 표본 이미지 수")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(cfg["seed"])
    device = get_device()
    arch = cfg["architecture"]["type"]

    run = f"{cfg['name']}_full_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"
    run_dir = Path(cfg["output"]["dir"]) / run
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {run} (best epoch {ckpt['epoch']+1})")

    # 데이터: 정규화 test(attention/ERF) + 정규화이전 raw test(equivariance의 shift용)
    _, _, test_loader = get_dataloaders(cfg)
    raw_test = datasets.MNIST(cfg["data"]["root"], train=False, download=True,
                              transform=transforms.ToTensor())
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    shifts = cfg["robustness"]["shift_px"]

    analysis = {"arch": cfg["name"], "run": run, "shifts": list(shifts)}

    # --- 3) equivariance error (both) ---
    eq = equivariance_error(model, raw_test, shifts, mean, std, device, args.max_images)
    analysis["equivariance"] = eq
    print("equivariance:", {k: round(v, 4) for k, v in eq.items()})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(eq.keys()), list(eq.values()), marker="o")
    ax.set_xlabel("shift (px)"); ax.set_ylabel("||g(shift x)-g(x)|| / ||g(x)||")
    ax.set_title(f"{run} equivariance error"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir / "equivariance.png", dpi=120); plt.close(fig)

    # --- 2) ERF (both) ---
    erf = effective_receptive_field(model, test_loader, device, arch, args.max_batches)
    analysis["erf"] = erf.tolist()
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(erf, cmap="viridis"); ax.set_title(f"{run} ERF"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(run_dir / "erf.png", dpi=120); plt.close(fig)

    # --- 1) attention distance (vit only) ---
    if arch == "vit":
        grid = cfg["data"].get("img_size", 28) // cfg["architecture"]["patch_size"]
        ad = attention_distance(model, test_loader, device, grid, args.max_batches)
        analysis["attention_distance"] = ad
        print("attention distance (per block):", [round(v, 3) for v in ad])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(1, len(ad) + 1), ad)
        ax.set_xlabel("transformer block"); ax.set_ylabel("mean attn distance (patches)")
        ax.set_title(f"{run} attention distance"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(run_dir / "attention_distance.png", dpi=120); plt.close(fig)
    else:
        analysis["attention_distance"] = None

    with open(run_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"saved -> {run_dir}/analysis.json (+ png)")


if __name__ == "__main__":
    main()
