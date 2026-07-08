"""학습 엔트리포인트.

    python scripts/train.py --config configs/cnn.yaml
    python scripts/train.py --config configs/vit.yaml --seed 0

데이터 효율성 축:
    python scripts/train.py --config configs/vit.yaml --subset-size 100 --subset-seed 0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt

from src.config import load_config
from src.data import get_dataloaders
from src.engine import fit
from src.models.registry import build_model
from src.utils import count_parameters, get_device, seed_everything


def resolve_budget(cfg: dict) -> dict:
    """subset이 활성일 때 예산을 subset_* 로 스왑한다.

    작은 n은 epoch당 step이 적어 수렴까지 더 많은 epoch이 필요하기 때문.
    """
    if cfg["data"]["subset_size"] is not None:
        cfg["train"]["max_epochs"] = cfg["train"]["subset_max_epochs"]
        cfg["train"]["patience"] = cfg["train"]["subset_patience"]
    return cfg


def run_name(cfg: dict) -> str:
    sz = cfg["data"]["subset_size"]
    tag = "full" if sz is None else f"n{sz}"
    return f"{cfg['name']}_{tag}_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"


def save_curves(history, out_dir: Path, name: str):
    ep = range(1, len(history["val_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, history["train_loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="val")
    ax[0].set_title(f"{name} loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, history["train_acc"], label="train")
    ax[1].plot(ep, history["val_acc"], label="val")
    ax[1].set_title(f"{name} accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--subset-size", type=int, default=None, help="예: 100/500/1000/5000")
    p.add_argument("--subset-seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.subset_size is not None:
        cfg["data"]["subset_size"] = args.subset_size
    if args.subset_seed is not None:
        cfg["data"]["subset_seed"] = args.subset_seed
    cfg = resolve_budget(cfg)

    seed_everything(cfg["seed"])
    device = get_device()
    name = run_name(cfg)
    print(f"device: {device} | run: {name} | "
          f"subset: {cfg['data']['subset_size']} | max_epochs: {cfg['train']['max_epochs']}")

    out_dir = Path(cfg["output"]["dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = build_model(cfg).to(device)
    total, trainable = count_parameters(model)
    print(f"params: total={total:,} | train examples: {len(train_loader.dataset)}")

    history = fit(model, train_loader, val_loader, cfg, device, out_dir / "best.pt")
    history.update({
        "params_total": total,
        "arch": cfg["name"],
        "subset_size": cfg["data"]["subset_size"],
        "subset_seed": cfg["data"]["subset_seed"],
        "seed": cfg["seed"],
    })
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    save_curves(history, out_dir, name)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
