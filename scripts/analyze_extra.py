"""입력 섭동 스윕 기반 추가 진단 (재학습 없음 — 무거운 eval 루프라 analyze.py와 분리).

    # 2D translation 민감도: 전체 run (test 2000장 서브샘플)
    python scripts/analyze_extra.py --which translation2d --all --max-images 2000
    # full run만 test 10k 전체로
    python scripts/analyze_extra.py --which translation2d --runs "*_full_*" --max-images 10000
    # Fourier 민감도 (full run만, eps는 스모크 테스트로 캘리브레이션)
    python scripts/analyze_extra.py --which fourier --runs "*_full_*" --eps 4.0
    # 첫 레이어 필터 비교 (full 체크포인트 가중치만)
    python scripts/analyze_extra.py --which filters

지표:
  - translation2d : acc(dx,dy), (dx,dy) ∈ [-R,R]² 히트맵.
                    기존 robustness.json의 1D 대각 shift 곡선을 2D로 확장 —
                    'CNN 고원 vs ViT 원뿔붕괴'가 한 장에 보인다.
                    -> <run>/translation2d.json + translation2d.png
  - fourier       : 단일 Fourier 기저 섭동별 오류율 28×28 맵 (Yin et al. 2019).
                    Conv=high-pass/MSA=low-pass(frequency_response)의 입력공간 버전.
                    -> <run>/fourier.json + fourier.png
  - filters       : CNN conv1(3×3×40) vs ViT patch-embed(7×7×64) 필터 몽타주.
                    -> experiments/compare_first_layer_filters.png

섭동 규약은 src/robustness.py와 동일: [0,1]에서 섭동(fill=0) -> clamp -> normalize.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.utils import get_device, seed_everything

# analyze.py의 run 로딩 헬퍼 재사용
from scripts.analyze import iter_run_dirs, load_run


# ------------------------------------------------------------------ #
# 공통: [0,1] test 텐서 일괄 적재 (batched perturb가 per-image보다 수십 배 빠름)
# ------------------------------------------------------------------ #
def load_raw_test_tensor(root: str, max_images: int):
    raw = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    n = min(max_images, len(raw))
    imgs = torch.stack([raw[i][0] for i in range(n)])          # [N,1,28,28] in [0,1]
    labels = raw.targets[:n].clone()
    return imgs, labels


@torch.no_grad()
def eval_batch_acc(model, x, labels, device, batch: int = 512) -> float:
    """정규화 완료된 텐서 [N,1,28,28] 에 대한 accuracy."""
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        pred = model(x[i:i + batch].to(device, non_blocking=True)).argmax(1).cpu()
        correct += (pred == labels[i:i + batch]).sum().item()
    return correct / len(x)


# ------------------------------------------------------------------ #
# 1) 2D translation 민감도: acc(dx, dy)
# ------------------------------------------------------------------ #
@torch.no_grad()
def translation2d_grid(model, imgs01, labels, mean, std, device, rng: int = 4) -> np.ndarray:
    """acc[dy+R, dx+R]. dx>0=오른쪽, dy>0=아래 (이미지 좌표)."""
    R = rng
    acc = np.zeros((2 * R + 1, 2 * R + 1))
    for iy, dy in enumerate(range(-R, R + 1)):
        for ix, dx in enumerate(range(-R, R + 1)):
            x = imgs01
            if dx != 0 or dy != 0:
                x = TF.affine(x, angle=0.0, translate=[dx, dy], scale=1.0,
                              shear=[0.0, 0.0], fill=0.0)
            x = TF.normalize(x, [mean], [std])
            acc[iy, ix] = eval_batch_acc(model, x, labels, device)
    return acc


def run_translation2d(run_dir: Path, device, imgs01, labels, rng: int):
    model, cfg, _ = load_run(run_dir, device)
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    acc = translation2d_grid(model, imgs01, labels, mean, std, device, rng)
    center = acc[rng, rng]
    print(f"{run_dir.name}: clean={center:.4f} | worst corner={acc.min():.4f}")

    out = {"arch": cfg["name"], "run": run_dir.name, "range_px": rng,
           "max_images": int(len(labels)),
           "subset_size": cfg["data"].get("subset_size"),
           "subset_seed": cfg["data"].get("subset_seed", 0),
           "acc": acc.tolist()}
    with open(run_dir / "translation2d.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(acc, cmap="viridis", vmin=0, vmax=1, origin="upper",
                   extent=[-rng - 0.5, rng + 0.5, rng + 0.5, -rng - 0.5])
    ax.set_xlabel("dx (px, right+)"); ax.set_ylabel("dy (px, down+)")
    ax.set_title(f"{run_dir.name}\nacc under 2D shift (clean={center:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(run_dir / "translation2d.png", dpi=120); plt.close(fig)


# ------------------------------------------------------------------ #
# 2) Fourier 민감도 히트맵 (Yin et al. 2019)
# ------------------------------------------------------------------ #
def fourier_basis(h: int, w: int, i: int, j: int) -> torch.Tensor:
    """(i,j) 주파수의 실수 단위(L2=1) Fourier 기저 [h,w]."""
    spec = torch.zeros(h, w, dtype=torch.complex64)
    spec[i, j] += 1.0
    spec[(-i) % h, (-j) % w] += 1.0            # conjugate 대칭 -> 실수 신호
    b = torch.fft.ifft2(spec).real
    return b / b.norm().clamp_min(1e-12)


@torch.no_grad()
def fourier_heatmap(model, imgs01, labels, eps, mean, std, device, seed: int = 0) -> np.ndarray:
    """주파수 (i,j)별 오류율 [28,28] (DC가 (0,0), 표시 시 fftshift 권장).

    conjugate 대칭인 (i,j)와 (-i,-j)는 같은 실수 기저 -> 절반만 평가하고 미러링.
    이미지별 부호 ±1은 seed 고정 (모든 주파수/모델에서 동일한 부호 -> 공정 비교).
    """
    h, w = imgs01.shape[-2:]
    g = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (len(imgs01), 1, 1, 1), generator=g) * 2 - 1).float()
    err = np.full((h, w), np.nan)
    done = set()
    total = 0
    for i in range(h):
        for j in range(w):
            if (i, j) in done:
                continue
            pair = ((-i) % h, (-j) % w)
            done.add((i, j)); done.add(pair)
            basis = fourier_basis(h, w, i, j)
            x = (imgs01 + eps * signs * basis).clamp(0.0, 1.0)
            x = TF.normalize(x, [mean], [std])
            e = 1.0 - eval_batch_acc(model, x, labels, device)
            err[i, j] = e
            err[pair] = e
            total += 1
    print(f"  evaluated {total} unique frequencies")
    return err


def run_fourier(run_dir: Path, device, imgs01, labels, eps: float, seed: int):
    model, cfg, _ = load_run(run_dir, device)
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    print(f"{run_dir.name}: fourier sweep (eps={eps}) ...")
    err = fourier_heatmap(model, imgs01, labels, eps, mean, std, device, seed)
    clean_err = 1.0 - eval_batch_acc(
        model, TF.normalize(imgs01, [mean], [std]), labels, device)
    print(f"  clean err={clean_err:.4f} | max perturbed err={np.nanmax(err):.4f}")

    out = {"arch": cfg["name"], "run": run_dir.name, "eps": eps,
           "max_images": int(len(labels)), "clean_err": clean_err,
           "err": err.tolist()}
    with open(run_dir / "fourier.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    disp = np.fft.fftshift(err)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(disp, cmap="magma")
    ax.set_title(f"{run_dir.name}\nFourier sensitivity (eps={eps}, err rate)")
    ax.set_xlabel("freq x (center=DC)"); ax.set_ylabel("freq y")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(run_dir / "fourier.png", dpi=120); plt.close(fig)


# ------------------------------------------------------------------ #
# 3) 첫 레이어 필터 몽타주 (가중치만)
# ------------------------------------------------------------------ #
def _montage(w: torch.Tensor, cols: int) -> np.ndarray:
    """[K,1,k,k] -> 필터별 정규화 후 격자 몽타주 (1px 간격)."""
    K, _, kh, kw = w.shape
    rows = (K + cols - 1) // cols
    canvas = np.full((rows * (kh + 1) + 1, cols * (kw + 1) + 1), np.nan)
    for idx in range(K):
        f = w[idx, 0].numpy()
        lo, hi = f.min(), f.max()
        f = (f - lo) / max(hi - lo, 1e-9)
        r, c = divmod(idx, cols)
        canvas[r * (kh + 1) + 1:r * (kh + 1) + 1 + kh,
               c * (kw + 1) + 1:c * (kw + 1) + 1 + kw] = f
    return canvas


def run_filters(exp_dir: Path, device):
    panels = []
    for d in sorted(exp_dir.glob("*_full_*")):
        if not (d / "best.pt").exists():
            continue
        model, cfg, _ = load_run(d, "cpu")
        if cfg["architecture"]["type"] == "simple_cnn":
            w = model.features[0].weight.detach()          # [40,1,3,3]
            panels.append((f"CNN conv1 3x3 x{w.shape[0]}", _montage(w, cols=8)))
        else:
            w = model.patch_embed.proj.weight.detach()     # [64,1,7,7]
            panels.append((f"ViT patch-embed 7x7 x{w.shape[0]}", _montage(w, cols=8)))
    if not panels:
        print("no full checkpoints found"); return
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, canvas) in zip(axes, panels):
        cmap = plt.get_cmap("gray").copy()
        cmap.set_bad("white")
        ax.imshow(canvas, cmap=cmap, interpolation="nearest")
        ax.set_title(title); ax.axis("off")
    fig.suptitle("First-layer filters: local edge detectors vs patch templates")
    fig.tight_layout()
    out = exp_dir / "compare_first_layer_filters.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved -> {out}")


# ------------------------------------------------------------------ #
def select_run_dirs(exp_dir: Path, args) -> list:
    if args.run_dir:
        return [Path(args.run_dir)]
    if args.runs:
        return [d for d in sorted(exp_dir.glob(args.runs)) if (d / "best.pt").exists()]
    if args.all:
        return list(iter_run_dirs(exp_dir))
    return [d for d in sorted(exp_dir.glob("*_full_*")) if (d / "best.pt").exists()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", required=True,
                   choices=["translation2d", "fourier", "filters"])
    p.add_argument("--run-dir", help="단일 run 디렉토리")
    p.add_argument("--runs", help="run 이름 glob (예: '*_full_*')")
    p.add_argument("--all", action="store_true", help="모든 run")
    p.add_argument("--exp-dir", default="./experiments")
    p.add_argument("--max-images", type=int, default=2000)
    p.add_argument("--range-px", type=int, default=4, help="translation2d 셔프트 범위")
    p.add_argument("--eps", type=float, default=4.0, help="fourier 섭동 L2 크기")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    exp_dir = Path(args.exp_dir)

    if args.which == "filters":
        run_filters(exp_dir, device)
        return

    run_dirs = select_run_dirs(exp_dir, args)
    if not run_dirs:
        print("no matching runs"); return
    # test 텐서는 전 run 공통 (cfg["data"]["root"]는 모두 ./data)
    _, cfg0, _ = load_run(run_dirs[0], "cpu")
    imgs01, labels = load_raw_test_tensor(cfg0["data"]["root"], args.max_images)
    print(f"{len(run_dirs)} runs | {len(labels)} test images")

    for i, rd in enumerate(run_dirs):
        if len(run_dirs) > 1:
            print(f"[{i + 1}/{len(run_dirs)}] ", end="")
        if args.which == "translation2d":
            run_translation2d(rd, device, imgs01, labels, args.range_px)
        else:
            run_fourier(rd, device, imgs01, labels, args.eps, args.seed)


if __name__ == "__main__":
    main()
