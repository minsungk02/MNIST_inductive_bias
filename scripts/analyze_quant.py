"""정량 보강 지표 (재학습 없음 — 저장된 best.pt에 forward 1회).

발표 '정량' 파트를 눈에 확 들어오게 보강하는 두 지표:

    # 1) 신뢰도/캘리브레이션: "ViT는 맞혀도 덜 확신한다"를 숫자·그림으로
    python scripts/analyze_quant.py --which calibration                 # full CNN/ViT
    python scripts/analyze_quant.py --which calibration --all           # 전 run 스윕

    # 2) shift 오류분해: "4px 밀 때 ViT가 어느 숫자를 새로 틀리나"
    python scripts/analyze_quant.py --which shift_errors --shift 4

지표 설명:
  - calibration : 각 예측의 확신도(max softmax) vs 실제 정확도.
                  ECE(Expected Calibration Error)=|정확도-확신도|의 가중평균(클수록 보정 나쁨).
                  reliability diagram(확신도 대비 실제 정확도) + 확신도 히스토그램(맞음/틀림).
                  -> experiments/compare_calibration.png + <run>/calibration.json
  - shift_errors: clean vs (dx,dy)=(k,k) shift 예측을 비교.
                  클래스별 clean/shift 정확도 + 'clean 정답 -> shift 오답' 플립 몽타주.
                  -> experiments/compare_shift_errors.png, experiments/shift_flip_montage.png

섭동 규약은 src/robustness.py와 동일: [0,1]에서 fill=0 shift -> 정규화.
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
from scripts.analyze import iter_run_dirs, load_run


# ------------------------------------------------------------------ #
# 공통 로더
# ------------------------------------------------------------------ #
def load_raw_test(root: str, max_images: int | None = None):
    """[0,1] test 텐서 [N,1,28,28] + 라벨 (섭동을 정규화 전에 적용하려고 raw로)."""
    raw = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    n = len(raw) if max_images is None else min(max_images, len(raw))
    imgs = torch.stack([raw[i][0] for i in range(n)])          # [N,1,28,28] in [0,1]
    labels = raw.targets[:n].clone().numpy()
    return imgs, labels


@torch.no_grad()
def forward_probs(model, imgs01, mean, std, device, batch: int = 512) -> np.ndarray:
    """정규화 후 softmax 확률 [N,10] 반환."""
    model.eval()
    out = []
    for i in range(0, len(imgs01), batch):
        x = TF.normalize(imgs01[i:i + batch], [mean], [std]).to(device)
        p = torch.softmax(model(x), dim=1).cpu()
        out.append(p)
    return torch.cat(out, 0).numpy()


def select_runs(exp_dir: Path, args) -> list:
    if args.run_dir:
        return [Path(args.run_dir)]
    if args.runs:
        return [d for d in sorted(exp_dir.glob(args.runs)) if (d / "best.pt").exists()]
    if args.all:
        return list(iter_run_dirs(exp_dir))
    return [d for d in sorted(exp_dir.glob("*_full_*")) if (d / "best.pt").exists()]


# ================================================================== #
# 1) 신뢰도 / 캘리브레이션
# ================================================================== #
def compute_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """confidence=max prob, ECE, 그리고 reliability용 bin 통계."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, N = 0.0, len(conf)
    bins = []  # (center, acc, avg_conf, frac)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c > 0:
            acc, avg = float(correct[m].mean()), float(conf[m].mean())
            ece += (c / N) * abs(avg - acc)
            bins.append((0.5 * (lo + hi), acc, avg, c / N))
        else:
            bins.append((0.5 * (lo + hi), np.nan, np.nan, 0.0))
    return {"ece": float(ece), "mean_conf": float(conf.mean()),
            "acc": float(correct.mean()),
            "conf_correct": conf[correct == 1], "conf_wrong": conf[correct == 0],
            "bins": bins, "n_bins": n_bins}


def _plot_reliability(ax, cal, arch):
    """reliability diagram: 파란 막대=실제 정확도, 빨간 빗금=과확신 gap, 대각선=완벽 보정."""
    centers = [b[0] for b in cal["bins"]]
    accs = [b[1] for b in cal["bins"]]
    w = 1.0 / cal["n_bins"]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7, label="완벽 보정")
    ax.bar(centers, accs, width=w, color="tab:blue", alpha=0.85,
           edgecolor="white", label="실제 정확도")
    # gap = 확신도(=bin 중심) - 실제 정확도 (양수면 과확신)
    for cx, acc in zip(centers, accs):
        if not np.isnan(acc) and cx > acc:
            ax.bar(cx, cx - acc, width=w, bottom=acc, color="tab:red",
                   alpha=0.35, hatch="///", edgecolor="tab:red")
    ax.axvline(cal["mean_conf"], color="tab:orange", lw=2, ls=":",
               label=f"평균 확신도 {cal['mean_conf']:.2f}")
    ax.axvline(cal["acc"], color="tab:green", lw=2, ls=":",
               label=f"정확도 {cal['acc']:.2f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("확신도 (max softmax)"); ax.set_ylabel("실제 정확도")
    ax.set_title(f"{arch.upper()}  ·  ECE = {cal['ece']:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)


def _plot_conf_hist(ax, cal, arch):
    """확신도 히스토그램: 맞은 예측(초록) vs 틀린 예측(빨강)."""
    bins = np.linspace(0, 1, 26)
    ax.hist(cal["conf_correct"], bins=bins, color="tab:green", alpha=0.6,
            label=f"맞음 ({len(cal['conf_correct'])})")
    ax.hist(cal["conf_wrong"], bins=bins, color="tab:red", alpha=0.7,
            label=f"틀림 ({len(cal['conf_wrong'])})")
    ax.set_xlim(0, 1); ax.set_xlabel("확신도"); ax.set_ylabel("예측 수 (log)")
    ax.set_yscale("log")
    ax.set_title(f"{arch.upper()} 확신도 분포", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)


def run_calibration(run_dirs, device, exp_dir: Path):
    cals = {}  # arch -> cal (full 기준 비교 그림)
    _, cfg0, _ = load_run(run_dirs[0], "cpu")
    imgs01, labels = load_raw_test(cfg0["data"]["root"])
    for rd in run_dirs:
        model, cfg, _ = load_run(rd, device)
        mean, std = cfg["data"]["mean"], cfg["data"]["std"]
        probs = forward_probs(model, imgs01, mean, std, device)
        cal = compute_calibration(probs, labels)
        cw = cal["conf_wrong"].mean() if len(cal["conf_wrong"]) else float("nan")
        print(f"{rd.name}: acc={cal['acc']:.4f} | mean_conf={cal['mean_conf']:.4f} | "
              f"ECE={cal['ece']:.4f} | 틀린예측 평균확신 {cw:.3f}")
        with open(rd / "calibration.json", "w", encoding="utf-8") as f:
            json.dump({"run": rd.name, "arch": cfg["name"], "ece": cal["ece"],
                       "mean_conf": cal["mean_conf"], "acc": cal["acc"],
                       "subset_size": cfg["data"].get("subset_size")}, f,
                      ensure_ascii=False, indent=2)
        if cfg["data"].get("subset_size") is None:   # full run만 compare 그림에
            cals[cfg["name"]] = cal

    if len(cals) >= 1:
        archs = sorted(cals)
        fig, axes = plt.subplots(2, len(archs), figsize=(5.4 * len(archs), 8.4))
        axes = np.atleast_2d(axes)
        if len(archs) == 1:
            axes = axes.reshape(2, 1)
        for c, arch in enumerate(archs):
            _plot_reliability(axes[0, c], cals[arch], arch)
            _plot_conf_hist(axes[1, c], cals[arch], arch)
        fig.suptitle("신뢰도(캘리브레이션): 확신도 vs 실제 정확도  —  빨간 빗금 = 과확신 gap, ECE 클수록 보정 나쁨",
                     fontsize=13)
        fig.tight_layout()
        out = exp_dir / "compare_calibration.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"saved -> {out}")


# ================================================================== #
# 2) shift 오류분해
# ================================================================== #
def run_shift_errors(run_dirs, device, exp_dir: Path, k: int, n_show: int):
    _, cfg0, _ = load_run(run_dirs[0], "cpu")
    imgs01, labels = load_raw_test(cfg0["data"]["root"])
    shifted01 = TF.affine(imgs01, angle=0.0, translate=[k, k], scale=1.0,
                          shear=[0.0, 0.0], fill=0.0)

    per_arch = {}   # arch -> dict(clean_acc[10], shift_acc[10], flips)
    for rd in run_dirs:
        model, cfg, _ = load_run(rd, device)
        arch = cfg["name"]
        mean, std = cfg["data"]["mean"], cfg["data"]["std"]
        clean_pred = forward_probs(model, imgs01, mean, std, device).argmax(1)
        shift_pred = forward_probs(model, shifted01, mean, std, device).argmax(1)
        clean_ok = clean_pred == labels
        shift_ok = shift_pred == labels
        clean_acc = np.array([clean_ok[labels == c].mean() for c in range(10)])
        shift_acc = np.array([shift_ok[labels == c].mean() for c in range(10)])
        flip = np.where(clean_ok & ~shift_ok)[0]           # clean 정답 -> shift 오답
        per_arch[arch] = {"clean_acc": clean_acc, "shift_acc": shift_acc,
                          "flip_idx": flip, "shift_pred": shift_pred}
        print(f"{rd.name}: clean {clean_ok.mean():.3f} -> shift(k={k}) {shift_ok.mean():.3f} "
              f"| 새로 틀린 예제 {len(flip)}장 ({len(flip)/len(labels)*100:.1f}%)")

    # --- (A) 클래스별 clean vs shift 정확도 (arch별 패널) ---
    archs = sorted(per_arch)
    fig, axes = plt.subplots(1, len(archs), figsize=(6.2 * len(archs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(10); w = 0.4
    for ax, arch in zip(axes, archs):
        d = per_arch[arch]
        ax.bar(x - w / 2, d["clean_acc"], w, color="tab:gray", alpha=0.7, label="clean")
        ax.bar(x + w / 2, d["shift_acc"], w, color="tab:red", alpha=0.85,
               label=f"{k}px shift")
        ax.set_xticks(x); ax.set_xlabel("숫자 클래스"); ax.set_ylim(0, 1)
        ax.set_title(f"{arch.upper()}: clean {d['clean_acc'].mean():.2f} "
                     f"-> shift {d['shift_acc'].mean():.2f}")
        ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("클래스별 정확도")
    fig.suptitle(f"{k}px 대각 shift에서 어느 숫자가 무너지나 (clean vs shift)", fontsize=13)
    fig.tight_layout()
    out = exp_dir / "compare_shift_errors.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"saved -> {out}")

    # --- (B) 'clean 정답 -> shift 오답' 플립 몽타주 (마지막 run 기준, 보통 ViT) ---
    tgt = "vit" if "vit" in per_arch else archs[-1]
    d = per_arch[tgt]
    flips = d["flip_idx"][:n_show]
    if len(flips):
        cols = min(n_show, len(flips))
        fig, axes = plt.subplots(1, cols, figsize=(1.5 * cols, 2.0))
        axes = np.atleast_1d(axes)
        for ax, idx in zip(axes, flips):
            ax.imshow(shifted01[idx, 0].numpy(), cmap="gray")
            ax.set_title(f"{labels[idx]}->{d['shift_pred'][idx]}", fontsize=10, color="tab:red")
            ax.axis("off")
        fig.suptitle(f"{tgt.upper()}: {k}px 밀었더니 새로 틀린 예 (정답->예측)", fontsize=12)
        fig.tight_layout()
        out = exp_dir / "shift_flip_montage.png"
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"saved -> {out}")


# ------------------------------------------------------------------ #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", required=True, choices=["calibration", "shift_errors"])
    p.add_argument("--run-dir", help="단일 run 디렉토리")
    p.add_argument("--runs", help="run glob (예: '*_full_*')")
    p.add_argument("--all", action="store_true", help="모든 run")
    p.add_argument("--exp-dir", default="./experiments")
    p.add_argument("--shift", type=int, default=4, help="shift_errors 대각 이동 px")
    p.add_argument("--n-show", type=int, default=10, help="플립 몽타주 표본 수")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()
    exp_dir = Path(args.exp_dir)
    run_dirs = select_runs(exp_dir, args)
    if not run_dirs:
        print("no matching runs"); return
    print(f"{len(run_dirs)} run(s): {[d.name for d in run_dirs]}")

    if args.which == "calibration":
        run_calibration(run_dirs, device, exp_dir)
    else:
        run_shift_errors(run_dirs, device, exp_dir, args.shift, args.n_show)


if __name__ == "__main__":
    main()
