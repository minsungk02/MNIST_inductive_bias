"""데이터 효율 그래프(주석형) — 기존 compare_data_efficiency.png 를 그대로 덮어씀.

    python scripts/plot_data_efficiency.py                 # 목표 정확도 0.90 기준
    python scripts/plot_data_efficiency.py --target 0.95

기존 선그래프에 '핵심 비교 포인트'를 눈에 박아준다:
  - 목표 정확도(기본 0.90) 가로 기준선
  - 각 모델이 그 선을 넘는 데이터 크기(수직 점선) + 'reaches @ ~N imgs' 주석
  - "ViT needs ~Nx more data" 강조 박스 (= 데이터효율 격차의 배수)
  - n=100 세로 격차(+XX%p) 양방향 화살표
그림 텍스트는 전부 영어(서버 한글 폰트 없어도 안 깨지게), 백엔드 Agg(헤드리스).
출력 파일명은 기존과 동일(experiments/compare_data_efficiency.png) — 그대로 교체된다.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False


def load_histories(exp_dir: Path):
    runs = []
    for hf in sorted(exp_dir.glob("*/history.json")):
        with open(hf, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def crossing(sizes, means, target):
    """means가 target을 처음 넘는 지점을 log(n) 선형보간으로 추정."""
    for i in range(1, len(sizes)):
        if means[i - 1] < target <= means[i]:
            l0, l1 = np.log10(sizes[i - 1]), np.log10(sizes[i])
            frac = (target - means[i - 1]) / (means[i] - means[i - 1])
            return 10 ** (l0 + frac * (l1 - l0))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.90, help="목표 정확도 기준선")
    ap.add_argument("--exp-dir", default="./experiments")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir)

    runs = load_histories(exp_dir)
    if not runs:
        print("no history.json under", exp_dir); return

    agg = defaultdict(lambda: defaultdict(list))     # arch -> size -> [best_val_acc]
    for h in runs:
        size = h.get("subset_size") or 54000
        agg[h["arch"]][size].append(max(h["val_acc"]))

    colors = {"cnn": "tab:blue", "vit": "tab:orange"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    cross, all_sizes = {}, None
    for arch in sorted(agg):
        sizes = sorted(agg[arch])
        means = [float(np.mean(agg[arch][s])) for s in sizes]
        stds = [float(np.std(agg[arch][s])) for s in sizes]
        c = colors.get(arch, "gray")
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, color=c,
                    label=arch.upper(), lw=2)
        cross[arch] = crossing(sizes, means, args.target)
        all_sizes = sizes

    # 목표 정확도 가로선
    ax.axhline(args.target, color="gray", ls="--", lw=1)
    ax.text(all_sizes[0], args.target + 0.006, f"target acc = {args.target:.2f}",
            fontsize=9, color="gray")

    # 각 모델이 목표선을 넘는 지점: 수직 점선 + 주석
    for arch, xc in cross.items():
        if xc:
            c = colors.get(arch, "gray")
            ax.axvline(xc, color=c, ls=":", lw=1.5, alpha=0.8)
            ax.annotate(f"{arch.upper()} reaches {args.target:.2f}\n@ ~{xc:,.0f} imgs",
                        xy=(xc, args.target), xytext=(xc, args.target - 0.20),
                        fontsize=9, color=c, ha="center",
                        arrowprops=dict(arrowstyle="->", color=c))

    # "ViT needs ~Nx more data" 강조 박스
    if cross.get("cnn") and cross.get("vit"):
        ratio = cross["vit"] / cross["cnn"]
        ax.text(0.5, 0.07, f"ViT needs ~{ratio:.0f}x more data to reach {args.target:.2f}",
                transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round", fc="#fff3cd", ec="#e0a800"))

    # n=100 세로 격차 화살표
    if 100 in agg.get("cnn", {}) and 100 in agg.get("vit", {}):
        c100 = float(np.mean(agg["cnn"][100]))
        v100 = float(np.mean(agg["vit"][100]))
        ax.annotate("", xy=(100, c100), xytext=(100, v100),
                    arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.5))
        ax.text(118, (c100 + v100) / 2, f"+{(c100 - v100) * 100:.0f}%p\n@ 100",
                color="tab:red", fontsize=9, va="center")

    ax.set_xscale("log")
    ax.set_xlabel("training images (log scale)")
    ax.set_ylabel("best val accuracy")
    ax.set_title("Data efficiency: CNN vs ViT")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout()
    out = exp_dir / "compare_data_efficiency.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    msg = ", ".join(f"{k}~{v:,.0f}" for k, v in cross.items() if v)
    print(f"saved -> {out}  | crossing@{args.target}: {msg}")


if __name__ == "__main__":
    main()
