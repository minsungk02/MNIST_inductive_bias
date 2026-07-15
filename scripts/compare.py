"""모든 run 을 모아 비교 곡선/표를 생성.

    python scripts/compare.py

experiments/ 아래를 스캔해서:
- 수렴 비교: full run의 val_loss/val_acc overlay (CNN vs ViT)
- 데이터 효율성: subset 크기별 best_val_acc 곡선 (seed 평균 ± std, error band)
- robustness: shift/rotation/noise 곡선 (있으면)
- [진단] train-val gap / equivariance / attention distance / ERF (analyze.py 산출이 있으면)
콘솔 표 + experiments/*.png 로 출력.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt

from src.analysis import train_val_gap

EXP = Path("./experiments")


def load_histories():
    runs = []
    for hf in sorted(EXP.glob("*/history.json")):
        with open(hf, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def load_analyses():
    """analyze.py 가 남긴 진단 결과들 (full run별 analysis.json)."""
    out = []
    for af in sorted(EXP.glob("*_full_*/analysis.json")):
        with open(af, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def plot_convergence(runs):
    full = [h for h in runs if h.get("subset_size") is None]
    if not full:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for h in full:
        ep = range(1, len(h["val_loss"]) + 1)
        lbl = f"{h['arch']} (s{h['seed']})"
        ax[0].plot(ep, h["val_loss"], label=lbl)
        ax[1].plot(ep, h["val_acc"], label=lbl)
    ax[0].set_title("val loss (full)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].set_title("val accuracy (full)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(EXP / "compare_convergence.png", dpi=120); plt.close(fig)


def plot_data_efficiency(runs):
    # arch -> size -> [best_val_acc over seeds]
    agg = defaultdict(lambda: defaultdict(list))
    for h in runs:
        sz = h.get("subset_size")
        size = sz if sz is not None else 54000
        best_val_acc = max(h["val_acc"])  # 수렴 지점 근사
        agg[h["arch"]][size].append(best_val_acc)
    if not agg:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for arch, per_size in agg.items():
        sizes = sorted(per_size)
        means = [np.mean(per_size[s]) for s in sizes]
        stds = [np.std(per_size[s]) for s in sizes]
        ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=arch)
    ax.set_xscale("log")
    ax.set_xlabel("train examples (log)"); ax.set_ylabel("best val accuracy")
    ax.set_title("Data efficiency: CNN vs ViT"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_data_efficiency.png", dpi=120); plt.close(fig)


def plot_robustness():
    files = sorted(EXP.glob("*_full_*/robustness.json"))
    if not files:
        return
    data = {}
    for rf in files:
        arch = rf.parent.name.split("_")[0]
        with open(rf, "r", encoding="utf-8") as f:
            data[arch] = json.load(f)
    kinds = ["shift", "rotation", "noise"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, kind in enumerate(kinds):
        for arch, res in data.items():
            if kind in res:
                mags = [p["mag"] for p in res[kind]]
                accs = [p["acc"] for p in res[kind]]
                ax[i].plot(mags, accs, marker="o", label=arch)
        ax[i].set_title(f"{kind}"); ax[i].set_xlabel("magnitude"); ax[i].set_ylabel("test acc")
        ax[i].legend(); ax[i].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_robustness.png", dpi=120); plt.close(fig)


# ----------------------- 진단 지표 (analyze.py 산출) ----------------------- #
def plot_train_val_gap(runs):
    """4) 과적합/암기: full run의 gap 곡선 + subset 크기별 best-epoch gap 표."""
    full = [h for h in runs if h.get("subset_size") is None]
    if full:
        fig, ax = plt.subplots(figsize=(7, 4))
        for h in full:
            g = train_val_gap(h)["gap_curve"]
            ax.plot(range(1, len(g) + 1), g, label=f"{h['arch']} (s{h['seed']})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title("train-val acc gap (full)"); ax.set_xlabel("epoch")
        ax.set_ylabel("train_acc - val_acc"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(EXP / "compare_train_val_gap.png", dpi=120); plt.close(fig)

    # subset 크기별 best-epoch gap (seed 평균) — ViT 암기 정량화
    agg = defaultdict(lambda: defaultdict(list))
    for h in runs:
        size = h.get("subset_size") or 54000
        agg[h["arch"]][size].append(train_val_gap(h)["gap_at_best"])
    if agg:
        print("\n[train-val gap @ best epoch]  (작을수록 일반화, 클수록 암기)")
        print(f"{'arch':<6}{'size':>8}{'gap':>10}")
        print("-" * 24)
        for arch in sorted(agg):
            for size in sorted(agg[arch]):
                print(f"{arch:<6}{size:>8}{np.mean(agg[arch][size]):>10.3f}")


def plot_gap_vs_size(runs):
    """4b) 과적합/암기의 '강한' 표현: subset 크기축 generalization gap(@best) 곡선.

    full run의 gap 곡선(compare_train_val_gap.png)은 full에선 gap이 작아 신호가 약하다.
    대신 gap@best 를 '데이터 크기'축으로 그리면 'ViT는 작은 n에서 암기(gap 큼),
    n이 커지면 CNN과 수렴'이 선명하다. CNN-ViT 격차는 중간 n(~300)에서 최대.
    오른쪽 패널: 그 최대격차 크기에서 train/val 곡선이 갈라지는 모습(=암기 signature).
    """
    agg = defaultdict(lambda: defaultdict(list))       # arch -> size -> [gap_at_best]
    first = {}                                          # (arch,size) -> history(첫 seed)
    for h in runs:
        size = h.get("subset_size") or 54000
        agg[h["arch"]][size].append(train_val_gap(h)["gap_at_best"])
        first.setdefault((h["arch"], size), h)
    if not agg:
        return

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for arch in sorted(agg):
        sizes = sorted(agg[arch])
        means = [np.mean(agg[arch][s]) for s in sizes]
        stds = [np.std(agg[arch][s]) for s in sizes]
        ax[0].errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=arch)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("train examples (log)")
    ax[0].set_ylabel("generalization gap = train_acc - val_acc (@best)")
    ax[0].set_title("Generalization gap vs training-set size")
    ax[0].grid(alpha=0.3); ax[0].legend()

    # CNN·ViT 둘 다 있는 크기 중 격차 최대인 n에서 train/val 곡선 발산 시각화
    common = sorted(set(agg.get("cnn", {})) & set(agg.get("vit", {})))
    if common:
        s_star = max(common, key=lambda s: np.mean(agg["vit"][s]) - np.mean(agg["cnn"][s]))
        for arch, c in [("cnn", "tab:blue"), ("vit", "tab:orange")]:
            h = first.get((arch, s_star))
            if not h:
                continue
            ep = range(1, len(h["train_acc"]) + 1)
            ax[1].plot(ep, h["train_acc"], color=c, ls="-", label=f"{arch} train")
            ax[1].plot(ep, h["val_acc"], color=c, ls="--", label=f"{arch} val")
        ax[1].set_title(f"Why (n={s_star}): ViT train->~1.0 while val stalls = memorization")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy")
        ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(EXP / "compare_train_val_gap_vs_size.png", dpi=130); plt.close(fig)


def plot_equivariance(analyses):
    """3) shift 내부표현 변화율 (CNN vs ViT overlay)."""
    have = [a for a in analyses if a.get("equivariance")]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in have:
        eq = a["equivariance"]
        xs = [int(k) for k in eq.keys()]
        ax.plot(xs, [eq[str(k)] if str(k) in eq else eq[k] for k in eq.keys()],
                marker="o", label=a["arch"])
    ax.set_title("Equivariance error vs shift"); ax.set_xlabel("shift (px)")
    ax.set_ylabel("||g(shift x)-g(x)|| / ||g(x)||"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_equivariance.png", dpi=120); plt.close(fig)


def plot_attention_distance(analyses):
    """1) ViT 블록별 attention 공간거리 (ViT만)."""
    have = [a for a in analyses if a.get("attention_distance")]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in have:
        ad = a["attention_distance"]
        ax.plot(range(1, len(ad) + 1), ad, marker="o", label=a["arch"])
    ax.set_title("Mean attention distance per block"); ax.set_xlabel("transformer block")
    ax.set_ylabel("distance (patches)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_attention_distance.png", dpi=120); plt.close(fig)


def plot_erf(analyses):
    """2) ERF 히트맵 CNN vs ViT 나란히."""
    have = [a for a in analyses if a.get("erf")]
    if not have:
        return
    fig, ax = plt.subplots(1, len(have), figsize=(4 * len(have), 4))
    if len(have) == 1:
        ax = [ax]
    for i, a in enumerate(have):
        im = ax[i].imshow(np.array(a["erf"]), cmap="viridis")
        ax[i].set_title(f"ERF: {a['arch']}"); ax[i].axis("off")
        fig.colorbar(im, ax=ax[i], fraction=0.046)
    fig.tight_layout(); fig.savefig(EXP / "compare_erf.png", dpi=120); plt.close(fig)


def print_table(runs):
    print(f"{'arch':<6}{'size':>8}{'seed':>6}{'ssd':>5}{'params':>10}"
          f"{'best_val_loss':>15}{'best_epoch':>12}")
    print("-" * 62)
    for h in sorted(runs, key=lambda r: (r["arch"], r.get("subset_size") or 54000, r["seed"])):
        sz = h.get("subset_size") or 54000
        print(f"{h['arch']:<6}{sz:>8}{h['seed']:>6}{h['subset_seed']:>5}"
              f"{h.get('params_total',0):>10,}{h['best_val_loss']:>15.4f}{h['best_epoch']+1:>12}")


def main():
    runs = load_histories()
    if not runs:
        print("no runs in ./experiments"); return
    plot_convergence(runs)
    plot_data_efficiency(runs)
    plot_robustness()

    analyses = load_analyses()
    plot_train_val_gap(runs)          # 4) history 기반
    plot_gap_vs_size(runs)            # 4b) 강한 표현: gap vs data size
    plot_equivariance(analyses)       # 3)
    plot_attention_distance(analyses) # 1)
    plot_erf(analyses)                # 2)

    print_table(runs)
    print("\nsaved -> experiments/compare_{convergence,data_efficiency,robustness}.png")
    print("saved -> experiments/compare_train_val_gap_vs_size.png")
    if analyses:
        print("saved -> experiments/compare_{train_val_gap,equivariance,attention_distance,erf}.png")


if __name__ == "__main__":
    main()
