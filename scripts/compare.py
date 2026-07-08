"""모든 run 을 모아 비교 곡선/표를 생성.

    python scripts/compare.py

experiments/ 아래를 스캔해서:
- 수렴 비교: full run의 val_loss/val_acc overlay (CNN vs ViT)
- 데이터 효율성: subset 크기별 best_val_acc 곡선 (seed 평균 ± std, error band)
- robustness: shift/rotation/noise 곡선 (있으면)
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

EXP = Path("./experiments")


def load_histories():
    runs = []
    for hf in sorted(EXP.glob("*/history.json")):
        with open(hf, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


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
    print_table(runs)
    print("\nsaved -> experiments/compare_{convergence,data_efficiency,robustness}.png")


if __name__ == "__main__":
    main()
