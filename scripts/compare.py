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
    """analyze.py 가 남긴 진단 결과들 (모든 run의 analysis.json).

    analysis.json에 subset_size/subset_seed 식별 키가 있어 run 이름 파싱이 불필요.
    (구버전 full 전용 analysis.json은 키가 없음 -> .get()으로 full 취급.)
    """
    out = []
    for af in sorted(EXP.glob("*/analysis.json")):
        with open(af, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def _size_of(a: dict) -> int:
    """analysis/history dict -> 데이터 크기 (full=54000)."""
    return a.get("subset_size") or 54000


def _full_only(analyses):
    return [a for a in analyses if a.get("subset_size") is None]


SIZE_LABEL = {54000: "full(54k)"}


def _size_lbl(s: int) -> str:
    return SIZE_LABEL.get(s, f"n={s}")


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


def plot_erf_depth(analyses):
    """5) ERF RMS 반경 vs 깊이 (CNN vs ViT overlay). CNN 성장 / ViT 평평."""
    have = [a for a in analyses if a.get("erf_depth")]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in have:
        y = a["erf_depth"]
        ax.plot(np.linspace(0, 1, len(y)), y, marker="o", label=a["arch"])
    ax.set_xlabel("relative depth (0=input -> 1=output)")
    ax.set_ylabel("ERF RMS radius (px)")
    ax.set_title("Effective receptive field vs depth")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_erf_depth.png", dpi=130); plt.close(fig)


def plot_freq_response(analyses):
    """6) 고주파 진폭 비율 vs 깊이. Conv=high-pass(높음) / MSA=low-pass(낮음)."""
    have = [a for a in analyses if a.get("freq_response")]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for a in have:
        y = a["freq_response"]
        ax.plot(np.linspace(0, 1, len(y)), y, marker="o", label=a["arch"])
    ax.set_xlabel("relative depth (0=input -> 1=output)")
    ax.set_ylabel("high-freq amplitude fraction")
    ax.set_title("Frequency response: Conv=high-pass vs MSA=low-pass")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "compare_freq_response.png", dpi=130); plt.close(fig)


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


# ----------------- 신규 진단 (전체 run 스윕 산출 기반) ----------------- #
def plot_erf_grid(analyses):
    """C1) ERF 히트맵 그리드 (arch 행 × 데이터크기 열).

    'CNN은 어느 n에서도 국소 / ViT는 데이터가 커져야(혹은 끝까지) 전역·산만'
    을 한 장으로. 각 맵은 sum=1 정규화 후 seed 평균, 행별 공유 컬러스케일.
    """
    agg = defaultdict(lambda: defaultdict(list))     # arch -> size -> [erf map]
    for a in analyses:
        if a.get("erf"):
            m = np.array(a["erf"], dtype=float)
            agg[a["arch"]][_size_of(a)].append(m / m.sum())
    if not agg:
        return
    archs = sorted(agg)
    sizes = sorted(set(s for per in agg.values() for s in per))
    fig, axes = plt.subplots(len(archs), len(sizes),
                             figsize=(2.1 * len(sizes), 2.3 * len(archs)))
    axes = np.atleast_2d(axes)
    for r, arch in enumerate(archs):
        row_maps = {s: np.mean(agg[arch][s], axis=0) for s in sizes if s in agg[arch]}
        vmax = max(m.max() for m in row_maps.values())
        for c, s in enumerate(sizes):
            ax = axes[r, c]
            if s in row_maps:
                ax.imshow(row_maps[s], cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(_size_lbl(s), fontsize=10)
            if c == 0:
                ax.set_ylabel(arch, fontsize=12)
    fig.suptitle("ERF vs training-set size (per-map sum-normalized, row-shared scale)")
    fig.tight_layout()
    fig.savefig(EXP / "compare_erf_grid.png", dpi=130); plt.close(fig)


def plot_bias_vs_n(analyses):
    """C1) 스칼라 inductive-bias 지표 4종 vs 데이터 크기 (seed 평균 ± std).

    '바이어스는 내장(CNN) vs 학습해야 함(ViT)'이 곡선의 평평함/이동으로 보인다.
    """
    def agg_metric(key_fn, arch_filter=None):
        agg = defaultdict(lambda: defaultdict(list))
        for a in analyses:
            if arch_filter and a["arch"] != arch_filter:
                continue
            v = key_fn(a)
            if v is not None:
                agg[a["arch"]][_size_of(a)].append(v)
        return agg

    def draw(ax, agg, ylabel, title):
        for arch in sorted(agg):
            sizes = sorted(agg[arch])
            means = [np.mean(agg[arch][s]) for s in sizes]
            stds = [np.std(agg[arch][s]) for s in sizes]
            ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label=arch)
        ax.set_xscale("log"); ax.set_xlabel("train examples (log)")
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=0.3); ax.legend()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draw(axes[0, 0],
         agg_metric(lambda a: (a.get("erf_stats") or {}).get("frac_within_7px")),
         "ERF energy within ±7px", "ERF locality vs data size")
    draw(axes[0, 1],
         agg_metric(lambda a: np.mean(a["attention_distance"])
                    if a.get("attention_distance") else None),
         "mean attention distance (patches)", "Attention distance vs data size (ViT)")
    base = next((a.get("attn_uniform_baseline") for a in analyses
                 if a.get("attn_uniform_baseline")), None)
    if base:
        axes[0, 1].axhline(base, color="k", ls="--", lw=1, label="uniform baseline")
        axes[0, 1].legend()
    draw(axes[1, 0],
         agg_metric(lambda a: (a.get("equivariance") or {}).get("4")),
         "||g(shift x)-g(x)|| / ||g(x)|| @ 4px", "Shift-equivariance error vs data size")
    draw(axes[1, 1],
         agg_metric(lambda a: a["freq_response"][-1] if a.get("freq_response") else None),
         "high-freq fraction (last block)", "Frequency response vs data size")
    fig.suptitle("Inductive bias metrics vs training-set size (mean ± std over subset seeds)")
    fig.tight_layout()
    fig.savefig(EXP / "compare_bias_vs_n.png", dpi=130); plt.close(fig)


def plot_pos_embed(analyses):
    """C2) ViT pos_embed cosine 유사도: 크기별 맵 + locality 점수 곡선.

    '2D locality는 ViT에 내장돼 있지 않고 데이터로부터 학습된다'의 직접 증거.
    """
    sims = defaultdict(list)                          # size -> [16x16 sim]
    locs = defaultdict(list)                          # size -> [locality]
    for a in analyses:
        if a["arch"] != "vit" or not a.get("pos_embed_sim"):
            continue
        sims[_size_of(a)].append(np.array(a["pos_embed_sim"]))
        locs[_size_of(a)].append(a.get("pos_embed_locality", np.nan))
    if not sims:
        return
    sizes = sorted(sims)
    fig, axes = plt.subplots(1, len(sizes) + 1, figsize=(2.4 * (len(sizes) + 1), 3.0))
    for c, s in enumerate(sizes):
        m = np.mean(sims[s], axis=0)
        axes[c].imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
        axes[c].set_title(f"{_size_lbl(s)}\nloc={np.mean(locs[s]):.2f}", fontsize=9)
        axes[c].set_xticks([]); axes[c].set_yticks([])
    ax = axes[-1]
    means = [np.mean(locs[s]) for s in sizes]
    stds = [np.std(locs[s]) for s in sizes]
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, color="tab:orange")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("train examples")
    ax.set_ylabel("locality score"); ax.set_title("pos_embed locality vs n", fontsize=9)
    ax.grid(alpha=0.3)
    fig.suptitle("ViT positional-embedding similarity: does it LEARN 2D locality?")
    fig.tight_layout()
    fig.savefig(EXP / "compare_pos_embed.png", dpi=130); plt.close(fig)


def plot_pca_grid(analyses):
    """C5) 분류 직전 표현의 2D PCA 산점도 (arch 행 × 크기 열, subset_seed=0).

    작은 n에서 'CNN은 이미 클러스터 / ViT는 뭉개짐' -> 표현공간에서 본 데이터효율.
    """
    sel = {}                                          # (arch,size) -> pca2d
    for a in analyses:
        if not a.get("pca2d"):
            continue
        key = (a["arch"], _size_of(a))
        if key not in sel or a.get("subset_seed", 0) < sel[key][0]:
            sel[key] = (a.get("subset_seed", 0), a["pca2d"])
    if not sel:
        return
    archs = sorted(set(k[0] for k in sel))
    sizes = sorted(set(k[1] for k in sel))
    fig, axes = plt.subplots(len(archs), len(sizes),
                             figsize=(2.4 * len(sizes), 2.5 * len(archs)))
    axes = np.atleast_2d(axes)
    for r, arch in enumerate(archs):
        for c, s in enumerate(sizes):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if (arch, s) in sel:
                p = sel[(arch, s)][1]
                ax.scatter(p["x"], p["y"], c=p["label"], cmap="tab10", s=2, alpha=0.6)
            if r == 0:
                ax.set_title(_size_lbl(s), fontsize=10)
            if c == 0:
                ax.set_ylabel(arch, fontsize=12)
    fig.suptitle("Penultimate representation, 2D PCA (color = digit class)")
    fig.tight_layout()
    fig.savefig(EXP / "compare_pca_grid.png", dpi=130); plt.close(fig)


# ----------------- 신규 진단 (예측 캐시 test_preds.npy 기반) ----------------- #
def _load_pred_runs():
    """test_preds.npy 가 있는 run들: [(arch, size, ssd, preds)]. + 정답 labels."""
    entries = []
    for pf in sorted(EXP.glob("*/test_preds.npy")):
        af = pf.parent / "analysis.json"
        if not af.exists():
            continue
        with open(af, "r", encoding="utf-8") as f:
            a = json.load(f)
        entries.append((a["arch"], _size_of(a), a.get("subset_seed", 0), np.load(pf)))
    if not entries:
        return [], None
    from torchvision import datasets
    targets = datasets.MNIST("./data", train=False, download=True).targets.numpy()
    return entries, targets


def plot_confusion(entries, targets):
    """C4a) full run confusion matrix (행 정규화, 대각 제외 -> 오류 패턴 강조)."""
    full = {arch: preds for arch, size, ssd, preds in entries if size == 54000}
    if not full:
        return
    archs = sorted(full)
    fig, axes = plt.subplots(1, len(archs), figsize=(5.4 * len(archs), 4.6))
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        preds = full[arch]
        cm = np.zeros((10, 10))
        for t, p in zip(targets, preds):
            cm[t, p] += 1
        cm = cm / cm.sum(axis=1, keepdims=True)
        err = cm.copy(); np.fill_diagonal(err, np.nan)      # 오류만 색으로
        cmap = plt.get_cmap("Reds").copy(); cmap.set_bad("#f0f0f0")
        im = ax.imshow(err, cmap=cmap, vmin=0)
        for i in range(10):
            ax.text(i, i, f"{cm[i, i] * 100:.0f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{arch} (full) — diag=acc%, color=error rate")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(EXP / "compare_confusion.png", dpi=130); plt.close(fig)


def plot_per_class_acc(entries, targets):
    """C4b) 클래스별 정확도 (class × 데이터크기 히트맵) + CNN-ViT 차이."""
    agg = defaultdict(lambda: defaultdict(list))      # arch -> size -> [per-class acc(10)]
    for arch, size, ssd, preds in entries:
        acc = np.array([(preds[targets == c] == c).mean() for c in range(10)])
        agg[arch][size].append(acc)
    if not agg:
        return
    archs = sorted(agg)
    sizes = sorted(set(s for per in agg.values() for s in per))
    mats = {arch: np.array([[np.mean(agg[arch][s], axis=0)[c] if s in agg[arch] else np.nan
                             for s in sizes] for c in range(10)]) for arch in archs}
    ncol = len(archs) + (1 if len(archs) == 2 else 0)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 4.4))
    for i, arch in enumerate(archs):
        im = axes[i].imshow(mats[arch], cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[i].set_title(f"{arch}: per-class acc")
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    if len(archs) == 2:
        d = mats[archs[0]] - mats[archs[1]]
        lim = np.nanmax(np.abs(d))
        im = axes[-1].imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        axes[-1].set_title(f"{archs[0]} − {archs[1]} (blue = {archs[1]} wins)")
        fig.colorbar(im, ax=axes[-1], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([_size_lbl(s) for s in sizes], rotation=45, fontsize=8)
        ax.set_yticks(range(10)); ax.set_ylabel("digit class")
    fig.tight_layout()
    fig.savefig(EXP / "compare_per_class_acc.png", dpi=130); plt.close(fig)


def plot_agreement(entries, targets):
    """C4c) CNN·ViT 오류 겹침: n별 {둘다정답 / CNN만 / ViT만 / 둘다오답} 비율.

    두 모델이 '같은' 문제를 틀리는지 '다른' 문제를 틀리는지 -> 서로 다른
    inductive bias가 서로 다른 실수를 만든다는 직접 증거 (앙상블 여지 포함).
    """
    by_key = defaultdict(dict)                        # (size,ssd) -> arch -> preds
    for arch, size, ssd, preds in entries:
        by_key[(size, ssd)][arch] = preds
    agg = defaultdict(list)                           # size -> [(bc, co, vo, bw)]
    for (size, ssd), d in by_key.items():
        if "cnn" not in d or "vit" not in d:
            continue
        c_ok = d["cnn"] == targets
        v_ok = d["vit"] == targets
        agg[size].append(((c_ok & v_ok).mean(), (c_ok & ~v_ok).mean(),
                          (~c_ok & v_ok).mean(), (~c_ok & ~v_ok).mean()))
    if not agg:
        return
    sizes = sorted(agg)
    fracs = np.array([np.mean(agg[s], axis=0) for s in sizes])    # [S,4]
    labels = ["both correct", "CNN only", "ViT only", "both wrong"]
    colors = ["#8dc63f", "tab:blue", "tab:orange", "#666666"]
    x = np.arange(len(sizes))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(sizes))
    for k in range(4):
        ax.bar(x, fracs[:, k], bottom=bottom, label=labels[k], color=colors[k])
        bottom += fracs[:, k]
    for i in range(len(sizes)):                       # 오류 겹침 수치 주석
        ax.text(x[i], min(fracs[i, 0], 0.98) - 0.02,
                f"disagree\n{(fracs[i, 1] + fracs[i, 2]) * 100:.1f}%",
                ha="center", va="top", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([_size_lbl(s) for s in sizes], rotation=30)
    ax.set_ylim(min(0.4, fracs[:, 0].min() - 0.05), 1.0)
    ax.set_ylabel("fraction of test set")
    ax.set_title("Error overlap: do CNN and ViT make the SAME mistakes?")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(EXP / "compare_agreement.png", dpi=130); plt.close(fig)


# ----------------- 신규 진단 (analyze_extra.py 산출 기반) ----------------- #
def plot_translation2d():
    """C3) 2D shift 히트맵 그리드 (arch 행 × 크기 열, seed 평균).

    기존 1D 대각 shift 곡선의 2D 확장 — 'CNN 고원 vs ViT 붕괴'.
    """
    agg = defaultdict(lambda: defaultdict(list))      # arch -> size -> [9x9 acc]
    rng = 4
    for tf_ in sorted(EXP.glob("*/translation2d.json")):
        with open(tf_, "r", encoding="utf-8") as f:
            t = json.load(f)
        rng = t.get("range_px", 4)
        agg[t["arch"]][t.get("subset_size") or 54000].append(np.array(t["acc"]))
    if not agg:
        return
    archs = sorted(agg)
    sizes = sorted(set(s for per in agg.values() for s in per))
    fig, axes = plt.subplots(len(archs), len(sizes),
                             figsize=(2.2 * len(sizes), 2.4 * len(archs)))
    axes = np.atleast_2d(axes)
    im = None
    for r, arch in enumerate(archs):
        for c, s in enumerate(sizes):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if s in agg[arch]:
                m = np.mean(agg[arch][s], axis=0)
                im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1,
                               extent=[-rng - .5, rng + .5, rng + .5, -rng - .5])
            if r == 0:
                ax.set_title(_size_lbl(s), fontsize=10)
            if c == 0:
                ax.set_ylabel(arch, fontsize=12)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02)
    fig.suptitle(f"Accuracy under 2D shift (dx,dy) ∈ [-{rng},{rng}]²  — plateau vs collapse")
    fig.savefig(EXP / "compare_translation2d.png", dpi=130); plt.close(fig)


def plot_fourier():
    """C6) Fourier 민감도 히트맵 나란히 (full run, 공유 스케일)."""
    data = {}
    for ff in sorted(EXP.glob("*_full_*/fourier.json")):
        with open(ff, "r", encoding="utf-8") as f:
            d = json.load(f)
        data[d["arch"]] = d
    if not data:
        return
    archs = sorted(data)
    vmax = max(np.nanmax(np.array(d["err"])) for d in data.values())
    fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 4.4))
    if len(archs) == 1:
        axes = [axes]
    for ax, arch in zip(axes, archs):
        d = data[arch]
        disp = np.fft.fftshift(np.array(d["err"]))
        im = ax.imshow(disp, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(f"{arch} (eps={d['eps']}, clean err={d['clean_err']:.3f})")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("freq (center=DC, edge=high)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fourier sensitivity: error rate under single-frequency perturbation")
    fig.tight_layout()
    fig.savefig(EXP / "compare_fourier.png", dpi=130); plt.close(fig)


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
    full = _full_only(analyses)       # 기존 overlay 플롯은 full run 기준 유지
    plot_train_val_gap(runs)          # 4) history 기반
    plot_gap_vs_size(runs)            # 4b) 강한 표현: gap vs data size
    plot_equivariance(full)           # 3)
    plot_attention_distance(full)     # 1)
    plot_erf(full)                    # 2)
    plot_erf_depth(full)              # 5) ERF vs depth
    plot_freq_response(full)          # 6) 주파수 응답

    # --- 신규: 전체 run 스윕 기반 (analyze.py --all 산출) ---
    plot_erf_grid(analyses)           # C1) ERF vs data size 그리드
    plot_bias_vs_n(analyses)          # C1) 스칼라 지표 4종 vs n
    plot_pos_embed(analyses)          # C2) pos_embed 유사도 + locality
    plot_pca_grid(analyses)           # C5) 표현공간 PCA 그리드

    # --- 신규: 예측 캐시 기반 ---
    entries, targets = _load_pred_runs()
    if entries:
        plot_confusion(entries, targets)     # C4a
        plot_per_class_acc(entries, targets) # C4b
        plot_agreement(entries, targets)     # C4c

    # --- 신규: analyze_extra.py 산출 기반 ---
    plot_translation2d()              # C3
    plot_fourier()                    # C6

    print_table(runs)
    print("\nsaved -> experiments/compare_{convergence,data_efficiency,robustness}.png")
    print("saved -> experiments/compare_train_val_gap_vs_size.png")
    if analyses:
        print("saved -> experiments/compare_{train_val_gap,equivariance,attention_distance,erf}.png")
        print("saved -> experiments/compare_{erf_grid,bias_vs_n,pos_embed,pca_grid}.png")
    if entries:
        print("saved -> experiments/compare_{confusion,per_class_acc,agreement}.png")
    print("saved -> experiments/compare_{translation2d,fourier}.png (analyze_extra 산출이 있으면)")


if __name__ == "__main__":
    main()
