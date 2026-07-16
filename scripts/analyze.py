"""저장된 체크포인트 기반 진단 지표 (재학습 없음 — 추론/autograd만).

    # 기존 방식 (full run 1개, 하위호환)
    python scripts/analyze.py --config configs/cnn.yaml --seed 42
    # 단일 run 디렉토리 (스모크 테스트용)
    python scripts/analyze.py --run-dir experiments/vit_n100_s42_ss0
    # 전체 스윕: experiments/ 아래 best.pt가 있는 모든 run
    python scripts/analyze.py --all

run마다 best.pt 로딩만으로 뽑는 지표:
  - equivariance error (both)    shift 내부표현 변화율      -> shift 붕괴 근거
  - ERF + 요약통계 (both)        중앙유닛 입력 gradient      -> 국소성 시각화/정량화
  - attention distance (vit)     블록별 attention 공간거리   -> locality 학습 여부
  - ERF radius by depth (both)   블록별 ERF RMS 반경         -> locality가 쌓이는 과정
  - frequency response (both)    블록별 고주파 비율          -> Conv=high-pass / MSA=low-pass
  - pos_embed similarity (vit)   위치임베딩 cosine 유사도    -> 2D locality를 '배웠는가'
  - penultimate PCA 2D (both)    분류 직전 표현 산점도 좌표  -> 표현공간 클러스터 구조
  - test 예측 캐시 (both)        test_preds.npy              -> confusion/오류겹침 (compare.py)

결과: experiments/<run>/analysis.json + erf.png / equivariance.png
      (+ vit: attention_distance.png, pos_embed_sim.png) + test_preds.npy
train-val gap(4)은 모델이 필요없어 compare.py 에서 history 로 집계한다.

best.pt에 cfg 전체가 내장되어 있어 --run-dir/--all 모드는 yaml이 필요 없다.
subset_size별로 지표를 모으면 'inductive bias가 데이터 크기에 따라 발현되는 과정'이 보인다.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.analysis import (attention_distance, effective_receptive_field, equivariance_error,
                          erf_radius_by_depth, frequency_response, penultimate_pca2d,
                          pos_embed_locality, pos_embed_similarity, predict_all,
                          _patch_distance_matrix)
from src.config import load_config
from src.data import build_transform
from src.models.registry import build_model
from src.utils import get_device, seed_everything


def iter_run_dirs(exp_dir: Path):
    """best.pt 가 있는 run 디렉토리들 (정렬)."""
    for d in sorted(exp_dir.iterdir()):
        if d.is_dir() and (d / "best.pt").exists():
            yield d


def load_run(run_dir: Path, device):
    """best.pt 하나로 (model, cfg, ckpt) 재구성 — cfg가 체크포인트에 내장돼 있음."""
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    cfg = ckpt["cfg"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def build_test_data(cfg: dict):
    """정규화 test loader(attention/ERF/PCA/예측) + [0,1] raw test(equivariance shift용).

    transform은 mean/std에만 의존 -> 모든 run에서 동일하므로 스윕 시 1회만 생성.
    shuffle=False 고정 -> 모든 run에서 같은 이미지 순서 (예측캐시/PCA 비교 가능).
    """
    d = cfg["data"]
    tfm = build_transform(d["mean"], d["std"])
    test_set = datasets.MNIST(d["root"], train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_set, batch_size=d["batch_size"], shuffle=False,
                             num_workers=d["num_workers"], pin_memory=True)
    raw_test = datasets.MNIST(d["root"], train=False, download=True,
                              transform=transforms.ToTensor())
    return test_loader, raw_test


def analyze_run(run_dir: Path, device, test_loader, raw_test,
                max_batches: int = 20, max_images: int = 2000) -> dict:
    """run 하나에 대해 모든 진단 지표 계산 -> analysis.json/png/test_preds.npy 저장."""
    model, cfg, ckpt = load_run(run_dir, device)
    run = run_dir.name
    arch = cfg["architecture"]["type"]
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    shifts = cfg["robustness"]["shift_px"]
    print(f"loaded {run} (best epoch {ckpt['epoch'] + 1})")

    # 식별 키: compare.py가 run 이름 파싱 없이 그룹핑할 수 있게.
    analysis = {"arch": cfg["name"], "run": run, "shifts": list(shifts),
                "subset_size": cfg["data"].get("subset_size"),
                "subset_seed": cfg["data"].get("subset_seed", 0),
                "seed": cfg.get("seed", 42)}

    # --- 3) equivariance error (both) ---
    eq = equivariance_error(model, raw_test, shifts, mean, std, device, max_images)
    analysis["equivariance"] = eq
    print("equivariance:", {k: round(v, 4) for k, v in eq.items()})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(eq.keys()), list(eq.values()), marker="o")
    ax.set_xlabel("shift (px)"); ax.set_ylabel("||g(shift x)-g(x)|| / ||g(x)||")
    ax.set_title(f"{run} equivariance error"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir / "equivariance.png", dpi=120); plt.close(fig)

    # --- 2) ERF (both) ---
    erf = effective_receptive_field(model, test_loader, device, arch, max_batches)
    analysis["erf"] = erf.tolist()
    # ERF 요약통계: peak / 배경(테두리 중앙값) / 비율 -> 'CNN 유한국소 vs ViT 전역' 정량화.
    _border = np.concatenate([erf[0], erf[-1], erf[:, 0], erf[:, -1]])
    _peak, _bg = float(erf.max()), float(np.median(_border))
    _cy, _cx = map(int, np.unravel_index(erf.argmax(), erf.shape))
    _frac7 = float(erf[max(14 - 7, 0):14 + 8, max(14 - 7, 0):14 + 8].sum() / erf.sum())
    analysis["erf_stats"] = {"peak": _peak, "bg_median": _bg,
                             "peak_over_bg": _peak / max(_bg, 1e-12),
                             "argmax": [_cy, _cx], "frac_within_7px": _frac7}
    print(f"ERF: peak={_peak:.3g} | bg={_bg:.3g} | peak/bg={_peak / max(_bg, 1e-12):.1f} "
          f"| argmax=({_cy},{_cx}) | central +/-7px energy={_frac7 * 100:.0f}%")
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(erf, cmap="viridis"); ax.set_title(f"{run} ERF"); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(run_dir / "erf.png", dpi=120); plt.close(fig)

    # --- 1) attention distance + 7) pos_embed similarity (vit only) ---
    if arch == "vit":
        grid = cfg["data"].get("img_size", 28) // cfg["architecture"]["patch_size"]
        ad = attention_distance(model, test_loader, device, grid, max_batches)
        analysis["attention_distance"] = ad
        print("attention distance (per block):", [round(v, 3) for v in ad])
        _base = float(_patch_distance_matrix(grid).mean())
        analysis["attn_uniform_baseline"] = _base
        print(f"  -> mean={sum(ad) / len(ad):.3f} patches | uniform baseline={_base:.3f} "
              f"(<=baseline면 국소학습, ~=baseline면 전역/미학습)")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(range(1, len(ad) + 1), ad)
        ax.set_xlabel("transformer block"); ax.set_ylabel("mean attn distance (patches)")
        ax.set_title(f"{run} attention distance"); ax.grid(alpha=0.3, axis="y")
        fig.tight_layout(); fig.savefig(run_dir / "attention_distance.png", dpi=120); plt.close(fig)

        sim = pos_embed_similarity(model)
        loc = pos_embed_locality(sim, grid)
        analysis["pos_embed_sim"] = sim.tolist()
        analysis["pos_embed_locality"] = loc
        print(f"pos_embed locality: {loc:.3f} (+1=2D locality 학습, 0=무구조/init)")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(f"{run}\npos_embed cosine sim (locality={loc:.2f})")
        ax.set_xlabel("patch idx"); ax.set_ylabel("patch idx")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout(); fig.savefig(run_dir / "pos_embed_sim.png", dpi=120); plt.close(fig)
    else:
        analysis["attention_distance"] = None

    # --- 5) ERF 반경 vs 깊이 (both) ---
    erf_depth = erf_radius_by_depth(model, test_loader, device, arch, max_batches)
    analysis["erf_depth"] = erf_depth
    print("ERF radius by depth:", [round(v, 2) for v in erf_depth])

    # --- 6) 주파수 응답 (both, Conv=high-pass / MSA=low-pass) ---
    freq = frequency_response(model, test_loader, device, arch, max_batches)
    analysis["freq_response"] = freq
    print("high-freq fraction by depth:", [round(v, 3) for v in freq])

    # --- 8) penultimate PCA 2D (both) ---
    coords, labels, evr = penultimate_pca2d(model, test_loader, device, max_images)
    analysis["pca2d"] = {"x": np.round(coords[:, 0], 4).tolist(),
                         "y": np.round(coords[:, 1], 4).tolist(),
                         "label": labels.astype(int).tolist(),
                         "explained_var": evr.tolist()}
    print(f"penultimate PCA: {coords.shape[0]} pts, explained var = "
          f"{evr[0] * 100:.0f}% + {evr[1] * 100:.0f}%")

    # --- 9) test 예측 캐시 (both) ---
    preds = predict_all(model, test_loader, device)
    np.save(run_dir / "test_preds.npy", preds)
    targets = raw_test.targets.numpy()
    test_acc = float((preds == targets).mean())
    analysis["test_acc"] = test_acc
    print(f"test acc: {test_acc:.4f} (preds cached -> test_preds.npy)")

    with open(run_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"saved -> {run_dir}/analysis.json (+ png)\n")
    return analysis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="(하위호환) full run의 config yaml")
    p.add_argument("--run-dir", help="단일 run 디렉토리 경로 (예: experiments/vit_n100_s42_ss0)")
    p.add_argument("--all", action="store_true", help="experiments/ 아래 모든 run 스윕")
    p.add_argument("--exp-dir", default="./experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=20, help="attention/ERF 표본 배치 수")
    p.add_argument("--max-images", type=int, default=2000, help="equivariance/PCA 표본 이미지 수")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device()

    if args.all:
        run_dirs = list(iter_run_dirs(Path(args.exp_dir)))
    elif args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.config:
        cfg = load_config(args.config)
        cfg["seed"] = args.seed
        run = f"{cfg['name']}_full_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"
        run_dirs = [Path(cfg["output"]["dir"]) / run]
    else:
        p.error("--config / --run-dir / --all 중 하나가 필요합니다")

    # test 데이터는 전 run 공통 -> 첫 run의 cfg로 1회만 구성
    _, cfg0, _ = load_run(run_dirs[0], "cpu")
    test_loader, raw_test = build_test_data(cfg0)

    for i, rd in enumerate(run_dirs):
        if len(run_dirs) > 1:
            print(f"[{i + 1}/{len(run_dirs)}] {rd.name}")
        analyze_run(rd, device, test_loader, raw_test, args.max_batches, args.max_images)


if __name__ == "__main__":
    main()
