"""진단 지표 모듈 — 저장된 체크포인트/history만으로 뽑는 4개 분석 지표.

전부 forward / autograd 만 사용한다 (재학습 없음). best.pt 로딩이면 충분.

  1) attention_distance(...)        ViT 블록별 attention 공간거리   -> locality 학습 여부
  2) effective_receptive_field(...) 중앙 유닛의 입력 gradient 히트맵 -> 국소성 시각화
  3) equivariance_error(...)        shift 시 내부표현 변화율         -> shift 붕괴의 내부근거
  4) train_val_gap(...)             history의 train-val 곡선          -> 과적합/암기 정량화

지표별로 어느 결과를 설명하는지:
  - 1,2 : 데이터효율 격차(=CNN의 locality prior)의 직접 근거
  - 3   : shift 정확도 붕괴(97.6->16.8)의 내부표현 근거
  - 4   : 소량 데이터에서 ViT가 지는 이유(=암기) 정량화
"""
from __future__ import annotations
import numpy as np
import torch


# ------------------------------------------------------------------ #
# 1) Mean attention distance (ViT 전용)
# ------------------------------------------------------------------ #
def _patch_distance_matrix(grid: int) -> torch.Tensor:
    """grid×grid 패치 중심 간 유클리드 거리행렬 [P, P] (P=grid^2, 단위=패치)."""
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    coords = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=1).float()   # [P,2]
    return torch.cdist(coords, coords)                                       # [P,P]


@torch.no_grad()
def attention_distance(model, loader, device, grid: int, max_batches: int = 20):
    """ViT 블록별 mean attention distance (패치 단위).

    CLS 토큰은 공간위치가 없어 제외하고, 패치-패치 attention만 행 기준 재정규화한 뒤
    패치거리로 가중평균한다. 낮을수록 국소(=CNN다움), 높을수록 전역.
    (Block이 average_attn_weights=True라 head 평균 기준. 반환: list[float] 길이 depth.)
    """
    model.eval()
    D = _patch_distance_matrix(grid).to(device)          # [P,P]
    sums, total = None, 0
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        _, attns = model(x, return_attn=True)            # list of [B, P+1, P+1]
        if sums is None:
            sums = [0.0] * len(attns)
        for li, A in enumerate(attns):
            Ap = A[:, 1:, 1:]                            # 패치-패치 [B,P,P]
            Ap = Ap / Ap.sum(-1, keepdim=True).clamp_min(1e-9)
            d = (Ap * D.unsqueeze(0)).sum(-1)            # [B,P] query별 평균거리
            sums[li] += d.mean().item() * x.size(0)
        total += x.size(0)
    return [s / total for s in sums]


# ------------------------------------------------------------------ #
# 2) Effective Receptive Field (CNN/ViT 공통)
# ------------------------------------------------------------------ #
def effective_receptive_field(model, loader, device, arch: str, max_batches: int = 20):
    """중앙 유닛의 입력에 대한 |gradient| 평균 히트맵 [H,W].

    중앙 '유닛'은:
      - simple_cnn : features 출력 [B,C,h,w] 의 중앙 셀 (h//2, w//2), 채널 합
      - vit        : norm 출력 [B,1+P,E] 의 패치토큰(g×g)을 보고 중앙 패치, 채널 합
    두 경우 모두 입력 28×28 로 역전파 -> '어느 입력픽셀이 중앙표현에 영향 주나'.
    CNN은 조밀(국소), ViT는 광역으로 퍼짐이 예상.
    """
    model.eval()
    cap = {}
    target = model.features if arch == "simple_cnn" else model.norm
    h = target.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o))

    erf, n = None, 0
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device).requires_grad_(True)
        model.zero_grad(set_to_none=True)
        model(x)
        f = cap["f"]
        if arch == "simple_cnn":
            s = f.shape[-1]
            scalar = f[:, :, s // 2, s // 2].sum()
        else:
            B, T, E = f.shape
            g = int(round((T - 1) ** 0.5))
            patches = f[:, 1:, :].reshape(B, g, g, E)
            scalar = patches[:, g // 2, g // 2, :].sum()
        scalar.backward()
        grad = x.grad.detach().abs().mean(dim=(0, 1))    # [H,W] (배치·채널 평균)
        erf = grad if erf is None else erf + grad
        n += 1
    h.remove()
    return (erf / n).cpu().numpy()


# ------------------------------------------------------------------ #
# 3) Equivariance (shift-invariance) error (CNN/ViT 공통)
# ------------------------------------------------------------------ #
@torch.no_grad()
def equivariance_error(model, raw_test, shifts, mean, std, device, max_images: int = 2000):
    """shift 에 따른 내부표현 변화율 ||g(shift x) - g(x)|| / ||g(x)||.

    g(x) = 분류 직전 표현 = head 의 입력 (forward hook 으로 캡처).
      - simple_cnn : GAP 직후 [B,160]
      - vit        : 최종 LN 후 CLS [B,64]
    CNN 은 shift 에 거의 불변(≈0), ViT 는 급증 -> shift 정확도 붕괴의 내부근거.
    반환: dict {shift(int): error(float)}.
    """
    from src.robustness import make_perturb
    model.eval()
    cap = {}
    h = model.head.register_forward_hook(lambda m, i, o: cap.__setitem__("g", i[0].detach()))

    imgs = [raw_test[i][0] for i in range(min(max_images, len(raw_test)))]   # [1,28,28], [0,1]

    def represent(fn):
        outs = []
        for i in range(0, len(imgs), 256):
            batch = torch.stack([fn(im) for im in imgs[i:i + 256]]).to(device)
            model(batch)
            outs.append(cap["g"].clone())
        return torch.cat(outs, 0)

    g0 = represent(make_perturb("shift", 0, mean, std))     # shift 0 = 정규화만
    denom = g0.norm(dim=1).clamp_min(1e-9)
    result = {}
    for s in shifts:
        gs = represent(make_perturb("shift", s, mean, std))
        result[int(s)] = ((gs - g0).norm(dim=1) / denom).mean().item()
    h.remove()
    return result


# ------------------------------------------------------------------ #
# 4) Train-val gap (history.json 후처리 — 모델/GPU 불필요)
# ------------------------------------------------------------------ #
def train_val_gap(history: dict) -> dict:
    """history 하나에서 과적합/암기 지표.

    gap = train_acc - val_acc. gap 이 클수록 '외웠다'.
    반환: {gap_curve, gap_at_best, max_gap}.
    """
    tr = np.asarray(history["train_acc"], dtype=float)
    va = np.asarray(history["val_acc"], dtype=float)
    be = int(history.get("best_epoch", len(tr) - 1))
    be = max(0, min(be, len(tr) - 1))
    return {
        "gap_curve": (tr - va).tolist(),
        "gap_at_best": float(tr[be] - va[be]),
        "max_gap": float(np.max(tr - va)),
    }


# ------------------------------------------------------------------ #
# 5) ERF 반경 vs 깊이 (CNN/ViT 공통) — locality가 '쌓이는 과정'
# ------------------------------------------------------------------ #
def _block_targets(model, arch: str):
    """블록별 출력 모듈 리스트 (얕은->깊은)."""
    if arch == "simple_cnn":
        seq = model.features                 # Conv-ReLU-Pool ×2 + Conv-ReLU
        return [seq[2], seq[5], seq[7]]      # pool1[14²], pool2[7²], relu3[7²]
    return list(model.blocks)                # transformer 블록 ×depth


def _erf_rms_radius(erf: np.ndarray) -> float:
    """ERF 히트맵의 RMS 반경(px). 작을수록 국소, 클수록 넓게 봄."""
    h, w = erf.shape
    ys, xs = np.mgrid[0:h, 0:w]
    m = erf / (erf.sum() + 1e-12)
    cy, cx = (m * ys).sum(), (m * xs).sum()
    return float(np.sqrt((m * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum()))


def erf_radius_by_depth(model, loader, device, arch, max_batches: int = 20):
    """블록마다 중앙유닛 ERF를 구해 RMS 반경을 잰다 -> 깊이별 리스트.

    CNN: 작게 시작해 층마다 커짐(국소성이 점진 확장).
    ViT: 1블록부터 이미 큼(patch-embed + 전역 attention) -> 거의 평평.
    """
    model.eval()
    targets = _block_targets(model, arch)
    radii = [[] for _ in targets]
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = x.to(device)
        for ti, tmod in enumerate(targets):
            cap = {}
            hk = tmod.register_forward_hook(lambda m, i, o: cap.__setitem__("f", o))
            xx = x.clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            model(xx)
            hk.remove()
            f = cap["f"]
            if arch == "simple_cnn":
                s = f.shape[-1]
                scalar = f[:, :, s // 2, s // 2].sum()
            else:
                B, T, E = f.shape
                g = int(round((T - 1) ** 0.5))
                scalar = f[:, 1:, :].reshape(B, g, g, E)[:, g // 2, g // 2, :].sum()
            scalar.backward()
            grad = xx.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()   # [H,W]
            radii[ti].append(_erf_rms_radius(grad))
    return [float(np.mean(r)) for r in radii]


# ------------------------------------------------------------------ #
# 6) 주파수 응답 (CNN/ViT 공통) — Conv=high-pass, MSA=low-pass (Park&Kim 2022)
# ------------------------------------------------------------------ #
@torch.no_grad()
def _highfreq_fraction(fmap: torch.Tensor) -> float:
    """feature map [B,C,h,w]의 2D FFT에서 고주파(바깥 절반) 진폭 비율."""
    amp = torch.fft.fftshift(torch.fft.fft2(fmap.float(), norm="ortho"),
                             dim=(-2, -1)).abs().mean(dim=(0, 1)).cpu()    # [h,w]
    h, w = amp.shape
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    r = torch.sqrt((ys - h // 2).float() ** 2 + (xs - w // 2).float() ** 2)
    hi = amp[r > 0.5 * r.max()].sum()
    return float(hi / (amp.sum() + 1e-8))


@torch.no_grad()
def frequency_response(model, loader, device, arch, max_batches: int = 20):
    """블록별 feature map의 고주파 비율 -> 깊이별 리스트.

    Conv는 엣지(고주파)를 살리고(high-pass), attention은 뭉갠다(low-pass).
    CNN은 높게/유지, ViT는 낮게 -> 정반대. noise 결과(ViT가 잡음에 약간 강함)의 근거.
    작은 feature map(7²·4²)이라 해상도는 거칠다(대표값).
    """
    model.eval()
    targets = _block_targets(model, arch)
    caps = [{} for _ in targets]
    hooks = [t.register_forward_hook(
        (lambda c: (lambda m, i, o: c.__setitem__("f", o)))(caps[k]))
        for k, t in enumerate(targets)]
    fracs = [[] for _ in targets]
    for bi, (x, _) in enumerate(loader):
        if bi >= max_batches:
            break
        model(x.to(device))
        for ti, c in enumerate(caps):
            f = c["f"]
            if arch != "simple_cnn":
                B, T, E = f.shape
                g = int(round((T - 1) ** 0.5))
                f = f[:, 1:, :].reshape(B, g, g, E).permute(0, 3, 1, 2)   # [B,E,g,g]
            fracs[ti].append(_highfreq_fraction(f))
    for hk in hooks:
        hk.remove()
    return [float(np.mean(v)) for v in fracs]
