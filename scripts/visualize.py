"""해석 시각화 (garnish): CNN Grad-CAM vs ViT attention rollout.

    python scripts/visualize.py --config configs/cnn.yaml --seed 42 --n 8
    python scripts/visualize.py --config configs/vit.yaml --seed 42 --n 8

"모델이 어디를 보고 판단하나"의 정성 대비. 결론을 떠받치는 축은 아니지만
발표/보고서 임팩트가 큼. full 체크포인트 사용.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.config import load_config
from src.models.registry import build_model
from src.utils import get_device, seed_everything


def gradcam_cnn(model, x):
    """마지막 conv feature에 대한 Grad-CAM."""
    feats = {}
    last_conv = [m for m in model.features if isinstance(m, torch.nn.Conv2d)][-1]
    h1 = last_conv.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    grads = {}
    h2 = last_conv.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))

    logits = model(x)
    cls = logits.argmax(1)
    model.zero_grad()
    logits[range(len(x)), cls].sum().backward()

    a, g = feats["a"], grads["g"]            # B,C,h,w
    w = g.mean(dim=(2, 3), keepdim=True)     # channel weights
    cam = torch.relu((w * a).sum(1))         # B,h,w
    h1.remove(); h2.remove()
    cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return cam.detach().cpu().numpy(), cls.cpu().numpy()


def attn_rollout_vit(model, x):
    """attention rollout: 층별 attention을 곱해 cls->patch 기여를 근사."""
    with torch.no_grad():
        logits, attns = model(x, return_attn=True)   # each B,T,T (T=17)
    cls = logits.argmax(1)
    B, T, _ = attns[0].shape
    result = torch.eye(T, device=x.device).expand(B, T, T).clone()
    for a in attns:
        a = a + torch.eye(T, device=x.device)        # residual
        a = a / a.sum(-1, keepdim=True)
        result = torch.bmm(a, result)
    cls_to_patch = result[:, 0, 1:]                  # B,16
    side = int(cls_to_patch.shape[1] ** 0.5)
    heat = cls_to_patch.reshape(B, side, side)
    heat = heat / (heat.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return heat.cpu().numpy(), cls.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    seed_everything(cfg["seed"])
    device = get_device()

    run = f"{cfg['name']}_full_s{cfg['seed']}_ss{cfg['data']['subset_seed']}"
    ckpt = torch.load(Path(cfg["output"]["dir"]) / run / "best.pt", map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((cfg["data"]["mean"],), (cfg["data"]["std"],))])
    test = datasets.MNIST(cfg["data"]["root"], train=False, download=True, transform=tfm)
    xs = torch.stack([test[i][0] for i in range(args.n)]).to(device)
    imgs = xs.cpu().numpy()

    if cfg["architecture"]["type"] == "simple_cnn":
        heat, cls = gradcam_cnn(model, xs)
        title = "Grad-CAM (CNN)"
    else:
        heat, cls = attn_rollout_vit(model, xs)
        title = "Attention rollout (ViT)"

    fig, ax = plt.subplots(2, args.n, figsize=(1.6 * args.n, 3.6))
    for i in range(args.n):
        ax[0, i].imshow(imgs[i, 0], cmap="gray"); ax[0, i].axis("off")
        ax[0, i].set_title(f"pred {cls[i]}", fontsize=9)
        ax[1, i].imshow(imgs[i, 0], cmap="gray")
        hm = np.kron(heat[i], np.ones((28 // heat[i].shape[0], 28 // heat[i].shape[1])))
        ax[1, i].imshow(hm, cmap="jet", alpha=0.5); ax[1, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(cfg["output"]["dir"]) / run / "interpret.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
