"""Test-time perturbation (train은 절대 건드리지 않음).

perturbation은 정규화 이전의 [0,1] 텐서에 fill=0(=MNIST 검은 배경)으로 적용한 뒤
정규화한다. 이렇게 해야 shift가 만드는 테두리가 배경과 일치해서 인공 edge가 안 생긴다.

- shift : Conv+GAP는 translation-invariant에 가까움 / pos-embed ViT는 취약 (핵심 대비)
- rotation : 둘 다 회전 prior 없음 (exploratory 대조군)
- noise : translation과 무관한 일반 OOD robustness (exploratory)
"""
from __future__ import annotations
import torch
import torchvision.transforms.functional as TF


def make_perturb(kind: str, mag: float, mean: float, std: float, seed: int = 0):
    """[0,1] 텐서(C,H,W)를 받아 perturb 후 정규화해서 반환하는 함수."""
    def fn(img01: torch.Tensor) -> torch.Tensor:
        x = img01
        if kind == "shift" and mag != 0:
            k = int(mag)
            x = TF.affine(x, angle=0.0, translate=[k, k], scale=1.0, shear=[0.0, 0.0], fill=0.0)
        elif kind == "rotation" and mag != 0:
            x = TF.affine(x, angle=float(mag), translate=[0, 0], scale=1.0, shear=[0.0, 0.0], fill=0.0)
        elif kind == "noise" and mag != 0:
            g = torch.Generator().manual_seed(seed)
            x = x + torch.randn(x.shape, generator=g) * float(mag)
            x = x.clamp(0.0, 1.0)
        return TF.normalize(x, [mean], [std])
    return fn
