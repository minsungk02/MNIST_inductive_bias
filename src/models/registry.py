"""모델 레지스트리: config의 architecture.type 문자열로 모델을 조립한다.

train.py는 이 build_model 하나만 호출한다 -> 모델 추가/교체가 여기 한 곳에서만.
"""
from __future__ import annotations
import torch.nn as nn

from .simple_cnn import build_simple_cnn
from .vit import build_vit


def build_model(cfg: dict) -> nn.Module:
    arch = cfg["architecture"]
    dropout = cfg["regularization"]["dropout"]
    num_classes = 10
    t = arch["type"]

    if t == "simple_cnn":
        return build_simple_cnn(arch, num_classes, dropout)
    if t == "vit":
        return build_vit(arch, num_classes, dropout)
    raise ValueError(f"Unknown architecture.type: {t}")
