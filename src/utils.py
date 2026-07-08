"""공통 유틸: 재현성 seed, device 자동 선택, 파라미터 카운트."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """init / shuffle / dropout mask 등 모든 무작위성을 통제."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """서버(3090 Ti)에서는 cuda, Mac 개발 시엔 mps, 그 외 cpu.

    같은 레포가 두 환경에서 그대로 돌아가도록 자동 감지한다.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """(total, trainable) 파라미터 수."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
