"""의도적으로 미니멀한 canonical CNN.

residual / BN / 복잡한 정규화를 일부러 넣지 않는다. conv의 본질
(local receptive field + weight sharing + pooling 계층 요약)만 담아서
'inductive bias 그 자체'의 효과를 순수하게 측정하기 위함.

Flatten+FC 대신 GAP를 써서 파라미터가 conv에 배분되게 한다
(거대 FC로 param을 부풀리면 band 매칭이 왜곡됨).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_ch: int = 1, channels=(40, 80, 160),
                 num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 14->7
            nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(),                       # 7x7 유지
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean(dim=(2, 3))   # Global Average Pooling
        x = self.dropout(x)
        return self.head(x)


def build_simple_cnn(arch: dict, num_classes: int, dropout: float) -> SimpleCNN:
    return SimpleCNN(
        in_ch=1,
        channels=tuple(arch["channels"]),
        num_classes=num_classes,
        dropout=dropout,
    )
