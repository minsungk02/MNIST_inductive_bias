"""Small Vision Transformer.

공간 구조에 대한 prior가 거의 없음 -> patch 간 관계를 데이터로부터 학습해야 함.
CNN과 param band를 맞추기 위해 embed_dim / depth 를 config로 조절한다.

attention weight를 뽑아 heatmap 시각화가 가능하도록 forward에 return_attn 옵션을 둠
(발표/보고서용 해석가능성 확장).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, embed_dim=64):
        super().__init__()
        assert img_size % patch_size == 0, "img_size는 patch_size로 나누어떨어져야 함"
        self.n_patches = (img_size // patch_size) ** 2
        # Conv stride=patch 로 patch 임베딩 (flatten+Linear와 등가, 더 깔끔)
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # B, E, H/p, W/p
        x = x.flatten(2).transpose(1, 2) # B, N, E
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        a, attn_w = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=True)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return (x, attn_w) if return_attn else x


class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, num_classes=10,
                 embed_dim=64, depth=4, heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                attns.append(a)
            else:
                x = blk(x)

        x = self.norm(x)
        logits = self.head(x[:, 0])       # cls token
        return (logits, attns) if return_attn else logits


def build_vit(arch: dict, num_classes: int, dropout: float) -> ViT:
    return ViT(
        img_size=28,
        patch_size=arch["patch_size"],
        in_ch=1,
        num_classes=num_classes,
        embed_dim=arch["embed_dim"],
        depth=arch["depth"],
        heads=arch["heads"],
        mlp_ratio=arch["mlp_ratio"],
        dropout=dropout,
    )
