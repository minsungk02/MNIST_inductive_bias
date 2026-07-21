"""Small Vision Transformer.

공간 구조에 대한 prior가 거의 없음 -> patch 간 관계를 데이터로부터 학습해야 함.
CNN과 param band를 맞추기 위해 embed_dim / depth 를 config로 조절한다.

attention weight를 뽑아 heatmap 시각화가 가능하도록 forward에 return_attn 옵션을 둠
(발표/보고서용 해석가능성 확장).

[2차 추가실험용 옵션 — config flag, 기본 off면 원본 ViT와 완전 동일]
  - spt: Shifted Patch Tokenization (Lee 2021) — 이미지를 대각 4방향으로 밀어 원본과
         겹쳐 토큰화 -> 각 패치가 '이웃 픽셀 정보'까지 담음 (ERF의 패치 고립·locality 부재 보완).
  - lsa: Locality Self-Attention (Lee 2021) — 학습가능 temperature + 대각 마스킹으로
         attention을 덜 균일하게(국소적으로) (attention distance ~2.0 평평 문제 보완).
  두 옵션 모두 pos_embed(절대 위치)는 건드리지 않음 -> shift/equivariance는 그대로 나빠야 정상.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# 패치 임베딩
# ------------------------------------------------------------------ #
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


class ShiftedPatchEmbed(nn.Module):
    """SPT: 원본 + 대각 4방향 shift(zero-fill)를 채널로 concat 후 patch 임베딩.

    in_ch -> in_ch*5 로 늘어난 뒤 conv. 각 패치가 이웃 픽셀 정보까지 담게 된다.
    """
    def __init__(self, img_size=28, patch_size=7, in_ch=1, embed_dim=64, shift=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.shift = int(shift) if shift is not None else max(1, patch_size // 2)
        self.proj = nn.Conv2d(in_ch * 5, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)      # 논문 권장(토큰 정규화)

    @staticmethod
    def _shift(x, dy, dx):
        """zero-fill 평행이동 (dy>0=아래, dx>0=오른쪽)."""
        B, C, H, W = x.shape
        out = x.new_zeros(B, C, H, W)
        sy = slice(max(0, -dy), H - max(0, dy))
        dyy = slice(max(0, dy), H - max(0, -dy))
        sx = slice(max(0, -dx), W - max(0, dx))
        dxx = slice(max(0, dx), W - max(0, -dx))
        out[:, :, dyy, dxx] = x[:, :, sy, sx]
        return out

    def forward(self, x):
        s = self.shift
        xs = torch.cat([x,
                        self._shift(x, -s, -s), self._shift(x, -s, s),
                        self._shift(x, s, -s), self._shift(x, s, s)], dim=1)  # [B, 5C, H, W]
        x = self.proj(xs).flatten(2).transpose(1, 2)    # [B, N, E]
        return self.norm(x)


# ------------------------------------------------------------------ #
# Attention
# ------------------------------------------------------------------ #
class LSAttention(nn.Module):
    """LSA: 학습가능 temperature + 대각 마스킹(자기 self-attention 제거).

    return_attn=True면 헤드 평균 attention [B, N, N] 반환 (attention_distance 지표 호환).
    """
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        # 학습가능 temperature (초기값 = 1/sqrt(head_dim), 표준 scale와 동일 출발)
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** -0.5))

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, heads, N, hd]
        attn = (q @ k.transpose(-2, -1)) * self.temperature  # [B, heads, N, N]
        diag = torch.eye(N, device=x.device, dtype=torch.bool)
        attn = attn.masked_fill(diag, float("-inf"))         # 대각 마스킹
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        if return_attn:
            return out, attn.mean(dim=1)                      # 헤드 평균 [B, N, N]
        return out


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout=0.0, lsa=False):
        super().__init__()
        self.lsa = lsa
        self.norm1 = nn.LayerNorm(dim)
        if lsa:
            self.attn = LSAttention(dim, heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        h = self.norm1(x)
        if self.lsa:
            if return_attn:
                a, attn_w = self.attn(h, return_attn=True)
            else:
                a, attn_w = self.attn(h), None
        else:
            a, attn_w = self.attn(h, h, h, need_weights=return_attn, average_attn_weights=True)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return (x, attn_w) if return_attn else x


# ------------------------------------------------------------------ #
class ViT(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_ch=1, num_classes=10,
                 embed_dim=64, depth=4, heads=4, mlp_ratio=2.0, dropout=0.0,
                 spt=False, lsa=False):
        super().__init__()
        if spt:
            self.patch_embed = ShiftedPatchEmbed(img_size, patch_size, in_ch, embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, heads, mlp_ratio, dropout, lsa=lsa) for _ in range(depth)
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
        spt=arch.get("spt", False),       # 2차 옵션 (기본 off = 원본 ViT)
        lsa=arch.get("lsa", False),
    )
