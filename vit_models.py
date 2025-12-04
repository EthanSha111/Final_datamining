"""
vit_models.py
--------------

Vision Transformer (ViT) components with pluggable positional encoding modules.

This file contains:
  - PatchEmbedding: 2D image to patch sequence.
  - MultiHeadSelfAttention: MHA that calls a positional module on (Q, K, pos).
  - TransformerEncoderBlock: Standard encoder block (MHA + MLP).
  - VisionTransformer: End-to-end ViT wrapper for classification.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from positional_encodings import (
    RoPEPositionalModule,
    CayleyStringPE,
    ReflectionStringPE,
    SparseCayleyStringPE,
)


class PatchEmbedding(nn.Module):
    """
    Image to patch embedding using a convolution layer.

    Args:
        img_size:   Input image size (assumed square).
        patch_size: Patch size (assumed square and divides img_size).
        in_chans:   Number of input channels (1 for MNIST, 3 for CIFAR-10).
        emb_dim:    Embedding dimension.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.emb_dim = emb_dim

        self.proj = nn.Conv2d(
            in_chans,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.n_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            patches: (B, N_patches, emb_dim)
        """
        b, c, h, w = x.shape
        assert (
            h == self.img_size and w == self.img_size
        ), f"Expected images of size {self.img_size}x{self.img_size}, got {h}x{w}"

        x = self.proj(x)  # (B, emb_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, emb_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, emb_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with a pluggable positional encoding module.

    The positional module is expected to implement:
        q_pos, k_pos = pos_module(q, k, pos)
    where q, k: (B, N, D) and pos: (N,) or (B, N).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        pos_module: nn.Module,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), "d_model must be divisible by n_heads for multi-head attention"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.pos_module = pos_module
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, D) input sequence.
            pos: (N,) or (B, N) positional indices.

        Returns:
            out: (B, N, D) attended sequence.
        """
        b, n, d = x.shape

        q = self.W_q(x)  # (B, N, D)
        k = self.W_k(x)
        v = self.W_v(x)

        # Apply positional encoding before splitting into heads
        q_pos, k_pos = self.pos_module(q, k, pos)

        # Reshape into heads
        q_pos = q_pos.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, N, d_h)
        k_pos = k_pos.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, N, d_h)
        v = v.view(b, n, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, N, d_h)

        # Scaled dot-product attention
        scores = torch.matmul(q_pos, k_pos.transpose(-2, -1))  # (B, h, N, N)
        scores = scores / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)  # (B, h, N, d_h)

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(b, n, d)  # (B, N, D)
        out = self.W_o(context)
        out = self.proj_dropout(out)

        return out


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block: MHA + MLP with residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        pos_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if pos_module is None:
            raise ValueError("pos_module must be provided for TransformerEncoderBlock.")

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            pos_module=pos_module,
            attn_dropout=attn_drop,
            proj_dropout=drop,
        )
        self.norm2 = nn.LayerNorm(d_model)

        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, D)
            pos: (N,) or (B, N)

        Returns:
            x:   (B, N, D)
        """
        # Attention block
        x = x + self.attn(self.norm1(x), pos)
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification with pluggable positional encodings.

    Args:
        img_size:    Input image size (assumed square).
        patch_size:  Patch size (assumed square).
        in_chans:    Number of input channels.
        num_classes: Number of output classes.
        emb_dim:     Embedding dimension.
        depth:       Number of Transformer encoder blocks.
        n_heads:     Number of attention heads.
        pos_module:  Positional encoding module instance.
        mlp_ratio:   Hidden size ratio for MLP.
        drop_rate:   Dropout rate.
        attn_drop:   Attention dropout rate.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        num_classes: int,
        emb_dim: int,
        depth: int,
        n_heads: int,
        pos_module: nn.Module,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            emb_dim=emb_dim,
        )
        self.n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # Fixed positional indices (CLS + patches)
        self.register_buffer(
            "pos_indices",
            torch.arange(self.n_patches + 1, dtype=torch.long),
            persistent=False,
        )

        # Share the same positional module across all blocks by default
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=emb_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    pos_module=pos_module,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            logits: (B, num_classes)
        """
        b = x.size(0)
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(b, 1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        pos = self.pos_indices.to(x.device)  # (N+1,)

        for block in self.blocks:
            x = block(x, pos)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        logits = self.head(cls_out)
        return logits


__all__ = [
    "PatchEmbedding",
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "VisionTransformer",
    "RoPEPositionalModule",
    "CayleyStringPE",
    "ReflectionStringPE",
    "SparseCayleyStringPE",
]


