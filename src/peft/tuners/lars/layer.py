# Copyright ...
# Adapted to mirror peft/tuners/ia3/layer.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer


class LARSLayer(BaseTunerLayer):
    """
    Low-memory activation-gated PEFT.
    Gating happens in feature space.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        learned_pooling: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.base_layer = base_layer
        self.rank = rank
        self.learned_pooling = learned_pooling

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.A_pool = nn.Linear(self.in_features, rank, bias=False)
        self.B_pool = nn.Linear(rank, self.out_features, bias=False)
        self.rank_gate_x = nn.Linear(self.in_features, rank, bias=False)
        self.rank_gate_h = nn.Linear(rank, rank, bias=False)

        if learned_pooling:
            self.pool_proj = nn.Linear(self.in_features, 1, bias=False)

        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        self.temp1 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.temp2 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.rank_mix = nn.Parameter(torch.eye(rank, dtype=torch.float32))

        self.rank_ffn = nn.Sequential(
            nn.Linear(rank, rank * 4),
            nn.GELU(),
            nn.Linear(rank * 4, rank),
        )

        self.rank_norm = nn.LayerNorm(rank)
        self.reset_lars_parameters()

    def get_base_layer(self) -> nn.Module:
        return self.base_layer

    def reset_lars_parameters(self):
        nn.init.normal_(self.A_pool.weight, std=0.01)
        nn.init.zeros_(self.B_pool.weight)
        nn.init.zeros_(self.rank_gate_x.weight)
        nn.init.zeros_(self.rank_gate_h.weight)
        if self.learned_pooling:
            nn.init.zeros_(self.pool_proj.weight)

    def get_lars_output(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.learned_pooling:
            pool_logits = self.pool_proj(x)                 # [B, S, 1]
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pool = (x * pool_weights).sum(dim=1)          # [B, H]
        else:
            x_pool = x.mean(dim=1) + x[:, -1]               # [B, H]

        h = self.A_pool(x_pool)                             # [B, r]
        h_norm = self.rank_norm(h)

        g = torch.sigmoid(
            self.temp1 * self.rank_gate_x(x_pool) +
            self.temp2 * self.rank_gate_h(h_norm)
        )

        h = h_norm + torch.matmul(g, self.rank_mix)
        h = F.dropout(h, p=0.1, training=self.training)
        h = h + self.rank_ffn(h)

        out = self.B_pool(h)                                # [B, out_features]
        out = out.to(base_out.dtype)

        return base_out + self.alpha.to(base_out.dtype) * out.unsqueeze(1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"learned_pooling={self.learned_pooling}"
        )

    @property
    def weight(self):
        return self.get_base_layer().weight

    @property
    def bias(self):
        return getattr(self.get_base_layer(), "bias", None)


class LARSLinear(nn.Module, LARSLayer):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        learned_pooling: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        LARSLayer.__init__(self, base_layer, rank=rank, learned_pooling=learned_pooling, **kwargs)

        self.get_base_layer().weight.requires_grad = False
        if getattr(self.get_base_layer(), "bias", None) is not None:
            self.get_base_layer().bias.requires_grad = False

    def forward(self, x: torch.Tensor):
        base_out = self.base_layer(x)
        return self.get_lars_output(x, base_out)