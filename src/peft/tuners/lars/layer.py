# Copyright ...
# Adapted to mirror peft/tuners/ia3/layer.py

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.checkpoint import checkpoint

class LARSLinear(nn.Module, BaseTunerLayer):
    """
    Low-memory activation-gated PEFT.
    Gating happens in feature space (critical).
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 8, learned_pooling: bool = False):
        # only per_token = True works
        super().__init__()
        self.base = base_layer
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_in = base_layer.in_features
        d_out = base_layer.out_features
        self.rank = rank
        self.learned_pooling = learned_pooling

        self.A_pool = nn.Linear(d_in, self.rank, bias=False)
        self.B_pool = nn.Linear(self.rank, d_out, bias=False)
       
        nn.init.normal_(self.A_pool.weight, std=0.01)
        nn.init.zeros_(self.B_pool.weight)

        self.rank_gate_x = nn.Linear(d_in, self.rank, bias=False)
        self.rank_gate_h = nn.Linear(self.rank, self.rank, bias=False)
        nn.init.zeros_(self.rank_gate_x.weight)
        nn.init.zeros_(self.rank_gate_h.weight)

        if self.learned_pooling:
            self.pool_proj = nn.Linear(d_in, 1, bias=False) 
            nn.init.zeros_(self.pool_proj.weight)

        self.alpha = nn.Parameter(torch.ones(1)*0.5)
        # self.beta = nn.Parameter(torch.ones(1)*0.1)
        self.temp1 = nn.Parameter(torch.ones(1))
        self.temp2 = nn.Parameter(torch.ones(1))

        self.rank_ffn = nn.Sequential(
            nn.Linear(rank, rank * 4),
            nn.GELU(),
            nn.Linear(rank * 4, rank)
        )

        self.rank_mix = nn.Parameter(torch.eye(rank, rank))
        self.rank_norm = nn.LayerNorm(rank)



    def forward(self, x):
        """
        x: [B,S,d] or [B,d]
        """
        dtype = x.float()
        self.A_pool = self.A_pool.to(dtype)
        self.B_pool = self.B_pool.to(dtype)
        self.rank_gate_x = self.rank_gate_x.to(dtype)
        self.rank_gate_h = self.rank_gate_h.to(dtype)
        if self.learned_pooling:
            self.pool_proj = self.pool_proj.to(dtype)
        self.rank_mix = self.rank_mix.to(dtype)
        self.rank_ffn = self.rank_ffn.to(dtype)
        self.rank_norm = self.rank_norm.to(dtype)
        self.alpha = self.alpha.to(dtype)
        self.temp1 = self.temp1.to(dtype)
        self.temp2 = self.temp2.to(dtype)

        B,S,d = x.shape
        base_out = self.base(x)
        if self.learned_pooling:
            pool_logits = self.pool_proj(x)                # [B, S, 1]
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pool = (x * pool_weights).sum(dim=1)
        else:
            x_pool = x.mean(dim=1) + x[:, -1] 

        h = self.A_pool(x_pool)  # [B,S,r]

        # h_norm = self.rank_norm(h)
        # g = torch.sigmoid(self.temp1 * self.rank_gate_x(x_pool) + self.temp2 * self.rank_gate_h(h_norm))
        g = h
        # h = g
        h_mixed = torch.matmul(g, self.rank_mix)
        h = h_mixed
        # h = h_norm + h_mixed
        h = F.dropout(h, p=0.1, training=self.training)
        h = h + self.rank_ffn(h) 

    #     out = self.B_pool(h)
    #     out = base_out + self.alpha * out.unsqueeze(1)
    #     return out






    # ---------------------------
    # PEFT compatibility
    # ---------------------------

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}"
        )

    @property
    def weight(self):
        """
        Expose base weight for PEFT / HF compatibility.
        """
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias
