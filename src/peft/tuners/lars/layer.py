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

# class LARSLinear(nn.Module, BaseTunerLayer):
#     """
#     PEFT-compatible low-rank activation scaling layer.
#     """

#     def __init__(
#         self,
#         base_layer: nn.Linear,
#         rank: int,
#         activation: str = "silu",
#     ):
#         super().__init__()

#         if not isinstance(base_layer, nn.Linear):
#             raise ValueError("LARSLinear only supports nn.Linear")

#         self.in_features = base_layer.in_features
#         self.out_features = base_layer.out_features
#         self.rank = rank
#         self.scaling = rank

#         self.base_layer = base_layer
#         self.base_layer.weight.requires_grad_(False)
#         if self.base_layer.bias is not None:
#             self.base_layer.bias.requires_grad_(False)

#         self.merged_adapters = []

#         # Low-rank activation adapter
#         self.U = nn.Linear(self.in_features, rank, bias=False)
#         nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
#         self.V = nn.Linear(rank, self.out_features, bias=False)
#         nn.init.kaiming_uniform_(self.V.weight, a=math.sqrt(5))
#         # nn.init.zeros_(self.V.weight)
#         self.norm = nn.LayerNorm(self.in_features)

#         # Nonlinearity
#         if activation == "silu":
#             self.act = nn.SiLU(inplace=True)

#         # Initialization: identity at start (important for stability)
#         # nn.init.zeros_(self.V.weight)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass:
#             h = W x
#             g = 1 + V(phi(U(h)))
#             y = h * g
#         """
#         # h = self.base_layer(x)
#         # # gate = 1.0 + self.V(self.act(self.U(h)))
#         # gate = 1.0 + torch.tanh(self.V(self.act(self.U(self.norm(x)))))

#         # return h * gate
#         with torch.no_grad():
#             h = self.base_layer(x)
#         gate = 1 + 0.5*torch.tanh(self.V(self.act(self.U(self.norm(x)))))
#         y = h*gate 
#         return y


class LARSLinear(nn.Module, BaseTunerLayer):
    def __init__(self, base_layer, rank, block_size=32):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad_(False)

        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        if hasattr(base_layer, "compute_dtype"):
            dtype = base_layer.compute_dtype
        elif base_layer.weight.dtype.is_floating_point:
            dtype = base_layer.weight.dtype
        else:
            # Fallback for quantized weights that don't expose compute_dtype
            dtype = torch.bfloat16

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank

        d_out = self.in_features  
        assert d_out % block_size == 0

        self.block = block_size
        self.g = d_out // block_size

        self.norm = nn.RMSNorm(self.in_features, device=device, dtype=dtype)    
        self.U = nn.Linear(self.in_features, rank, bias=False, device=device, dtype=dtype)
        self.V = nn.Linear(rank, self.g, bias=False, device=device, dtype=dtype)
        
        nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
        nn.init.normal_(self.V.weight, std=1e-4)
        
        self.alpha = nn.Parameter(
            torch.tensor(0.1, device=device, dtype=dtype), 
            requires_grad=True
        )

    def forward(self, x):
        # print(f"DEBUG: x on {x.device}, alpha on {self.alpha.device}")
        input_device = x.device
        input_dtype = x.dtype

        alpha = self.alpha.to(input_device)
        
        x_adapter = x.to(self.alpha.dtype)
        
        z = self.norm.to(input_device)(x_adapter)
        gate_small = 1 + alpha * self.V.to(input_device)(self.U.to(input_device)(z))
        gate = gate_small.repeat_interleave(self.block, dim=-1)
    
        scaled_input = (x_adapter * gate).to(input_dtype)
        
        return self.base_layer(scaled_input)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}"

    @property
    def weight(self):
        return self.base_layer.weight
    @property
    def bias(self):
        return self.base_layer.bias   
