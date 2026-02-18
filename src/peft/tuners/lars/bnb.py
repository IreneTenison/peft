import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from torch.utils.checkpoint import checkpoint
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from .layer import LARSLayer

if is_bnb_available():
    import bitsandbytes as bnb

    class Linear8bitLt(torch.nn.Module, LARSLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            rank: int,
            block_size: int = 32,
            init_lars_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            LARSLayer.__init__(self, base_layer, rank=rank, block_size=block_size)
        
            self.get_base_layer().weight.requires_grad = False
            if getattr(self.get_base_layer(), "bias", None) is not None:
                self.get_base_layer().bias.requires_grad = False

            self._active_adapter = adapter_name
            self.update_layer(adapter_name, init_lars_weights)

        def forward(self, x, *args, **kwargs):
            if self.disable_adapters:
                return self.base_layer(x, *args, **kwargs)

            def gated_x(t):
                gate_small = self._compute_gate_logic(t).unsqueeze(-1)  # [..., g, 1]
                t_view = t.view(*t.shape[:-1], self.g, self.block_size)
                return (t_view * gate_small).reshape_as(t)

            # gate_small = self._compute_gate_logic(x).unsqueeze(-1)  # [..., g, 1]
            # x_view = x.view(*x.shape[:-1], self.g, self.block_size)
            # x_gated = (x_view * gate_small).reshape_as(x)
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                x_gated = checkpoint(gated_x, x, use_reentrant=False)
            # x_gated = checkpoint(gated_x, x, use_reentrant=False)
            # print("x_gated: ", x_gated.dtype)
            return self.base_layer(x_gated, *args, **kwargs)

        def __repr__(self) -> str:
            return "lars." + super().__repr__()


if is_bnb_4bit_available():
    import bitsandbytes as bnb 

    class Linear4bit(torch.nn.Module, LARSLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            rank: int,
            block_size: int = 32,
            init_lars_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            LARSLayer.__init__(self, base_layer, rank=rank, block_size=block_size)

            # Freeze pretrained weight
            self.get_base_layer().weight.requires_grad = False
            if getattr(self.get_base_layer(), "bias", None) is not None:
                self.get_base_layer().bias.requires_grad = False

            self._active_adapter = adapter_name
            self.update_layer(adapter_name, init_lars_weights)

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if self.disable_adapters:
                return self.base_layer(x, *args, **kwargs)

            def gated_x(t):
                gate_small = self._compute_gate_logic(t).unsqueeze(-1)  # [..., g, 1]
                t_view = t.view(*t.shape[:-1], self.g, self.block_size)
                return (t_view * gate_small).reshape_as(t)

            # gate_small = self._compute_gate_logic(x).unsqueeze(-1)   # [..., g, 1]
            # x_view = x.view(*x.shape[:-1], self.g, self.block_size)
            # x_gated = (x_view * gate_small).reshape_as(x)

            x_gated = checkpoint(gated_x, x, use_reentrant=False)
            out = self.base_layer(x_gated, *args, **kwargs)
            return out.clone()   # keep this workaround if you want

        def __repr__(self) -> str:
            return "lars." + super().__repr__()