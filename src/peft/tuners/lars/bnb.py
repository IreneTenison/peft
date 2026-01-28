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

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            if self.disable_adapters:
                return self.base_layer(x, *args, **kwargs)

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()

            gate = self._compute_gate_logic(x)  # fp32
            out = self.base_layer(x * gate.to(x.dtype), *args, **kwargs)

            if requires_conversion:
                out = out.to(out.dtype)
            return out

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

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()

            gate = self._compute_gate_logic(x)
            out = self.base_layer(x * gate.to(x.dtype), *args, **kwargs)

            out = out.clone()  # IA3/LoRA workaround for some torch/bnb combos

            if requires_conversion:
                out = out.to(out.dtype)
            return out

        def __repr__(self) -> str:
            return "lars." + super().__repr__()