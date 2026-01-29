# peft/src/peft/tuners/lars/layer.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.checkpoint import checkpoint

class LARSLayer(BaseTunerLayer):
    adapter_layer_names = ("lars_params",)
    
    def __init__(self, base_layer, rank, block_size=32):
            self.base_layer = base_layer 
            self.lars_params = nn.ModuleDict({})
            
            self._disable_adapters = False
            self.merged_adapters = [] # not used, for PEFT compatability

            self.rank = rank
            self.block_size = block_size

            base = self.get_base_layer()
            if isinstance(base, nn.Linear):
                in_features, out_features = base.in_features, base.out_features
            # elif isinstance(base, Conv1D):
            #     in_features, out_features = (
            #         base.weight.ds_shape if hasattr(base.weight, "ds_shape") else base.weight.shape
            #     )
            else:
                raise ValueError(f"Unsupported base layer type for LARS: {type(base)}")

            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features

            if self.in_features % self.block_size != 0:
                raise ValueError(
                    f"LARS requires in_features divisible by block_size: {self.in_features=} {self.block_size=}"
                )

            self.g = self.in_features // block_size

    def _infer_adapter_dtype_device(self):
        base = self.get_base_layer()
        # Prefer weight dtype if it exists and is floating point; else default fp16
        if hasattr(base, "weight") and base.weight is not None and base.weight.is_floating_point():
            return base.weight.dtype, base.weight.device
        # bnb int8 weight is not floating; choose fp16 on the same device as the module parameters
        device = next(self.base_layer.parameters(), torch.empty(0)).device
        return torch.float16, device

    def update_layer(self, adapter_name: str, init_lars_weights: bool, inference_mode: bool = False, **kwargs):
        dtype, device = self._infer_adapter_dtype_device()
        
        U = nn.Parameter(torch.empty((self.in_features, self.rank), device=device, dtype=dtype))
        V = nn.Parameter(torch.empty((self.rank, self.g), device=device, dtype=dtype))
        alpha = nn.Parameter(torch.tensor(0.1, device=device, dtype=dtype))

        self.lars_params[adapter_name] = nn.ParameterDict(
            {"U": U, "V": V, "alpha": alpha}
        )

        if init_lars_weights:
            nn.init.kaiming_uniform_(self.lars_params[adapter_name]["U"], a=math.sqrt(5))
            nn.init.normal_(self.lars_params[adapter_name]["V"], std=1e-4)
            with torch.no_grad():
                self.lars_params[adapter_name]["alpha"].fill_(0.1)

        # Move adapter to device/dtype of base layer (PEFT helper)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_lars_parameters(self, adapter_name: str):
        # Keep U/V as initialized; set alpha to small default.
        if adapter_name in self.lars_params:
            with torch.no_grad():
                self.lars_params[adapter_name]["alpha"].fill_(0.1)

    def _compute_gate_logic(self, x):
        z = F.rms_norm(x, (self.in_features,), eps=1e-5)        
        
        gate_accum = None
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lars_params:
                continue
            p = self.lars_params[active_adapter]
            # Projection logic in FP32
            U = p["U"].to(dtype=x.dtype)
            V = p["V"].to(dtype=x.dtype)
            alpha = p["alpha"].to(dtype=x.dtype)

            proj = (z @ U) @ V
            inc = 1.0 + alpha * proj
            gate_accum = inc if gate_accum is None else gate_accum * inc # for multiple adapters
        
        if gate_accum is None:
            gate_accum = torch.ones(z.shape[:-1] + (self.g,), device=z.device, dtype=z.dtype)
  
        return gate_accum

class Linear(nn.Module, LARSLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        rank: int,
        block_size: int = 32,
        init_lars_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        LARSLayer.__init__(self, base_layer, rank=rank, block_size=block_size)

        # Freeze pretrained weights
        base = self.get_base_layer()
        if hasattr(base, "weight") and base.weight is not None:
            base.weight.requires_grad = False
        if getattr(base, "bias", None) is not None:
            base.bias.requires_grad = False

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, init_lars_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs)

        gate_small = self._compute_gate_logic(x).unsqueeze(-1)        
        x_view = x.view(*x.shape[:-1], self.g, self.block_size)  
        x_gated = (x_view * gate_small).reshape_as(x)   

        return self.base_layer(x_gated, *args, **kwargs)          

    def __repr__(self) -> str:
        return "lars." + super().__repr__()