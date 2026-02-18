# peft/src/peft/tuners/lars/layer.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from torch.utils.checkpoint import checkpoint


LARS_PARAMETER_BANK = {}

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

            self.layer_id = None
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features

            if self.in_features % self.block_size != 0:
                raise ValueError(
                    f"LARS requires in_features divisible by block_size: {self.in_features=} {self.block_size=}"
                )

            self.g = self.in_features // block_size

    def _infer_adapter_dtype_device(self):
        base = self.get_base_layer()

        if hasattr(base, "compute_dtype"):
            return base.compute_dtype, base.weight.device

        if hasattr(base, "weight") and base.weight is not None and base.weight.is_floating_point():
            return base.weight.dtype, base.weight.device

        device = next(self.base_layer.parameters(), torch.empty(0)).device
        return torch.bfloat16, device

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

    def _compute_gate_logic(self, x: torch.Tensor) -> torch.Tensor:
        # print("x type: ", x.dtype)
        # x = x.to(torch.float16)               
        z = F.rms_norm(x, (self.in_features,), eps=1e-5) 
        # z = F.rms_norm(x, (self.in_features,), eps=1e-5)

        gate_accum = None
        for active_adapter in self.active_adapters:
            # 2. Check if the "Bake" (Vectorization) is complete
            if active_adapter in LARS_PARAMETER_BANK:
                bank = LARS_PARAMETER_BANK[active_adapter]
                
                # Retrieve the slices for this specific layer ID
                start_u, end_u = bank['indices_u'][self.layer_id]
                start_v, end_v = bank['indices_v'][self.layer_id]
                
                # .view() ensures we are looking at the boulder, not making a new pebble
                U = bank['U'][start_u:end_u].view(self.in_features, self.rank)
                V = bank['V'][start_v:end_v].view(self.rank, self.g)
                alpha = bank['alpha'][self.layer_id]
            else:
                # Fallback to standard params if vectorization hasn't run yet
                p = self.lars_params[active_adapter]
                if p is None: continue
                U, V, alpha = p["U"], p["V"], p["alpha"]

            proj = (z @ U) @ V          
            inc = proj.mul(alpha).add_(1.0)
            gate_accum = inc if gate_accum is None else gate_accum.mul_(inc)

        if gate_accum is None:
            gate_accum = torch.ones(z.shape[:-1] + (self.g,), device=z.device, dtype=x.dtype)

        return gate_accum

        # gate_accum = None
        # for active_adapter in self.active_adapters:
        #     p = self.lars_params[active_adapter]
        #     if p is None:
        #         continue
        #     # U, V, alpha = p["U"].to(torch.float16), p["V"].to(torch.float16), p["alpha"].to(torch.float16)
        #     U, V, alpha = p["U"], p["V"], p["alpha"]
        #     # print("x", x.dtype, "z", z.dtype, "U", U.dtype)

        #     proj = (z @ U) @ V          
        #     inc = proj.mul(alpha).add_(1.0)
        #     gate_accum = inc if gate_accum is None else gate_accum.mul_(inc)

        # if gate_accum is None:
        #     gate_accum = torch.ones(z.shape[:-1] + (self.g,), device=z.device, dtype=x.dtype)

        # return gate_accum

def finalize_lars_vectorization(peft_model, adapter_name):
    """
    Finds all LARSLayers and glues their parameters into a contiguous bank.
    """
    lars_layers = [m for m in peft_model.modules() if isinstance(m, LARSLayer)]
    if not lars_layers: return

    device = lars_layers[0].get_base_layer().weight.device
    dtype = getattr(lars_layers[0].get_base_layer(), "compute_dtype", torch.bfloat16)
    
    total_u = sum(l.in_features * l.rank for l in lars_layers)
    total_v = sum(l.rank * l.g for l in lars_layers)
    num_layers = len(lars_layers)

    # Create the Vectorized Boulders
    bank_u = nn.Parameter(torch.empty(total_u, device=device, dtype=dtype))
    bank_v = nn.Parameter(torch.empty(total_v, device=device, dtype=dtype))
    bank_alpha = nn.Parameter(torch.empty(num_layers, device=device, dtype=dtype))

    indices_u, indices_v = [], []
    curr_u, curr_v = 0, 0
    
    for i, layer in enumerate(lars_layers):
        layer.layer_id = i 
        
        # Copy existing initialized weights into the bank
        p = layer.lars_params[adapter_name]
        
        u_flat = p["U"].data.view(-1)
        v_flat = p["V"].data.view(-1)
        
        len_u, len_v = u_flat.numel(), v_flat.numel()
        
        bank_u.data[curr_u : curr_u + len_u].copy_(u_flat)
        bank_v.data[curr_v : curr_v + len_v].copy_(v_flat)
        bank_alpha.data[i].copy_(p["alpha"].data)
        
        indices_u.append((curr_u, curr_u + len_u))
        indices_v.append((curr_v, curr_v + len_v))
        
        curr_u += len_u
        curr_v += len_v

    # Register the bank globally
    LARS_PARAMETER_BANK[adapter_name] = {
        'U': bank_u, 'V': bank_v, 'alpha': bank_alpha,
        'indices_u': indices_u, 'indices_v': indices_v
    }
    
    # Register the banks as parameters on the PeftModel so they move to GPU & get saved
    peft_model.register_parameter(f"{adapter_name}_lars_bank_u", bank_u)
    peft_model.register_parameter(f"{adapter_name}_lars_bank_v", bank_v)
    peft_model.register_parameter(f"{adapter_name}_lars_bank_alpha", bank_alpha)

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