# Copyright ...
# Closely mirrors peft/tuners/ia3/model.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import bitsandbytes as bnb

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import ModulesToSaveWrapper

from .layer import LARSLinear
from .config import LARSConfig

from peft.import_utils import is_bnb_available, is_bnb_4bit_available 
from .bnb import LARSLinear4bit, LARSLinear8bitLt 


# --------------------------------------------------
# Model
# --------------------------------------------------

class LARSModel(BaseTuner):
    """
    PEFT model for low-rank activation scaling.

    Closely follows IA3Model structure but:
      - adapters are activation-conditioned
      - adapters are NOT mergeable
      - scaling is input-dependent
    """

    prefix = "lars_"
    tuner_layer_cls = LARSLinear

    def __init__(self, model: nn.Module, config: LARSConfig, adapter_name: str):
        super().__init__(model=model, peft_config=config, adapter_name=adapter_name)

        # HuggingFace convention
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.num_layers = len(model.model.layers)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            self.num_layers = len(model.encoder.layer)
        else:
            raise ValueError("Cannot infer number of transformer layers")

    # --------------------------------------------------
    # Required BaseTuner overrides
    # --------------------------------------------------

    @staticmethod
    def _check_target_module_feedforward(lars_config, key) -> bool:
        """
        Same logic as IA3: regex or exact match.
        """
        if self.peft_config.target_modules is None:
            return False

        for target in self.peft_config.target_modules:
            if re.fullmatch(target, key):
                return True
            if key.endswith(target):
                return True
        return False

    @staticmethod
    def _create_new_module(peft_config: LARSConfig, adapter_name: str, target: nn.Module, **kwargs):
        if is_bnb_available():
            import bitsandbytes as bnb

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        # Unwrap if already an adapter
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # Quantized 4-bit
        if loaded_in_4bit and "bnb" in globals() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            return LARSLinear(base_layer=target, rank=peft_config.rank)

        # Quantized 8-bit
        if loaded_in_8bit and "bnb" in globals() and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            return LARSLinear(base_layer=target, rank=peft_config.rank)

        # Regular linear
        if isinstance(target_base_layer, nn.Linear):
            return LARSLinear(base_layer=target, rank=peft_config.rank)

        return None


    def _create_and_replace(
        self,
        peft_config: LARSConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **kwargs,
    ):
        """
        Replace nn.Linear with LARSLinear.
        """

        if not isinstance(target, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            return

        qkwargs = {
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        new_module = self._create_new_module(peft_config, adapter_name, target, **qkwargs)
        if new_module is not None:
            # Important: keep reference for PEFT bookkeeping
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(
        self,
        parent: nn.Module,
        child_name: str,
        new_module: nn.Module,
        old_module: nn.Module,
    ):
        """
        Same replacement logic as IA3.
        """
        setattr(parent, child_name, new_module)

        device = old_module.weight.device
        new_module.to(device) 
        
        if hasattr(old_module, "weight"):
            new_module.base_layer.weight = old_module.weight
        if hasattr(old_module, "bias") and old_module.bias is not None:
            new_module.base_layer.bias = old_module.bias

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        print("=== MARK TRAINABLE DEBUG ===")
        # 1. Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # 2. Unfreeze only LARS adapter params (floating point only)
        trainable_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, LARSLinear):
                # print(f"Found LARSLinear: {name}")
                for param_name, param in module.named_parameters():
                    # Skip quantized base_layer weights (int8) - they MUST stay frozen
                    if param.dtype not in (torch.float16, torch.float32, torch.bfloat16, torch.float64):
                        # print(f"  SKIP {param_name}: dtype={param.dtype} (quantized, stays frozen)")
                        continue
                    
                    param.requires_grad = True
                    trainable_count += param.numel()
                    # print(f"  Unfroze {param_name}: {param.numel()} params, requires_grad={param.requires_grad}")
        
        print(f"Total LARS trainable params: {trainable_count}")
        print("=== END DEBUG ===")



    # --------------------------------------------------
    # Explicitly unsupported IA3 features
    # --------------------------------------------------

    def merge_and_unload(self):
        raise NotImplementedError(
            "ActivationScaling adapters are activation-conditioned and cannot be merged into base weights."
        )

    def unload(self):
        raise NotImplementedError(
            "ActivationScaling adapters cannot be unloaded independently of the base model."
        )
