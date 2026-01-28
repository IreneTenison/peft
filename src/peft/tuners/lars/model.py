# Copyright ...
# Closely mirrors peft/tuners/ia3/model.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import ModulesToSaveWrapper
from peft.import_utils import is_bnb_available, is_bnb_4bit_available 

from .layer import Linear, LARSLayer
from .config import LARSConfig
from .bnb import Linear4bit, Linear8bitLt 


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
    tuner_layer_cls = LARSLayer

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
    # Required BaseTuner overrides (COMPLETE)
    # --------------------------------------------------

    @staticmethod
    def _check_target_module_exists(lars_config: LARSConfig, key: str) -> bool:
        """Check if key matches target_modules."""
        return check_target_module_exists(lars_config, key)

    @staticmethod
    def _check_target_module_feedforward(lars_config: LARSConfig, key: str) -> bool:
        """True for FFN layers (input gating), False for attention (output gating)."""
        if lars_config.feedforward_modules is None:
            return False
        return any(re.fullmatch(target, key) or key.endswith(target) 
                  for target in lars_config.feedforward_modules)

    @staticmethod
    def _create_new_module(
        lars_config: LARSConfig, 
        adapter_name: str, 
        target: nn.Module, 
        **kwargs
    ):
        if is_bnb_available():
            import bitsandbytes as bnb
            from .bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from .bnb import Linear4bit
        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)

        init_lars_weights = kwargs.pop("init_lars_weights", True)
        rank = kwargs.pop("rank")
        block_size = kwargs.pop("block_size")

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        # quantized dispatch
        if loaded_in_8bit and is_bnb_available() and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            # match IA3's extra args pass-through
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            return Linear8bitLt(
                target,
                adapter_name,
                rank=rank,
                block_size=block_size,
                init_lars_weights=init_lars_weights,
                **eightbit_kwargs,
            )

        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            return Linear4bit(
                target,
                adapter_name,
                rank=rank,
                block_size=block_size,
                init_lars_weights=init_lars_weights,
                **fourbit_kwargs,
            )

        # non-quant dispatch
        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs.get("fan_in_fan_out", False):
                warnings.warn(
                    "fan_in_fan_out=True but target is nn.Linear; setting fan_in_fan_out=False."
                )
                kwargs["fan_in_fan_out"] = lars_config.fan_in_fan_out = False
            return Linear(
                target,
                adapter_name,
                rank=rank,
                block_size=block_size,
                init_lars_weights=init_lars_weights,
                **kwargs,
            )

        # if isinstance(target_base_layer, Conv1D):
        #     # Conv1D stores weight like (fan_in, fan_out)
        #     if not kwargs.get("fan_in_fan_out", False):
        #         warnings.warn(
        #             "fan_in_fan_out=False but target is Conv1D; setting fan_in_fan_out=True."
        #         )
        #         kwargs["fan_in_fan_out"] = lars_config.fan_in_fan_out = True
        #     return Linear(
        #         target,
        #         adapter_name,
        #         rank=rank,
        #         block_size=block_size,
        #         init_lars_weights=init_lars_weights,
        #         **kwargs,
        #     )

        raise ValueError(f"Target module {target} not supported for LARS.")

    def _create_and_replace(
        self,
        lars_config: LARSConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ):
        kwargs = {
            "fan_in_fan_out": lars_config.fan_in_fan_out,
            "init_lars_weights": lars_config.init_lars_weights,
            "rank": lars_config.rank,
            "block_size": lars_config.block_size,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, LARSLayer):
            target.update_layer(adapter_name, lars_config.init_lars_weights)
        else:
            new_module = self._create_new_module(lars_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(
        self,
        parent: nn.Module,
        child_name: str,
        new_module: nn.Module,
        old_module: nn.Module,
    ):
        """Replace module and move to correct device."""
        device = old_module.weight.device
        new_module.to(device) 
        setattr(parent, child_name, new_module)

    @staticmethod
    def _prepare_adapter_config(peft_config: LARSConfig, model_config: dict):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config` for LARS.")
        return peft_config

    # --------------------------------------------------
    # Unsupported (LARS cannot merge/unload)
    # --------------------------------------------------

    def merge_and_unload(self):
        raise NotImplementedError(
            "LARS adapters are activation-conditioned and cannot be merged."
        )

    def unload(self):
        raise NotImplementedError(
            "LARS adapters cannot be unloaded independently."
        )
