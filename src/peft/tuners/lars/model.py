# Copyright ...
# Closely mirrors peft/tuners/ia3/model.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import ModulesToSaveWrapper

from .layer import LARSLinear
from .config import LARSConfig


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

        # HuggingFace convention (fixed for T5)
        if hasattr(model, "model"):
            # Check for encoder + decoder blocks
            if hasattr(model.model, "encoder") and hasattr(model.model.encoder, "block"):
                self.num_encoder_layers = len(model.model.encoder.block)
            else:
                self.num_encoder_layers = 0

            if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "block"):
                self.num_decoder_layers = len(model.model.decoder.block)
            else:
                self.num_decoder_layers = 0

            self.num_layers = self.num_encoder_layers + self.num_decoder_layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "block"):
            self.num_encoder_layers = len(model.encoder.block)
            self.num_decoder_layers = len(model.decoder.block) if hasattr(model, "decoder") else 0
            self.num_layers = self.num_encoder_layers + self.num_decoder_layers
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

        if not isinstance(target, nn.Linear):
            return

        new_module = LARSLinear(
            base_layer=target,
            rank=peft_config.rank,
        )

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

        # Preserve device placement
        new_module.to(old_module.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module):
        # 1. Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False
        
        if hasattr(self.model.config, "enable_input_reqs"):
            self.model.config.enable_input_reqs = True


        # 2. Unfreeze only LARS adapter params
        for module in self.model.modules():

            if isinstance(module, LARSLinear):
                for p in module.A_pool.parameters():
                    p.requires_grad = True
                for p in module.B_pool.parameters():
                    p.requires_grad = True
                # for p in module.experts.parameters():
                #     p.requires_grad = True
                for p in module.rank_ffn.parameters():
                    p.requires_grad = True
                for p in module.rank_gate_x.parameters():
                    p.requires_grad = True
                for p in module.rank_gate_h.parameters():
                    p.requires_grad = True
                for p in module.rank_norm.parameters():
                    p.requires_grad = True
                # for p in module.token_scale.parameters():
                #     p.requires_grad = True
                module.pool_proj.requires_grad = True
                module.alpha.requires_grad = True
                # module.beta.requires_grad = True
                module.temp1.requires_grad = True
                module.temp2.requires_grad = True
                module.rank_mix.requires_grad = True

                
                




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
