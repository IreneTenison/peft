import torch
from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .layer import LARSLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, LARSLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            rank: int = 8,
            learned_pooling: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LARSLayer.__init__(self, base_layer, rank=rank, learned_pooling=learned_pooling, **kwargs)

            self.get_base_layer().weight.requires_grad = False
            if getattr(self.get_base_layer(), "bias", None) is not None:
                self.get_base_layer().bias.requires_grad = False

            self.is_loaded_in_8bit = True

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            base_out = self.base_layer(x)
            return self.get_lars_output(x, base_out)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lars." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, LARSLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            rank: int = 8,
            learned_pooling: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            LARSLayer.__init__(self, base_layer, rank=rank, learned_pooling=learned_pooling, **kwargs)

            self.get_base_layer().weight.requires_grad = False
            if getattr(self.get_base_layer(), "bias", None) is not None:
                self.get_base_layer().bias.requires_grad = False

            self.is_loaded_in_4bit = True

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            base_out = self.base_layer(x)
            result = self.get_lars_output(x, base_out)
            return result.clone()

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lars." + rep