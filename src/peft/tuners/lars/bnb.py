# lars/bnb.py
import bitsandbytes as bnb
from .layer import LARSLinear


class LARSLinear4bit(bnb.nn.Linear4bit):
    def __init__(self, base_layer: bnb.nn.Linear4bit, adapter_name, **kwargs):
        # Extract dims from existing layer
        input_features = base_layer.in_features
        output_features = base_layer.out_features
        bias = base_layer.bias is not None

        compute_dtype = kwargs.pop("compute_dtype", base_layer.compute_dtype)
        compress_statistics = kwargs.pop("compress_statistics", base_layer.weight.compress_statistics)
        quant_type = kwargs.pop("quant_type", base_layer.weight.quant_type)
        rank = kwargs.pop("rank", None)
        block_size = kwargs.pop("block_size", 32)

        super().__init__(
            input_features=input_features,
            output_features=output_features,
            bias=bias,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )

        # Reuse quantized weights
        self.weight = base_layer.weight

        # Wrap with your existing LARSLinear, **without** changing it
        if rank is None:
            raise ValueError("LARSLinear4bit requires 'rank' kwarg from peft_config.")
        self.lars = LARSLinear(base_layer=self, rank=rank, block_size=block_size)

    def forward(self, x, *args, **kwargs):
        base_out = super().forward(x, *args, **kwargs)
        return self.lars(base_out)


class LARSLinear8bitLt(bnb.nn.Linear8bitLt):
    def __init__(self, base_layer: bnb.nn.Linear8bitLt, adapter_name, **kwargs):
        input_features = base_layer.in_features
        output_features = base_layer.out_features
        bias = base_layer.bias is not None

        has_fp16_weights = kwargs.pop("has_fp16_weights", base_layer.state.has_fp16_weights)
        threshold = kwargs.pop("threshold", base_layer.state.threshold)
        index = kwargs.pop("index", base_layer.index)
        rank = kwargs.pop("rank", None)
        block_size = kwargs.pop("block_size", 32)

        super().__init__(
            input_features=input_features,
            output_features=output_features,
            bias=bias,
            has_fp16_weights=has_fp16_weights,
            threshold=threshold,
            index=index,
        )

        # Reuse quantized weights/state
        self.weight = base_layer.weight
        self.state = base_layer.state
        self.index = base_layer.index

        if rank is None:
            raise ValueError("LARSLinear8bitLt requires 'rank' kwarg from peft_config.")
        self.lars = LARSLinear(base_layer=self, rank=rank, block_size=block_size)

    def forward(self, x, *args, **kwargs):
        base_out = super().forward(x, *args, **kwargs)
        return self.lars(base_out)
