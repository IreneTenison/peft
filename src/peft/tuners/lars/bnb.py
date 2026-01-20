# lars/bnb.py
import bitsandbytes as bnb
from .layer import LARSLinear

class LARSLinear4bit(bnb.nn.Linear4bit):
    def __init__(self, base_layer, adapter_name, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lars = LARSLinear(base_layer=base_layer, adapter_name=adapter_name)
    
    def forward(self, x, *args, **kwargs):
        base_out = super().forward(x, *args, **kwargs)
        return self.lars(base_out)  # LARS after quantized matmul

class LARSLinear8bitLt(bnb.nn.Linear8bitLt):
    def __init__(self, base_layer, adapter_name, **kwargs):
        super().__init__(base_layer, **kwargs)
        self.lars = LARSLinear(base_layer=base_layer, adapter_name=adapter_name)
    
    def forward(self, x, *args, **kwargs):
        base_out = super().forward(x, *args, **kwargs)
        return self.lars(base_out)
