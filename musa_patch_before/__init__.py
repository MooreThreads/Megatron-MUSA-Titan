import sys
from . import transformer_engine
sys.modules['megatron.core.transformer.custom_layers.transformer_engine'] = transformer_engine