TEColumnParallelLinear = None
TELayerNormColumnParallelLinear = None
TEDotProductAttention = None
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
TENorm = FusedLayerNorm
TERowParallelLinear = None
SplitAlongDim = None
get_cpu_offload_context = None