import os
from ..utils import record_function_decorator
import megatron
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
original_fwdbwd_epoverlap_run = TransformerModelChunkSchedulePlan.run

@record_function_decorator
def fwdbwd_epoverlap_run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        f_microbatch_id_layer_id=None,
        b_microbatch_id_layer_id=None,

    ):
    return original_fwdbwd_epoverlap_run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad,
        pre_forward,
        pre_backward,
        post_forward,
        post_backward,
        f_microbatch_id_layer_id=f_microbatch_id_layer_id,
        b_microbatch_id_layer_id=b_microbatch_id_layer_id,
    )


enable_profiler = int(os.getenv("ENABLE_PROFILER", 0))
if enable_profiler:
    TransformerModelChunkSchedulePlan.run = fwdbwd_epoverlap_run
