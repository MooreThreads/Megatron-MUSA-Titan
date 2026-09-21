import contextlib
import gc
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from typing import Generator, Iterable, List, Optional, Set, Union
import functools
import torch
import json

from megatron.training.global_vars import get_args
from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler
from megatron.core.rerun_state_machine import get_rerun_state_machine


DEEPEP_INVALID_GRAD = float(os.environ.get("DEEPEP_INVALID_GRAD", "0"))
NUM_DEEPEP_DUMP_TENSOR = int(os.environ.get("NUM_DEEPEP_DUMP_TENSOR", "5"))

GLOBAL_DEBUG_DATAS = {
    
}

def get_iteration_id():
    # ARGS = get_args()
    # iteration_id = ARGS.curr_iteration if hasattr(ARGS, 'curr_iteration') else 0
    rerun_state_machine = get_rerun_state_machine()
    iteration_id = getattr(rerun_state_machine, "current_iteration", -1)
    return iteration_id

def get_run_batch_id():
    batch_id = get_fine_grained_offload_handler().current_microbatch_id
    return batch_id

def get_tensor_desc(t: torch.Tensor, tensor_key=None, sync=True):
    # sync = False， 需要放在deepep comm stream上去做
    t = t.detach()
    data =  {
        "shape": str(list(t.shape)),
        "max": t.max().item() if sync else t.max(),
        # "data_ptr": t.data_ptr(),
    }  
    # 统计每个token有多少large grad (>1), 输出 large_grad={token_{i}:num_large_grad}
    # 统计每个token有多少个large grad (>1)
    # 假设 t 的形状是 [num_token, hidden_size]
    if t.dim() == 2 and sync and DEEPEP_INVALID_GRAD > 0:
        # 计算每个token的大梯度数量
        num_large_grad_per_token = (t > DEEPEP_INVALID_GRAD).sum(dim=1, dtype=torch.float32)
        
        # 筛选非零的
        nonzero_mask = num_large_grad_per_token > 0
        if nonzero_mask.any():
            nonzero_indices = torch.where(nonzero_mask)[0]
            nonzero_counts = num_large_grad_per_token[nonzero_mask]
            
            # ========== 汇总统计信息 ==========
            total_tokens = t.size(0)
            tokens_with_large_grad = len(nonzero_indices)
            total_large_grad_elements = int(num_large_grad_per_token.sum().item())
            
            
            # 如果数量 <= 10，直接全部输出，无需排序
            large_grad_dict = {
                "tokens_with_large_grad": tokens_with_large_grad,
                "total_large_grad_elements": total_large_grad_elements,
            }
            if len(nonzero_counts) <= 10:
                large_grad_dict.update({
                    f"token_{idx.item()}": int(count.item())
                    for idx, count in zip(nonzero_indices, nonzero_counts)
                })
            else:
                try:
                    # 数量 > 10，取前10个最大的
                    top_counts, top_local_indices = torch.topk(nonzero_counts, k=10)
                    token_indices = nonzero_indices[top_local_indices]
                    large_grad_dict.update({
                        f"token_{idx.item()}": int(count.item())
                        for idx, count in zip(token_indices, top_counts)
                    })
                except RuntimeError:
                    # 方案C: 降级到 Python 实现
                    print(f"[sherry info] Fall back Topk to cpu, nonzero_counts.shape={nonzero_counts.shape}!!!")
                    counts_list = [(i, int(count.item())) for i, count in enumerate(nonzero_counts)]
                    counts_list.sort(key=lambda x: x[1], reverse=True)
                    top_10 = counts_list[:10]
                    large_grad_dict.update({
                        f"token_{nonzero_indices[idx].item()}": count
                        for idx, count in top_10
                    })
            
            data["large_grad"] = large_grad_dict
    return data

def try_dump_invalid_grad_tensor(iteration_id, batch_id, dump_key, t, max_val):
    save_dir =  get_node_save_dir()
    if len(os.listdir(save_dir)) < NUM_DEEPEP_DUMP_TENSOR and DEEPEP_INVALID_GRAD > 0:
        if isinstance(max_val, torch.Tensor):
            compare_val = max_val.item() if max_val.numel() == 1 else max_val.max().item()
        else:
            compare_val = max_val
        if compare_val > DEEPEP_INVALID_GRAD:
            globak_rank = torch.distributed.get_rank()
            local_rank = globak_rank % 8
            node_id = globak_rank // 8
            save_path = os.path.join(save_dir, f"iteration{iteration_id}_batch{batch_id}_node{node_id}_rank{local_rank}_{dump_key}.max.pt")
            torch.save(t, save_path)
            print(f"[sherry info] dump 大grad的tensor: {save_path}, compare_val={compare_val}")
                    
    
def get_node_save_dir(save_path = None):
    if save_path is None:
        ARGS = get_args()
        save_path = ARGS.tensorboard_dir if ARGS.tensorboard_dir is not None else './debug_datas'
    
    node_save_root = os.path.join(save_path,"debug_hook_deepep_datas")
    os.makedirs(node_save_root, exist_ok=True)

    return node_save_root


def add_global_debug_datas(iteration_id, batch_id, key, data):
    global GLOBAL_DEBUG_DATAS
    
    if iteration_id not in GLOBAL_DEBUG_DATAS:
        GLOBAL_DEBUG_DATAS[iteration_id] = {}
    if batch_id not in GLOBAL_DEBUG_DATAS[iteration_id]:
        GLOBAL_DEBUG_DATAS[iteration_id][batch_id] = {}
    
    if len(data) > 0:
        if key in GLOBAL_DEBUG_DATAS[iteration_id][batch_id]:
            num_key = len(GLOBAL_DEBUG_DATAS[iteration_id][batch_id])
            key = f"{key}_repeat{num_key}"
            # print(f"[sherry info] attention override {key}!!!!!")
        GLOBAL_DEBUG_DATAS[iteration_id][batch_id][key] = data
        return key
    return ""
                        
def dump_global_debug_datas(save_dir=None):
    global GLOBAL_DEBUG_DATAS
    if len(GLOBAL_DEBUG_DATAS) > 0:
        iteration_id = get_iteration_id()
        node_save_dir = get_node_save_dir(save_dir)

        globak_rank = torch.distributed.get_rank()
        local_rank = globak_rank % 8
        node_id = globak_rank // 8
    
        iteration_save_dir = os.path.join(node_save_dir, f"iteration_{iteration_id}", f"node_{node_id}")
        os.makedirs(iteration_save_dir, exist_ok=True)

        save_path = os.path.join(iteration_save_dir, f'rank{local_rank}_deepep_data.json')
        
        count = 1
        while os.path.exists(save_path):
            save_path = os.path.join(iteration_save_dir, f'rank{local_rank}_deepep_data_rerun{count}.json')
            count += 1

        with open(save_path, 'w', encoding='utf-8') as f:
            # 直接用 json.dump，indent=4
            json.dump(GLOBAL_DEBUG_DATAS, f, ensure_ascii=False, indent=4)
            GLOBAL_DEBUG_DATAS = {}
            print(f"global_debug_datas已成功保存到JSON文件:{save_path} 已清空debug_hook_datas")
                
        
# hardcoded BF16 type peak flops for Moore Threads S4000, and S5000 GPU
def get_peak_flops(device_name: str) -> int:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci", "-d", "1ed5:"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12

def record_function_decorator(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        with torch.profiler.record_function(func.__name__):
            return func(*args, **kwargs)

    return new_func

@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""

def musa_patched_get_device_arch_version():
    # """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    # return torch.cuda.get_device_properties(torch.device("cuda:0")).major
    """ Returns MT GPU arch version """
    return torch.musa.get_device_properties(torch.device("musa:0")).major

import megatron.training.utils
megatron.training.utils.get_device_arch_version = musa_patched_get_device_arch_version
# arguments.py does `from megatron.training.utils import get_device_arch_version`,
# binding the original by value at import time; patching the utils module attribute
# above does not update that copy. Rebind the name in arguments too.
import megatron.training.arguments
megatron.training.arguments.get_device_arch_version = musa_patched_get_device_arch_version