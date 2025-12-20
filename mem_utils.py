import torch
import os, traceback


class MemMonitor:
    depth = 0
    get_batch_cnt = 0
    before_first_batch_allocated = 99999999.9
    max_token_num = 0
    max_rank_ratio_list = []
    
    def __init__(self, tensor_name="", only_print_allocated_diff=True):
        self.tensor_name = tensor_name
        self.only_print_allocated_diff = only_print_allocated_diff

    def __enter__(self):
        MemMonitor.depth += 1
        self.memory_allocated_before = torch.musa.memory_allocated()
        self.memory_reserved_before = torch.musa.memory_reserved()
        self.max_memory_allocated_before = torch.musa.max_memory_allocated()
        self.max_memory_reserved_before = torch.musa.max_memory_reserved() 
        str_tabs = "    " * (MemMonitor.depth - 1)
        print(str_tabs + self.tensor_name + " start:")
        
        if not self.only_print_allocated_diff:
            print(str_tabs + "  memory allocated before(MB):", self.memory_allocated_before/(1024*1024))
            print(str_tabs + "   memory reserved before(MB):", self.memory_reserved_before/(1024*1024))
            print(str_tabs + "  max memory allocated before(MB):", self.max_memory_allocated_before/(1024*1024))
            print(str_tabs + "  max memory reserved before(MB):", self.max_memory_reserved_before/(1024*1024))
        else:
            print(str_tabs + " memory allocated before(MB):", self.memory_allocated_before/(1024*1024))
            pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        memory_allocated_after = torch.musa.memory_allocated()
        memory_reserved_after = torch.musa.memory_reserved()
        max_memory_allocated_after = torch.musa.max_memory_allocated()
        max_memory_reserved_after = torch.musa.max_memory_reserved()
        str_tabs = "    " * (MemMonitor.depth - 1)
        print(str_tabs + self.tensor_name + " end.")
        if self.only_print_allocated_diff:
            print(str_tabs + "  memory allocated after(MB):", memory_allocated_after/(1024*1024))
            print(str_tabs + "   memory allocated diff(MB):", (memory_allocated_after - self.memory_allocated_before)/(1024*1024))
            pass
        else:
            print(str_tabs + "  memory allocated after(MB):", memory_allocated_after/(1024*1024))
            print(str_tabs + "  memory reserved after(MB):", memory_reserved_after/(1024*1024))
            print(str_tabs + "  max memory allocated after(MB):", max_memory_allocated_after/(1024*1024))
            print(str_tabs + "  max memory reserved after(MB):", max_memory_reserved_after/(1024*1024))
            print(str_tabs + "  memory allocated diff(MB):", (memory_allocated_after - self.memory_allocated_before)/(1024*1024))
            print(str_tabs + "  memory reserved diff(MB):", (memory_reserved_after - self.memory_reserved_before)/(1024*1024))
            print(str_tabs + "  max memory allocated diff(MB):", (max_memory_allocated_after - self.max_memory_allocated_before)/(1024*1024))
            print(str_tabs + "  max memory reserved diff(MB):", (max_memory_reserved_after - self.max_memory_reserved_before)/(1024*1024))
        # print("end " + self.tensor_name + " memory")
        print("")
        MemMonitor.depth -= 1
        
def show_tensor_memory(tensor, tensor_name=""):
    str_tabs = "    " * (MemMonitor.depth - 1)
    if tensor != None:
        # check the data type
        if isinstance(tensor, torch.Tensor):
            nelements = tensor.numel()
            
            nbytes = tensor.element_size() * nelements
            print(str_tabs + tensor_name + " shape:", tensor.shape, f", memory occupied(MB): {nbytes/(1024*1024)}, dtype: {tensor.dtype}, device: {tensor.device}, type: {type(tensor)}, elment_size: {tensor.element_size()}")
        else:
            print(str_tabs + tensor_name + " =", tensor)
    else:
        print(str_tabs + tensor_name + " is None")

        
def _short_stack(filters=("gpt_model", "mlp", "multi_latent_attention", "transformer_layer", "experts", "moe_layer", "moe_utils", "transformer_engine", "moe_utils")):
    frames = traceback.extract_stack(limit=20)
    picked = [f for f in frames if any(k in f.filename for k in filters)]
    # picked = []
    out = ""
    if picked:
        last_k = 2 if len(picked) <= 2 else 3
        # pick last 3 frames
        for p in picked[-last_k:]:
            out += f"{os.path.basename(p.filename)}:{p.lineno}@{p.name} \n"
    else:
        for p in frames:
            out += f"{os.path.basename(p.filename)}:{p.lineno}@{p.name} \n"

    return out

import inspect
import gc

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def retrieve_all_names(var):
    """
    获取所有引用同一对象的变量名，包括全局、局部和多层调用栈中的变量
    """
    all_names = []
    
    # 1. 获取当前调用栈中所有帧的局部变量
    frame = inspect.currentframe()
    while frame:
        if frame.f_back:  # 跳过当前函数帧
            frame_locals = frame.f_back.f_locals
            names = [name for name, val in frame_locals.items() if val is var]
            if names:
                # 添加帧信息以区分不同作用域
                frame_info = frame.f_back.f_code.co_name
                all_names.extend([f"{name}@{frame_info}" for name in names])
        frame = frame.f_back
    
    # 2. 检查全局变量
    frame = inspect.currentframe().f_back
    if frame and hasattr(frame, 'f_globals'):
        global_names = [name for name, val in frame.f_globals.items() 
                       if val is var and not name.startswith('__')]
        all_names.extend([f"{name}@global" for name in global_names])
    
    # 3. 使用gc模块查找所有引用（更彻底但开销较大）
    referrers = gc.get_referrers(var)
    for referrer in referrers:
        if isinstance(referrer, dict):
            ref_names = [name for name, val in referrer.items() 
                        if val is var and isinstance(name, str) and not name.startswith('__')]
            all_names.extend([f"{name}@ref" for name in ref_names])
    
    # 去重并返回
    return list(set(all_names))


import inspect, os

PROJECT_HINTS = ("megatron", "transformer_engine", "xiaoteng")

def guess_var_names_from_stack(t, max_frames=25):
    results = []
    stack = inspect.stack()  # 慎用：会创建对 frame 的强引用
    try:
        for fi in stack[1:max_frames]:
            fn = fi.filename
            # 过滤掉 torch 内部帧，优先匹配你工程相关的帧
            if ("/dist-packages/torch" in fn) and not any(h in fn for h in PROJECT_HINTS):
                continue
            names = []
            for k, v in fi.frame.f_locals.items():
                try:
                    if v is t:
                        names.append(k)
                    elif isinstance(v, (list, tuple)) and any(x is t for x in v):
                        idx = [i for i, x in enumerate(v) if x is t]
                        names.append(f"{k}[{idx}]")
                    elif isinstance(v, dict):
                        ks = [kk for kk, vv in v.items() if vv is t]
                        if ks: names.append(f"{k}{ {tuple(ks)} }")
                except Exception:
                    pass
            if names and names[0]!="t":
                results.append({
                    "file": os.path.basename(fn),
                    "line": fi.lineno,
                    "func": fi.function,
                    "names": names,
                })
        return results
    finally:
        # 释放 frame 引用，避免内存/循环引用问题
        del stack

str_data_ptr_set = set()
def pack_hook(t):
    # import pdb; pdb.set_trace()
    # fn = type(t.grad_fn).__name__ if t.grad_fn is not None else "LeafTensor"

    numel = t.numel()
    elsize = t.element_size()
    saved_meta = {
        "shape": tuple(t.shape),
        "MB": numel * elsize / (1024 * 1024),
        "dtype": str(t.dtype),
        # "autograd_fn": fn,
        # "data_ptr": int(t.data_ptr()),
        # "device": str(t.device),
    }
    str_data_ptr=str(t.data_ptr())
    if str_data_ptr not in str_data_ptr_set:
        saved_meta["ptr"] = "new"
        str_data_ptr_set.add(str_data_ptr)
    else:
        saved_meta["ptr"] = "old"
    #print(saved_meta)
    #print("src:", _short_stack())
    #name_list = retrieve_all_names(t)
    #print("info:", guess_var_names_from_stack(t))
    #print("")
    return t  

def unpack_hook(t):
    return t

import subprocess
import re

    
def get_max_gpu0_to_7_mem_usage():
    result = subprocess.run(["bash", "-c", "mthreads-gmi"], capture_output=True, text=True)
    output = result.stdout

    # 匹配所有 “数字MiB(” 模式
    pattern = re.compile(r"(\d+)MiB\(")
    matches = pattern.findall(output)

    if not matches:
        result = subprocess.run(["bash", "-c", "zjlab-gmi"], capture_output=True, text=True)
        output = result.stdout
        # 匹配所有 “数字MiB(” 模式
        pattern = re.compile(r"(\d+)MiB\(")
        matches = pattern.findall(output)
        
        if not matches:
            print("未找到任何显存使用量的匹配。")
            return None, None

    # 转为浮点数
    usages = [float(x) for x in matches]

    # 如果有至少 8 个匹配，取前 8 个；否则取全部
    if len(usages) >= 8:
        usages = usages[:8]
    else:
        print(f"警告：只找到 {len(usages)} 个显存使用匹配（少于 8 个）。")

    # 找最大值及其索引
    max_usage = max(usages)
    max_idx = usages.index(max_usage)  # GPU ID 假定与索引相同

    return max_idx, max_usage