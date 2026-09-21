import os
import json
from pprint import pprint
import inspect
import torch
import torch.nn as nn
import hashlib
from megatron.training.global_vars import (
    get_args,
)
from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler

ARGS = get_args()
def get_list_key(key, num):
        return tuple([f"{key}{i}" for i in range(num)])
def get_iteration_id():
    iteration_id = ARGS.curr_iteration if hasattr(ARGS, 'curr_iteration') else 0
    return iteration_id

def get_run_batch_id():
    batch_id = get_fine_grained_offload_handler().current_microbatch_id
    return batch_id

def get_node_save_dir(save_path):
    if save_path is None:
        save_path = ARGS.tensorboard_dir if ARGS.tensorboard_dir is not None else './hook_datas'
    globak_rank = torch.distributed.get_rank()
    
    node_id = globak_rank // 8
    node_save_root = os.path.join(save_path,"debug_hook_manager_datas", f"node_{node_id}")
    os.makedirs(node_save_root, exist_ok=True)

    return node_save_root

def get_rank_log_file(save_path):
    batch_id = get_run_batch_id()
    iteration_id = get_iteration_id()

    node_save_root = get_node_save_dir(save_path)
    batch_save_root = os.path.join(node_save_root, f"iteration_{iteration_id}", f"batch_{batch_id}")
    os.makedirs(batch_save_root, exist_ok=True)

    local_rank = torch.distributed.get_rank()%8
    # rank_log_file = node_save_root/batch_save_root/rank_{local_rank}.log
    rank_log_file = os.path.join(batch_save_root, f'rank_{local_rank}.log')

    return rank_log_file

class TensorChecker:
    @classmethod
    def check_for_large_grads(cls, cur_info:dict, module_or_tensor_name, valid_gradnorm_threshold=500):
        invalid_gradnorm = False
        if 'shape' in cur_info:
            # single tensor 
            v = cur_info
            if isinstance(v, dict) and 'norm' in v.keys() and v['norm'] > valid_gradnorm_threshold:
                print(f"[ERRRO] Find large gradnorm!!!!!! {module_or_tensor_name} {v['norm']}")
                invalid_gradnorm = True
        else:
            for k, v in cur_info.items():
                if isinstance(v, dict) and 'norm' in v.keys() and v['norm'] > valid_gradnorm_threshold:
                    print(f"Find large gradnorm!!!!!! {module_or_tensor_name} {k} {v['norm']}")
                    invalid_gradnorm = True
        return invalid_gradnorm
    
    @classmethod
    def check_inf_nan(cls, cur_info:dict, module_or_tensor_name):
        invalid_data = False
        if 'shape' in cur_info:
            # single tensor
            v = cur_info
            if 'nan_inf_count' in v.keys() and v['nan_inf_count'].item() > 0:
                print(f"[ERRRO] Find invalid data!!!!!! {module_or_tensor_name}")
                invalid_data = True
        else:
            for k, v in cur_info.items():
                if isinstance(v, dict) and ('nan_inf_count' in v.keys() and v['nan_inf_count'].item() > 0):
                    print(f"Find invalid data!!!!!! {module_or_tensor_name}")
                    invalid_data = True
        return invalid_data
    
def hash_tensor(t: torch.Tensor) -> str:
    # x = t.detach().float().contiguous().cpu().numpy().tobytes()
    # return hashlib.md5(x).hexdigest()
    t = t.detach()
    if t.dim() == 0:
        t = t.reshape(1)  # 处理标量 (0维张量)
        
    raw = t.cpu().contiguous().view(torch.uint8).numpy()
    h = hashlib.sha256(raw).hexdigest()
    return h
    

class TensorDumpManager:
    def __init__(self, enable_dump, save_dir, blocking, enable_details):
        self.save_dir = save_dir
        self.enable_dump = enable_dump
        self.blocking = blocking
        self.enable_details = enable_details
    def get_single_tensor_info(self, tensor:torch.Tensor, get_raw_id = False):
        """
            "input0":{
                "mean":0,
                "min": 0,
                "max": 0,
                "std": [],
                "norm": 0,
                "shape": [],
                "dtype": torch.bfloat16,
            },
            "input1":{
            },
            "output":{
            }
        """
        stats = {}
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach()
            stats.update({
                'shape': (list(tensor.shape), str(tensor.dtype)),
                # 'dtype': str(tensor.dtype),
            })
            if tensor.dtype not in [torch.bool, torch.long]:
                is_nan = torch.isnan(tensor).sum()
                is_inf = torch.isinf(tensor).sum()
                stats['is_nan/inf'] = (is_nan, is_inf)
                stats['nan_inf_count'] = is_nan + is_inf
                if self.blocking:
                    if self.enable_details:
                        stats["hash"] = hash_tensor(tensor)
                    
                    nan_count = is_nan.item()
                    if nan_count > 0 and  nan_count < 50:
                        stats["nan_positions"] = torch.isnan(tensor).nonzero().tolist()
            elif tensor.dtype == torch.int64 and self.blocking and get_raw_id and self.enable_details:
                stats['raw_token_id'] = tensor.tolist()
                    
            # 只对浮点类型计算统计值
            if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                try:
                    stats.update({
                        # 'mean': tensor.detach().mean().item(),
                        # 'std': tensor.detach().std().item(),
                        # 'min': tensor.detach().min().item(),
                        # 'max': tensor.detach().max().item(),
                        'norm': tensor.norm()
                    })
                    
                    if os.environ.get("DUMP_DEEPEP", "0") == "1":
                        stats["max"] = tensor.max()
                        
                    if self.enable_details:
                        tensor_fp32 = tensor.float()
                        stats['fp32_norm'] = tensor_fp32.norm()
                        stats['fp32_norm1'] = tensor_fp32.norm(1)
                        stats['fp32_max'] = tensor_fp32.max().item()
                        stats['fp32_min'] = tensor_fp32.min().item()
                        stats['fp32_mean'] = tensor_fp32.mean().item()
                        
                except Exception as e:
                    stats['error'] = f"统计计算失败: {e}"
            else:
                # 对于非浮点类型，只显示基本信息和一些s安全的值
                try:
                    stats.update({
                        # 'min': tensor.min().item() if tensor.numel() > 0 else 0,
                        # 'max': tensor.max().item() if tensor.numel() > 0 else 0,
                    })
                except Exception as e:
                    stats['error'] = f"非浮点tensor统计失败: {e}"
            return stats
        else:
            return str(tensor)
        
    def get_input_output_info(self, module_name, inputs, outputs, stage = "forward", input_keys = None, output_keys = None):
        input_prefix = f"input" if stage in ["forward", "wait_recompute"] else "grad_input"
        output_prefix = f"output" if stage in ["forward", "wait_recompute"] else "grad_output"

        def wrap_tuple(data):
            if data is None:
                return data
            if isinstance(data, torch.Tensor):
                data = (data, )
            # 添加类型检查：如果 data 不是可迭代的，包装成元组
            if not isinstance(data, tuple):
                try:
                    return tuple(data)
                except TypeError:
                    # 如果 data 不可迭代（如 method、int、float等），包装成单元素元组
                    return (data,)
            return data
        
        inp_out_datas = {}
        inputs = wrap_tuple(inputs)
        outputs = wrap_tuple(outputs)
        input_keys = wrap_tuple(input_keys)
        output_keys = wrap_tuple(output_keys)
            
        inputs_key = [f"{input_prefix}{i}" if input_keys is None else input_keys[i] for i in range(len(inputs))]
        outputs_key = [f"{output_prefix}{i}" if output_keys is None else output_keys[i] for i in range(len(outputs))]
        
        all_datas = inputs + outputs
        all_keys = inputs_key + outputs_key
        for d, key in zip(all_datas, all_keys):
            if isinstance(d, torch.Tensor):
                if 'input' in key and 'word_embeddings' in module_name and d.dtype == torch.int64:
                    inp_out_datas[key] = self.get_single_tensor_info(d, get_raw_id=False)
                else:
                    inp_out_datas[key] = self.get_single_tensor_info(d, get_raw_id=False)
            else:
                inp_out_datas[key] = str(d)
        return inp_out_datas
    
    def dump_tensor_list(self, tensor_list, save_dir, module_prefix, name, tensor_keysor_keys, local_rank, prefix):
        if isinstance(tensor_list, torch.Tensor):
            tensor_list = [tensor_list]
        if isinstance(tensor_list, (tuple, list)):
            for i, out in enumerate(tensor_list):
                if tensor_keysor_keys is not None:
                    key = tensor_keysor_keys[i]
                else:
                    key = f"{prefix}{i}"
                save_path = os.path.join(save_dir, f'{module_prefix}.{name}.rank{local_rank}.{key}.pt')
                torch.save(out, save_path)
                print(f"save {prefix}{i} to {save_path} success")
        else:
            print(f'{prefix}{i} is not a tensor, value={str(tensor_list)}')

    def dump_fwd_bwd_inp_and_out_tensors(self, 
                                      inputs, 
                                      outputs, 
                                      module_name, 
                                      stage="forward", 
                                      input_keys = None,
                                      output_keys = None,
                                      input_prefix=None, 
                                      output_prefix=None, 
                                      module_prefix=""):
        if not self.enable_dump:
            return 
        if input_prefix is None:
            input_prefix = (f"input" if stage in ["forward", "wait_recompute"] else "grad_input")
        if output_prefix is None:
            output_prefix = f"output" if stage in ["forward", "wait_recompute"] else "grad_output"

        node_save_dir = get_node_save_dir(self.save_dir)
        local_rank = torch.distributed.get_rank() % 8
        if isinstance(inputs, dict):
            for param_name, inp in inputs.items():
                save_path = os.path.join(node_save_dir, f'{module_prefix}.{module_name}.rank{local_rank}.{param_name}.pt')
                torch.save(inp, save_path)
                print(f"- save {param_name} to {save_path} success")
            return

        self.dump_tensor_list(tensor_list=inputs, 
                              save_dir = node_save_dir,
                              module_prefix=module_prefix, 
                              name=module_name, 
                              tensor_keysor_keys = input_keys,
                              local_rank=local_rank, 
                              prefix=input_prefix)
        
        self.dump_tensor_list(tensor_list=outputs,
                              save_dir = node_save_dir, 
                              module_prefix=module_prefix, 
                              name=module_name, 
                              tensor_keysor_keys = output_keys,
                              local_rank=local_rank, 
                              prefix=output_prefix)
        
    def cache_forward_inp_tensors(self, module, inputs):
        is_grad = torch.is_grad_enabled()
        stage = "forward" if is_grad else "wait_recompute"
        if is_grad and module is not None:
            if not hasattr(module, "debug_fwd_inputs"):
                setattr(module, "debug_fwd_inputs", None)
           # 对元组中的每个张量进行 detach
            detached_inputs = tuple(
                inp.detach() if isinstance(inp, torch.Tensor) else inp 
                for inp in inputs
            )
            module.debug_fwd_inputs = [stage, detached_inputs]
            # weights_dict = self.export_weight_attributes(module)
            # module.debug_fwd_inputs.append(weights_dict)

    def free_forward_inp_tensors(self, module):
        if hasattr(module, "debug_fwd_inputs"):
            module.debug_fwd_inputs = None # free fwd cache

class TensorEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)  

class CompactListEncoder(TensorEncoder):
    def encode(self, obj):
        def _format(obj, indent=0):
            if isinstance(obj, dict):
                if not obj:
                    return '{}'
                items = []
                for k, v in obj.items():
                    key_str = json.dumps(k, ensure_ascii=False)
                    value_str = _format(v, indent + 1)
                    items.append('    ' * (indent + 1) + f'{key_str}: {value_str}')
                return '{\n' + ',\n'.join(items) + '\n' + '    ' * indent + '}'
            elif isinstance(obj, list):
                # 判断是否是最内层列表（元素不是 dict/list/tuple）
                if all(not isinstance(item, (dict, list, tuple)) for item in obj):
                    items = [json.dumps(item, ensure_ascii=False) for item in obj]
                    return '[' + ', '.join(items) + ']'
                else:
                    items = []
                    for item in obj:
                        items.append(_format(item, indent + 1))
                    return '[\n' + ',\n'.join('    ' * (indent + 1) + item for item in items) + '\n' + '    ' * indent + ']'
            elif isinstance(obj, torch.Tensor):
                return json.dumps(obj.item() if obj.numel() == 1 else obj.tolist())
            else:
                return json.dumps(obj)
        return _format(obj, 0)
    

class DebugHookManager:
    """管理调试 hooks，支持多个 batch 的跟踪"""
    debug_hook_datas = {
    }
    record_data = {}
    def __init__(self, 
                 enable_log_online = False, 
                 enable_log_to_file = False,
                 enable_hook_fn = None,
                 enforce_register_hook_check_fn = None,
                 enable_forward_hook = True, 
                 enable_pre_backward_hook = True,
                 enable_backward_hook = True,
                 enable_cache_fwd = False,
                 enable_details = False,
                 valid_gradnorm_threshold = 100,
                 max_batches = None,
                 blocking = False, # 阻塞，实时判定inf/nan, large gradnorm
                 save_dir = None,
                 enable_dump = False,
                 exit_when_invalid = False):
        self.hooks = []
        self.enable_log_online = enable_log_online
        self.enable_log_to_file = enable_log_to_file
        self.enable_hook_fn = enable_hook_fn
        self.enforce_register_hook_check_fn = enforce_register_hook_check_fn
        
        self.rank = int(os.environ.get('RANK', 0))
        
        is_hook_rank = enable_hook_fn(self.rank) if enable_hook_fn else True
                
        self.enable_forward_hook = enable_forward_hook and is_hook_rank
        self.enable_backward_hook = enable_backward_hook and is_hook_rank
        self.enable_cache_fwd = enable_cache_fwd
        self.enable_details = enable_details
        self.enable_pre_backward_hook = enable_pre_backward_hook and is_hook_rank
        self.valid_gradnorm_threshold = valid_gradnorm_threshold
        self.max_batches = max_batches
        self.blocking = blocking
        
        self.save_dir = save_dir
        self.enable_dump = enable_dump
        self.exit_when_invalid = exit_when_invalid
        
        
        env_blocking = os.environ.get("BLOCKING_HOOK", None)
        env_hook_decision =  os.environ.get("HOOK_DECISION", None)
        
        if env_blocking is not None:
            print(f"[Sherry HOOK INFO] Override defult blocking setting!!!!!")
            self.blocking = env_blocking
        if env_hook_decision:
            print(f"[Sherry HOOK INFO] Override defult fwd/pre_bwd/bwd setting!!!!!")
            fwd_hook, pre_bwd_hook, bwd_hook = [i == "1" for i in env_hook_decision.split(',')]
            self.enable_forward_hook = self.enable_forward_hook and fwd_hook
            self.enable_pre_backward_hook = self.enable_pre_backward_hook and pre_bwd_hook
            self.enable_backward_hook = self.enable_backward_hook and bwd_hook
        
        if not self.blocking: # 非阻塞模式，不在线打印
            print(f"[Sherry HOOK INFO] Close online logging!!!!!")
            self.enable_log_online = False
            self.enable_log_to_file = False
        
        
        self.tensor_dump_manager = TensorDumpManager(enable_dump= self.enable_dump, 
                                                     save_dir = self.save_dir, 
                                                     blocking=self.blocking, 
                                                     enable_details=self.enable_details)
        
        print(f"[Sherry HOOK INFO]  rank={self.rank}/local_rank={self.rank%8}, blocking={self.blocking}, enable_forward_hook={self.enable_forward_hook}, enable_pre_backward_hook={self.enable_pre_backward_hook}, enable_backward_hook={self.enable_backward_hook}, enable_log_online={self.enable_log_online}, enable_log_to_file={self.enable_log_to_file}, enable_cache_fwd={self.enable_cache_fwd}, max_batches={self.max_batches}, save_dir={self.save_dir}, enable_dump={self.enable_dump}, exit_when_invalid={self.exit_when_invalid}")
        
        self.find_blocking_invalid = False
        self.cur_batch_id = -1
        
    def get_key(self, iteration_id, batch_id, module_name):
        iteration_key = f'iteration_{iteration_id}'
        batch_key = f'batch_{batch_id}'
        datas = DebugHookManager.debug_hook_datas
        if iteration_key not in datas:
            datas[iteration_key] = {}
        
        if self.max_batches and batch_id >= self.max_batches:
            return None, None, None
        
        if batch_key not in datas[iteration_key]:
            datas[iteration_key][batch_key] = {}
            
        if module_name not in datas[iteration_key][batch_key]:
            datas[iteration_key][batch_key][module_name] = {}
        
        run_id = sum(['run' in key for key in datas[iteration_key][batch_key][module_name].keys()])
        run_id_key = f'run_{run_id}'
        if run_id_key not in datas[iteration_key][batch_key][module_name]:
            datas[iteration_key][batch_key][module_name][run_id_key] = {}
        self.cur_batch_id = batch_id
        return iteration_key, batch_key, run_id_key
     
    def is_leaf_module(self, module:torch.nn.Module):
        if len(list(module.children())) == 0:
            return True
        else:
            return False
    
    def _get_main_grad_attr(self, param: torch.nn.Parameter, use_megatron_fsdp: bool = False):
        if hasattr(param, "main_grad"):
            return "main_grad"
        return "grad"
    
    def get_weights(self, module, return_grad=False):
        if module is None:
            return tuple()
        all_weights = []
        if hasattr(module, "weight"):
            all_weights = [module.weight]
        if hasattr(module, "weight0"):
            num_experts = module.num_gemms
            all_weights = [getattr(module, f"weight{i}") for i in range(num_experts)]
        
        if return_grad:
            all_grads = []
            grads_key = []
            for param in all_weights:
                grad_attr = self._get_main_grad_attr(param)
                if hasattr(param, grad_attr):
                    all_grads.append(getattr(param, grad_attr))
                    grads_key.append(f"{grad_attr}_{len(grads_key)}")
                
            return tuple(all_weights), tuple(all_grads), tuple(grads_key)
        return tuple(all_weights)
    
    def record_and_logging(self, inputs, outputs, stage, input_keys, output_keys, module_or_tensor_name, enable_log_online, enable_log_to_file):
        batch_id = get_run_batch_id()
        iteration_id = get_iteration_id()

        # save inp/out info
        iteration_key, batch_key, run_id_key = self.get_key(iteration_id, batch_id, module_or_tensor_name)
        if iteration_key and batch_key and run_id_key:
            info = self.tensor_dump_manager.get_input_output_info(module_name=module_or_tensor_name,
                                                                  inputs=inputs, 
                                                                  outputs=outputs, 
                                                                  stage=stage, 
                                                                  input_keys=input_keys, 
                                                                  output_keys=output_keys)
            DebugHookManager.debug_hook_datas[iteration_key][batch_key][module_or_tensor_name][run_id_key][stage] = info
            if enable_log_online:
                print(f"[{stage}][rank{self.rank}] batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name} hooks info!!!!")
                # pprint(info)
            if enable_log_to_file:
                with open(get_rank_log_file(self.save_dir), "a") as f:
                    print(f"[{stage}] batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name} hooks info!!!!", file=f)
                    # pprint(info, f)
            return info
        else:
            return {}

    def check_and_dump(self, inputs, outputs, stage, valid_gradnorm_threshold, module_or_tensor_name, enable_log_online, enable_log_to_file, input_keys=None, output_keys=None):
        batch_id = get_run_batch_id()
        iteration_id = get_iteration_id()
        info = self.record_and_logging(inputs=inputs,
                                       outputs=outputs,
                                       stage=stage,
                                       input_keys=input_keys,
                                       output_keys=output_keys,
                                       module_or_tensor_name=module_or_tensor_name,
                                       enable_log_online=enable_log_online,
                                       enable_log_to_file=enable_log_to_file)
        if self.blocking:
            if TensorChecker.check_inf_nan(info, module_or_tensor_name):
                print(f"[ERROR] Find invalid data nan or inf in {stage} at batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name}, try save and exit!!!!!!!!!")
                # dump fwd/bwd grad_input and grad_output
                self.tensor_dump_manager.dump_fwd_bwd_inp_and_out_tensors(inputs=inputs, 
                                                                        outputs = outputs, 
                                                                        module_name = module_or_tensor_name, 
                                                                        stage=stage, 
                                                                        input_keys = input_keys,
                                                                        output_keys = output_keys,
                                                                        module_prefix=f"iteration{iteration_id}_batch{batch_id}")
                self.find_blocking_invalid = True
                self.max_batches = self.cur_batch_id + 1
                if self.exit_when_invalid:
                    self.save()
                    exit()
            if "backward" in stage:
                if TensorChecker.check_for_large_grads(info, module_or_tensor_name, valid_gradnorm_threshold):
                    print(f"[ERROR] Find invalid gradnorm in {stage} at batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name}, try save and exit!!!!!!!!!")
                    # dump bwd grad_input and grad_output
                    self.tensor_dump_manager.dump_fwd_bwd_inp_and_out_tensors(inputs=inputs, 
                                                                                outputs=outputs, 
                                                                                module_name=module_or_tensor_name, 
                                                                                    stage=stage, 
                                                                                    input_keys = input_keys,
                                                                                    output_keys = output_keys,
                                                                                    module_prefix=f"iteration{iteration_id}_batch{batch_id}")
                    self.find_blocking_invalid = True
                    self.max_batches = self.cur_batch_id + 1
                    if self.exit_when_invalid:
                        self.save()
                        exit()
    
              
    #################### Tensor fwd/bwd hook ####################
    def create_tensor_backward_hook(self, 
                             tensor_name,
                             valid_gradnorm_threshold=200):
        def backward_hook(grad):
            stage = "backward"
            self.check_and_dump(inputs=(grad,),
                                outputs=(),
                                stage=stage,
                                valid_gradnorm_threshold=valid_gradnorm_threshold,
                                module_or_tensor_name=tensor_name,
                                enable_log_online=self.enable_log_online,
                                enable_log_to_file = self.enable_log_to_file)
        return backward_hook

    def register_tensorwise_backward_hook(self, tensor:torch.Tensor, tensor_name):
        if self.enable_backward_hook:
            tensor.requires_grad_(True)
            tensor.register_hook(self.create_tensor_backward_hook(tensor_name))
        
    

    @classmethod
    def run_forward_hooks_by_handle(self, inputs, outputs, 
                                    module_name,
                                     enable_log_online=False,
                                     enable_log_to_file=False):
        DebugHookManager().run_forward(
            module=None,
            inputs=inputs,
            outputs=outputs,
            module_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file=enable_log_to_file
        )
        
    
    #################### Module fwd/bwd hook ####################
    def record_info(self, key, data):
        batch_id = get_run_batch_id()
        iteration_id = get_iteration_id()
        repeat = 1
        base_key = f"iter{iteration_id}_batch{batch_id}_{key}"
        if base_key in DebugHookManager.record_data:
            base_key = f"iter{iteration_id}_batch{batch_id}_{key}_{repeat}"
            repeat += 1
        DebugHookManager.record_data[base_key] = data
        
    def run_forward(self, module:nn.Module, inputs, outputs, module_name = None,
                            enable_log_online=False,
                            enable_log_to_file=False
                            ):
        is_grad = torch.is_grad_enabled()
        # when enable_recompute:
        #   "wait_recompute" means first forward
        #   "forward" means recompute
        stage = "forward" if is_grad else "wait_recompute"
        
        all_weight = self.get_weights(module)
        input_keys = get_list_key("input", len(inputs)) + \
                     get_list_key("weight", len(all_weight))
        self.check_and_dump(
            inputs=inputs + all_weight,
            outputs=outputs,
            stage=stage,
            valid_gradnorm_threshold=None,
            module_or_tensor_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file = enable_log_to_file,
            input_keys = input_keys,
        )
        
        # if stage == "forward" and module is not None and self.enable_backward_hook and self.enable_cache_fwd:
        #     print(f"[sherry HOOK INFO] cache forward!")
        #     self.tensor_dump_manager.cache_forward_inp_tensors(module, inputs)
    
    def run_backward(self, module:nn.Module, stage, grad_input=None, grad_output=None, valid_gradnorm_threshold=0, module_name=None,
                            enable_log_online=False,
                            enable_log_to_file=False):
        
        if stage == "backward":
            fwd_act=tuple()
            if hasattr(module, "debug_fwd_inputs") and module.debug_fwd_inputs is not None and len(module.debug_fwd_inputs) > 1:
                fwd_act = module.debug_fwd_inputs[1]
            weights, grads, grad_keys = self.get_weights(module, return_grad=True)
            inputs = grad_output + fwd_act + weights
            outputs = grad_input + grads
            
            input_keys = get_list_key("dgrad_input", len(grad_output)) + \
                        get_list_key("activation", len(fwd_act)) + \
                        get_list_key("weight", len(weights))
            output_keys = get_list_key("dgrad_output", len(grad_input)) + grad_keys
            assert len(input_keys) == len(inputs), f"len(input_keys)={len(input_keys)}, len(inputs)={len(inputs)}， input_keys={input_keys}"
            assert len(output_keys) == len(outputs), f"len(output_keys)={len(output_keys)}, outputs={len(outputs)}， output_keys={output_keys}"
        elif stage == "pre_backward":
            # 只观察grad_output
            inputs = grad_output
            input_keys = get_list_key("pre_dgrad_input", len(grad_output))
            outputs = tuple()
            output_keys = tuple()
        else:
            raise NotImplementedError()
            
        self.check_and_dump(
            inputs=inputs, # dgrad_input + activation + weights
            outputs=outputs, # dgrad_output + weights.grad
            stage=stage,
            valid_gradnorm_threshold=valid_gradnorm_threshold,
            module_or_tensor_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file=enable_log_to_file,
            input_keys=input_keys,
            output_keys=output_keys
        )
       
        self.tensor_dump_manager.free_forward_inp_tensors(module)
    
    def create_module_forward_hook(self, module_name: str):
        def forward_hook(module, inputs, outputs):
            enfore_hook = self.enforce_register_hook_check_fn(module, module_name) if self.enforce_register_hook_check_fn else False
            if not self.is_leaf_module(module) and not enfore_hook:
                return
            self.run_forward(
                module=module,
                inputs=inputs,
                outputs=outputs,
                module_name=module_name,
                enable_log_online=self.enable_log_online,
                enable_log_to_file=self.enable_log_to_file
            )
        return forward_hook
    
    def create_module_backward_hook(self, module_name: str):
        def backward_hook(module, grad_input, grad_output):
            enfore_hook = self.enforce_register_hook_check_fn(module, module_name) if self.enforce_register_hook_check_fn else False
            if not self.is_leaf_module(module) and not enfore_hook:
                return
            self.run_backward(
                module=module,
                stage="backward",
                grad_input=grad_input,
                grad_output=grad_output,
                valid_gradnorm_threshold=self.valid_gradnorm_threshold,
                module_name=module_name,
                enable_log_online=self.enable_log_online,
                enable_log_to_file=self.enable_log_to_file
            )
        return backward_hook
    
    def create_module_pre_backward_hook(self, module_name: str):
        def pre_backward_hook(module, grad_output):
            enfore_hook = self.enforce_register_hook_check_fn(module, module_name) if self.enforce_register_hook_check_fn else False
            if not self.is_leaf_module(module) and not enfore_hook:
                return
            self.run_backward(
                module=module,
                stage="pre_backward",
                grad_input=None,
                grad_output=grad_output,
                valid_gradnorm_threshold=self.valid_gradnorm_threshold,
                module_name=module_name,
                enable_log_online=self.enable_log_online,
                enable_log_to_file=self.enable_log_to_file
            )
        return pre_backward_hook
    
    def register_modulewise_hooks(self, model: nn.Module, model_prefix = ""):
        """为模型注册 hooks（只需要调用一次）"""
        self.remove_existing_hooks()  # 先清理现有的 hooks
        
        for name, module in model.named_modules():
            if name == "":  # 跳过根模块（空名字）
                continue
            
            name = model_prefix + name
            # 注册 forward hook
            if self.enable_forward_hook:
                forward_handle = module.register_forward_hook(self.create_module_forward_hook(name))
                self.hooks.append(forward_handle)
            # 注册 backward hook
            try:
                if self.enable_backward_hook:
                    backward_handle = module.register_full_backward_hook(self.create_module_backward_hook(name))
                    self.hooks.append(backward_handle)
                if self.enable_pre_backward_hook:
                    pre_backward_handle = module.register_full_backward_pre_hook(self.create_module_pre_backward_hook(name))
                    self.hooks.append(pre_backward_handle)
            except Exception as e:
                print(f"警告: 无法为 {name} 注册 backward hook: {e}")
                self.hooks.append(forward_handle)
        
        print(f"已为模型注册 {len(self.hooks)} 个 hooks")
    
    def remove_existing_hooks(self):
        """移除所有 hooks（完成后调用）"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("已清理所有 hooks")

    def save(self):
        def tensor_to_python(obj):
            """递归转换所有 tensor 为 Python 类型"""
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: tensor_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                if not isinstance(obj[0], (list, tuple)):
                    return str([tensor_to_python(v) for v in obj])
                return [tensor_to_python(v) for v in obj]
            return obj
        print(f"[Sherry HOOK INFO] Try dump debug_hook_datas to {self.save_dir}")
        
        if len(DebugHookManager.debug_hook_datas) > 0:
            iteration_id = get_iteration_id()
            node_save_dir = get_node_save_dir(self.save_dir)

            local_rank = torch.distributed.get_rank() % 8
            iteration_save_dir = os.path.join(node_save_dir, f"iteration_{iteration_id}")
            os.makedirs(iteration_save_dir, exist_ok=True)

            save_path = os.path.join(iteration_save_dir, f'rank{local_rank}_hook_data.json')
            
            
            count = 1
            while os.path.exists(save_path):
                save_path = os.path.join(iteration_save_dir, f'rank{local_rank}_hook_data_rerun{count}.json')
                count += 1
            
            with open(save_path, 'w', encoding='utf-8') as f:
                # 先转换所有 tensor
                data = tensor_to_python(DebugHookManager.debug_hook_datas)
                # 直接用 json.dump，indent=4
                json.dump(data, f, ensure_ascii=False, indent=4)
                                 
                DebugHookManager.debug_hook_datas = {}
                DebugHookManager.record_data = {}
                print(f"debug_hook_datas已成功保存到JSON文件:{save_path} 已清空debug_hook_datas")
        
        if self.find_blocking_invalid:
            print(f"训练中找到异常数据，退出")
            exit()
global_debug_hook_manager:DebugHookManager = None


class AutoWrappedModule(nn.Module):
    def __init__(self, fn, name=None, strict=False, **preset_kwargs):
        super().__init__()
        self.fn = fn
        self.name = name or getattr(fn, "__name__", self.__class__.__name__)
        self.strict = strict
        self.preset_kwargs = preset_kwargs
        self.sig = inspect.signature(fn)
        self.accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self.sig.parameters.values()
        )

    def _filter_kwargs(self, kwargs):
        if self.accepts_var_kwargs:
            return kwargs
        accepted = {}
        for k, v in kwargs.items():
            if k in self.sig.parameters:
                accepted[k] = v
            elif self.strict:
                raise TypeError(f"{self.name} got unexpected kwarg: {k}")
        return accepted

    def forward(self, *args, **kwargs):
        merged = {**self.preset_kwargs, **kwargs}
        filtered = self._filter_kwargs(merged)
        return self.fn(*args, **filtered)

    def extra_repr(self):
        return f"name={self.name}, strict={self.strict}, preset_keys={list(self.preset_kwargs.keys())}"

def auto_wrap_module(fn, name=None, strict=False, **preset_kwargs):
    return AutoWrappedModule(fn, name=name, strict=strict, **preset_kwargs)


