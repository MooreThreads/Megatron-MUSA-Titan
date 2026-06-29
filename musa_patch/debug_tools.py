import os
import json
from pprint import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from megatron.training.global_vars import (
    get_args,
    get_timers,
    get_tensorboard_writer,
    get_wandb_writer,
    get_one_logger
)
from transformer_engine.pytorch.cpu_offload import get_fine_grained_offload_handler

ARGS = get_args()

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
    def check_for_large_grads(cls, cur_info:dict, module_or_tensor_name, valid_gradnorm_threads=500):
        invalid_gradnorm = False
        if 'shape' in cur_info:
            # single tensor 
            v = cur_info
            if isinstance(v, dict) and 'norm' in v.keys() and v['norm'] > valid_gradnorm_threads:
                print(f"Find large gradnorm!!!!!! {module_or_tensor_name} {v['norm']}")
                invalid_gradnorm = True
        else:
            for k, v in cur_info.items():
                if isinstance(v, dict) and 'norm' in v.keys() and v['norm'] > valid_gradnorm_threads:
                    print(f"Find large gradnorm!!!!!! {module_or_tensor_name} {k} {v['norm']}")
                    invalid_gradnorm = True
        return invalid_gradnorm
    
    @classmethod
    def check_inf_nan(cls, cur_info:dict, module_or_tensor_name):
        invalid_data = False
        if 'shape' in cur_info:
            # single tensor
            v = cur_info
            if ('is_nan' in v.keys() and v['is_nan']  or 
                'is_inf' in v.keys() and v['is_inf']):
                print(f"Find invalid data!!!!!! {module_or_tensor_name} {k} is_nan={v['is_nan']}, is_inf={v['is_inf']}")
                invalid_data = True
        else:
            for k, v in cur_info.items():
                if isinstance(v, dict) and ('is_nan' in v.keys() and v['is_nan']  or 
                                            'is_inf' in v.keys() and v['is_inf']):
                    print(f"Find invalid data!!!!!! {module_or_tensor_name} {k} is_nan={v['is_nan']}, is_inf={v['is_inf']}")
                    invalid_data = True
        return invalid_data
    

class TensorDumpManager:
    def __init__(self, enable_dump, save_dir):
        self.save_dir = save_dir
        self.enable_dump = enable_dump
    def get_single_tensor_info(self, tensor:torch.Tensor):
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
            stats.update({
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
            })
            if tensor.dtype not in [torch.bool, torch.long]:
                stats['is_nan'] = torch.isnan(tensor).any().item()
                stats['is_inf'] = torch.isinf(tensor).any().item()
            # 只对浮点类型计算统计值
            if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                try:
                    stats.update({
                        'mean': tensor.detach().mean().item(),
                        'std': tensor.detach().std().item(),
                        'min': tensor.detach().min().item(),
                        'max': tensor.detach().max().item(),
                        'norm': tensor.detach().norm().item(),
                    })
                except Exception as e:
                    stats['error'] = f"统计计算失败: {e}"
            else:
                # 对于非浮点类型，只显示基本信息和一些安全的值
                try:
                    stats.update({
                        'min': tensor.min().item() if tensor.numel() > 0 else 0,
                        'max': tensor.max().item() if tensor.numel() > 0 else 0,
                    })
                except Exception as e:
                    stats['error'] = f"非浮点tensor统计失败: {e}"
            return stats
        else:
            return str(tensor)
        
    def get_input_output_info(self, inputs, outputs, stage = "forward"):
        input_prefix = f"input" if stage in ["forward", "recompute"] else "grad_input"
        output_prefix = f"output" if stage in ["forward", "recompute"] else "grad_output"

        inp_out_datas = {}
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        if isinstance(inputs, (tuple, list)):
            for i, inp in enumerate(inputs):
                inp_out_datas[f"{input_prefix}{i}"] = self.get_single_tensor_info(inp)
        else:
            inp_out_datas[f"{input_prefix}"] = str(inputs)

        if isinstance(outputs, (tuple, list)):
            for i, out in enumerate(outputs):
                inp_out_datas[f"{output_prefix}{i}"] = self.get_single_tensor_info(out)
        else:
            inp_out_datas[f"{output_prefix}"] = str(outputs)
        return inp_out_datas
    
    def dump_tensor_list(self, tensor_list, save_dir, module_prefix, name, local_rank, prefix):
        if isinstance(tensor_list, torch.Tensor):
            tensor_list = [tensor_list]
        if isinstance(tensor_list, (tuple, list)):
            for i, out in enumerate(tensor_list):
                save_path = os.path.join(save_dir, f'{module_prefix}.{name}.rank{local_rank}.{prefix}{i}.pt')
                torch.save(out, save_path)
                print(f"save {prefix}{i} to {save_path} success")
        else:
            print(f'{prefix}{i} is not a tensor, value={str(tensor_list)}')

    def dump_fwd_bwd_inp_and_out_tensors(self, 
                                      inputs, 
                                      outputs, 
                                      name, 
                                      stage="forward", 
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
                save_path = os.path.join(node_save_dir, f'{module_prefix}.{name}.rank{local_rank}.{param_name}.pt')
                torch.save(inp, save_path)
                print(f"- save {param_name} to {save_path} success")
            return

        self.dump_tensor_list(tensor_list=inputs, 
                              save_dir = node_save_dir,
                              module_prefix=module_prefix, 
                              name=name, 
                              local_rank=local_rank, 
                              prefix=input_prefix)
        
        self.dump_tensor_list(tensor_list=outputs,
                              save_dir = node_save_dir, 
                              module_prefix=module_prefix, 
                              name=name, 
                              local_rank=local_rank, 
                              prefix=output_prefix)
        
    def export_weight_attributes(self, module):
        """
        导出所有以weight或weight{number}开头的属性
        
        Returns:
            dict: 属性名到属性值的映射
        """
        weight_dict = {}
        for attr_name in dir(module):
            # 匹配 weight 或 weight后跟数字的模式
            if attr_name.startswith('weight'):
                # 检查是否是 weight 后跟数字的模式
                remaining = attr_name[6:]  # 去掉 'weight'
                
                # 匹配: weight, weight1, weight2, weight123 等
                if remaining == '' or remaining.isdigit():
                    try:
                        attr_value = getattr(module, attr_name)
                        weight_dict[attr_name] = attr_value
                    except Exception as e:
                        print(f"无法获取属性 {attr_name}: {e}")
        
        return weight_dict        

    def cache_forward_inp_tensors(self, module, input):
        is_grad = torch.is_grad_enabled()
        stage = "forward" if is_grad else "wait_recompute"
        if is_grad and module is not None:
            if not hasattr(module, "debug_fwd_inputs"):
                setattr(module, "debug_fwd_inputs", None)
            module.debug_fwd_inputs = [stage, input]
            weights_dict = self.export_weight_attributes(module)
            module.debug_fwd_inputs.append(weights_dict)

    def free_forward_inp_tensors(self, module):
        if hasattr(module, "debug_fwd_inputs"):
            module.debug_fwd_inputs = None # free fwd cache
        
    def dump_forward_inp_and_weights(self, module, module_name):
        if hasattr(module, "debug_fwd_inputs") and module.debug_fwd_inputs is not None and len(module.debug_fwd_inputs) >=2:
            # save forward inputs for debug
            batch_id = get_run_batch_id()
            iteration_id = get_iteration_id()
            fwd_stage = module.debug_fwd_inputs[0]
            # dump inputs
            self.dump_fwd_bwd_inp_and_out_tensors(module.debug_fwd_inputs[1], 
                                               (), 
                                               module_name, 
                                                stage=fwd_stage, 
                                                input_prefix="fwd_input",
                                                module_prefix=f"iteration{iteration_id}_batch{batch_id}")
            if len(module.debug_fwd_inputs) > 2:
                # dump weights
                self.dump_fwd_bwd_inp_and_out_tensors(module.debug_fwd_inputs[2], (), module_name, 
                                                    stage=fwd_stage, 
                                                    module_prefix=f"iteration{iteration_id}_batch{batch_id}")

   
class DebugHookManager:
    """管理调试 hooks，支持多个 batch 的跟踪"""
    debug_hook_datas = {
    }
    def __init__(self, 
                 enable_log_online = True, 
                 enable_log_to_file = True,
                 enable_forward_hook = True, 
                 enable_backward_hook = True,
                 valid_gradnorm_threads = None,
                 save_dir = None,
                 enable_dump = True):
        self.hooks = []
        self.forward_grad_states = {}
        self.enable_log_online = enable_log_online
        self.enable_log_to_file = enable_log_to_file
        self.enable_forward_hook = enable_forward_hook
        self.enable_backward_hook = enable_backward_hook
        self.valid_gradnorm_threads = valid_gradnorm_threads
        self.save_dir = save_dir
        self.tensor_dump_manager = TensorDumpManager(enable_dump, self.save_dir)

    def get_key(self, iteration_id, batch_id, module_name):
        iteration_key = f'iteration_{iteration_id}'
        batch_key = f'batch_{batch_id}'
        datas = DebugHookManager.debug_hook_datas
        if iteration_key not in datas:
            datas[iteration_key] = {}

        if batch_key not in datas[iteration_key]:
            datas[iteration_key][batch_key] = {}

        if module_name not in datas[iteration_key][batch_key]:
            datas[iteration_key][batch_key][module_name] = {}

        run_id = sum(['run' in key for key in datas[iteration_key][batch_key][module_name].keys()])
        run_id_key = f'run_{run_id}'
        if run_id_key not in datas[iteration_key][batch_key][module_name]:
            datas[iteration_key][batch_key][module_name][run_id_key] = {}
        return iteration_key, batch_key, run_id_key
     
    def is_leaf_module(self, module:torch.nn.Module):
        if len(list(module.children())) == 0:
            return True
        else:
            return False
         
    def record_and_logging(self, input, output, stage, module_or_tensor_name, enable_log_online, enable_log_to_file):
        batch_id = get_run_batch_id()
        iteration_id = get_iteration_id()

        # save inp/out info
        info = self.tensor_dump_manager.get_input_output_info(input, output)
        iteration_key, batch_key, run_id_key = self.get_key(iteration_id, batch_id, module_or_tensor_name)
        DebugHookManager.debug_hook_datas[iteration_key][batch_key][module_or_tensor_name][run_id_key][stage] = info

        if enable_log_online:
            print(f"[{stage}] batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name} hooks info!!!!")
            pprint(info)
        if enable_log_to_file:
            with open(get_rank_log_file(self.save_dir), "a") as f:
                print(f"[{stage}] batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name} hooks info!!!!", file=f)
                pprint(info, f)
        return info
    
    def check_and_dump(self, module, input, output, stage, valid_gradnorm_threads, module_or_tensor_name, enable_log_online, enable_log_to_file):
        batch_id = get_run_batch_id()
        iteration_id = get_iteration_id()
        info = self.record_and_logging(input=input,
                                       output=output,
                                       stage=stage,
                                       module_or_tensor_name=module_or_tensor_name,
                                       enable_log_online=enable_log_online,
                                       enable_log_to_file=enable_log_to_file)
        if TensorChecker.check_inf_nan(info, module_or_tensor_name):
            print(f"[ERROR] Find invalid data nan or inf in {stage} at batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name}, try save and exit!!!!!!!!!")
            if not enable_log_online:
                pprint(info)
            # dump fwd/bwd grad_input and grad_output
            self.tensor_dump_manager.dump_fwd_bwd_inp_and_out_tensors(input, 
                                                                   output, 
                                                                   module_or_tensor_name, 
                                                                   stage=stage, 
                                                                   module_prefix=f"iteration{iteration_id}_batch{batch_id}")
            exit()
        if stage == "backward":
            if TensorChecker.check_for_large_grads(info, module_or_tensor_name, valid_gradnorm_threads):
                
                print(f"[ERROR] Find invalid gradnorm in {stage} at batch_id={batch_id}, iteration_id={iteration_id}, {module_or_tensor_name}, try save and exit!!!!!!!!!")
                if not enable_log_online:
                    pprint(info)
                # dump bwd grad_input and grad_output
                self.tensor_dump_manager.dump_fwd_bwd_inp_and_out_tensors(input, output, module_or_tensor_name, 
                                                                stage=stage, 
                                                                module_prefix=f"iteration{iteration_id}_batch{batch_id}")
                # dump fwd input and weights
                if module is not None:
                    self.tensor_dump_manager.dump_forward_inp_and_weights(module, module_or_tensor_name)
                exit()

    #################### Tensor bwd hook ####################
    def create_tensor_backward_hook(self, 
                             tensor_name,
                             valid_gradnorm_threads=200,
                             enable_log_online=False,
                             enable_log_to_file=False):
        def backward_hook(grad):
            stage = "backward"
            self.check_and_dump(module=None,
                                input=(grad,),
                                output=(),
                                stage=stage,
                                valid_gradnorm_threads=valid_gradnorm_threads,
                                module_or_tensor_name=tensor_name,
                                enable_log_online=enable_log_online,
                                enable_log_to_file = enable_log_to_file)
        return backward_hook

    def register_tensorwise_backward_hook(self, tensor:torch.Tensor, tensor_name):
        tensor.requires_grad_(True)
        tensor.register_hook(DebugHookManager.create_tensor_backward_hook(tensor_name, enable_log_online=True))
    
   
    #################### Module bwd hook ####################
    def run_forward(self, module:nn.Module, input, output, module_name = None,
                            enable_log_online=False,
                            enable_log_to_file=False
                            ):
        is_grad = torch.is_grad_enabled()
        # when enable_recompute:
        #   "wait_recompute" means first forward
        #   "forward" means recompute
        stage = "forward" if is_grad else "wait_recompute"

        self.check_and_dump(
            module=module,
            input=input,
            output=output,
            stage=stage,
            valid_gradnorm_threads=None,
            module_or_tensor_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file = enable_log_to_file,
        )
        
        if stage == "forward" and module is not None:
            self.tensor_dump_manager.cache_forward_inp_tensors(module, input)
    
    def run_backward(self, module:nn.Module, grad_input, grad_output, valid_gradnorm_threads, module_name=None,
                            enable_log_online=False,
                            enable_log_to_file=False):
        
        stage = "backward"
        self.check_and_dump(
            module=module,
            input=grad_input,
            output=grad_output,
            stage=stage,
            valid_gradnorm_threads=valid_gradnorm_threads,
            module_or_tensor_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file=enable_log_to_file
        )
       
        self.tensor_dump_manager.free_forward_inp_tensors(module)
    
    def create_module_forward_hook(self, module_name: str):
        def forward_hook(module, input, output):
            if not self.is_leaf_module(module):
                return
            if not self.enable_forward_hook:
                return
            self.run_forward(
                module=module,
                input=input,
                output=output,
                module_name=module_name,
                enable_log_online=self.enable_log_online,
                enable_log_to_file=self.enable_log_to_file
            )
        return forward_hook
    
    def create_module_backward_hook(self, module_name: str):
        def backward_hook(module, grad_input, grad_output):
            if not self.is_leaf_module(module):
                return
            if not self.enable_backward_hook:
                return
            self.run_backward(
                module=module,
                grad_input=grad_input,
                grad_output=grad_output,
                valid_gradnorm_threads=self.valid_gradnorm_threads,
                module_name=module_name,
                enable_log_online=self.enable_log_online,
                enable_log_to_file=self.enable_log_to_file
            )
        return backward_hook
    
    def register_modulewise_hooks(self, model: nn.Module):
        """为模型注册 hooks（只需要调用一次）"""
        self.remove_existing_hooks()  # 先清理现有的 hooks
        
        for name, module in model.named_modules():
            if name == "":  # 跳过根模块（空名字）
                continue
                
            # 注册 forward hook
            forward_handle = module.register_forward_hook(self.create_module_forward_hook(name))
            
            # 注册 backward hook
            try:
                backward_handle = module.register_full_backward_hook(self.create_module_backward_hook(name))
                self.hooks.extend([forward_handle, backward_handle])
            except Exception as e:
                print(f"警告: 无法为 {name} 注册 backward hook: {e}")
                self.hooks.append(forward_handle)
        
        print(f"已为模型注册 {len(self.hooks)} 个 hooks")
    
    #################### Handled hook ####################
    @classmethod
    def run_forward_hooks_by_handle(self, input, output, 
                                    module_name,
                                     enable_log_online=False,
                                     enable_log_to_file=False):
        DebugHookManager().run_forward(
            module=None,
            input=input,
            output=output,
            module_name=module_name,
            enable_log_online=enable_log_online,
            enable_log_to_file=enable_log_to_file
        )

    
    def remove_existing_hooks(self):
        """移除所有 hooks（完成后调用）"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.forward_grad_states = {}
        print("已清理所有 hooks")

    def save(self):
        iteration_id = get_iteration_id()
        node_save_dir = get_node_save_dir(self.save_dir)

        local_rank = torch.distributed.get_rank() % 8
        iteration_save_dir = os.path.join(node_save_dir, f"iteration_{iteration_id}")
        os.makedirs(iteration_save_dir, exist_ok=True)

        save_path = os.path.join(iteration_save_dir, f'rank{local_rank}_hook_data.json')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(DebugHookManager.debug_hook_datas, f, ensure_ascii=False, indent=4)
            DebugHookManager.debug_hook_datas={}
            print(f"debug_hook_datas已成功保存到JSON文件:{save_path}， 已清空debug_hook_datas")