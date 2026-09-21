from queue import Queue

import torch

import ce_alltoall

class _CEAllToAllBuffer:
    def __init__(self):
        self._tensor_f = None
        self._tensor_b = None
        self._event = None
        self.size_in_bytes = None

    def alloc_buffer(self, size_in_bytes):
        self._tensor_f: torch.Tensor = ce_alltoall.alloc_tensor(size_in_bytes)
        print(f'alloc buffer for ce alltoall: {self._tensor_f.data_ptr()}')
        self._tensor_b: torch.Tensor = ce_alltoall.alloc_tensor(size_in_bytes)
        print(f'alloc buffer for ce alltoall: {self._tensor_b.data_ptr()}')
        self.size_in_bytes = size_in_bytes

    def get_tensor(self, for_backward=False):
        return self._tensor_b if for_backward else self._tensor_f
    

_ce_alltoall_token_buffer = _CEAllToAllBuffer()
_ce_alltoall_probs_buffer = _CEAllToAllBuffer()

def get_ce_alltoall_token_buffer():
    return _ce_alltoall_token_buffer

def get_ce_alltoall_probs_buffer():
    return _ce_alltoall_probs_buffer
