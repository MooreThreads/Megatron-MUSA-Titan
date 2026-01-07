import megatron
from megatron.core.dist_checkpointing.strategies.filesystem_async import *

class FileSystemWriterAsync_patch(FileSystemWriterAsync):

	def get_save_function_and_args(self) -> Tuple[Optional[Callable], Optional[Callable], List]:
		"""
		Get function that saves the data to storage along with its arguments.
		Allows the external caller to apply the save function synchronously or asynchronously.

		Returns: None (if there is nothing to write on this rank) or a tuple of:
			1) the function that saves the data.
			2) the function that stages the GPU tensors to a destination for async checkpointing.
				This function should be self-contained.
			3) arguments to that function in 1).
		"""
		if not self.write_buckets:
			return None, None, []
		transform_list = [self.transforms] if hasattr(self, "transforms") else []
		return (
			partial(self.write_preloaded_data_multiproc, transform_list, self.use_msc),
			partial(self.preload_tensors, self.write_buckets, True),
			[torch.distributed.get_rank(), self.write_buckets, self.results_queue],
		)

	@staticmethod
	def preload_tensors(write_buckets: List[WriteBucket], non_blocking=True) -> List[WriteBucket]:
		"""
		Preloads tensors in `state_dict` to host memory via CPU memory.

		Args:
			write_buckets (List): List of `WriteBucket` objects that define what to
				save in a checkpoint.
			non_blocking (bool, optional): knob to enable pinned D2H memcpy. Default is True.
		"""

		if non_blocking:
			import warnings
			warnings.warn("non_blocking is not supported in FileSystemWriterAsync.preload_tensors yet")
			non_blocking = False

		result = []

		for bucket in write_buckets:
			file_name, storage_key, (bytes_data, tensor_data) = bucket
			tensor_data = [
				(item, tensor.to("cpu", non_blocking=non_blocking)) for item, tensor in tensor_data
			]
			result.append((file_name, storage_key, (bytes_data, tensor_data)))
		if non_blocking:
			torch.cuda.synchronize()
		return result

# patch
megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync = FileSystemWriterAsync_patch
