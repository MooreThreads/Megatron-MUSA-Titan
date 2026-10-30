import os

import torch

# Keep the Library alive for the lifetime of the process. PyTorch removes
# Python-registered kernels when their owning Library object is destroyed.
_ATEN_IMPL_LIBRARY = None


def install_jagged_to_padded_dense() -> None:
    """Install a MUSA fallback for jagged NestedTensor padding."""
    global _ATEN_IMPL_LIBRARY

    if os.getenv("MUSA_PATCH_JAGGED_TO_PADDED_DENSE", "1") != "1":
        return

    operator = "aten::_jagged_to_padded_dense_forward"
    dispatch_key = "PrivateUse1"
    if torch._C._dispatch_has_kernel_for_dispatch_key(operator, dispatch_key):
        return

    def jagged_to_padded_dense_musa(
        values: torch.Tensor,
        offsets: list[torch.Tensor],
        max_lengths: list[int],
        padding_value: float = 0.0,
    ) -> torch.Tensor:
        if len(offsets) != 1 or len(max_lengths) != 1:
            raise NotImplementedError(
                "The MUSA jagged-to-padded fallback supports exactly one jagged dimension"
            )

        offsets_list = offsets[0].tolist()
        batch_size = len(offsets_list) - 1
        max_length = int(max_lengths[0])
        output = values.new_full(
            (batch_size, max_length, *values.shape[1:]),
            padding_value,
        )

        for index, (start, end) in enumerate(
            zip(offsets_list[:-1], offsets_list[1:], strict=True)
        ):
            sequence_length = end - start
            if sequence_length > max_length:
                raise ValueError(
                    f"sequence {index} length {sequence_length} exceeds padded length {max_length}"
                )
            output[index, :sequence_length] = values[start:end]

        return output

    _ATEN_IMPL_LIBRARY = torch.library.Library("aten", "IMPL", dispatch_key)
    _ATEN_IMPL_LIBRARY.impl(
        "_jagged_to_padded_dense_forward",
        jagged_to_padded_dense_musa,
    )


def install() -> None:
    """Install MUSA NestedTensor compatibility patches."""
    install_jagged_to_padded_dense()
