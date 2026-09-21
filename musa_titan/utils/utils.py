"""Expose the shared Megatron debug helpers without duplicating their state."""

import sys

from ..megatron import utils

# Share the implementation and mutable debug state with the new import path.
sys.modules[__name__] = utils
