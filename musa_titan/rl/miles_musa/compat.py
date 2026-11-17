"""Source-level compatibility rewrites for supported Miles/Megatron versions."""

from __future__ import annotations

from musa_titan.rl.common import _MILES_AUTHORED, _MILES_COMMIT
from musa_titan.utils.patching import replace_source_fragment


_REWRITE_IMPORT_NAMES = {
    "miles.backends.megatron_utils.arguments": {"megatron.training.tokenizer"},
    "miles.backends.megatron_utils.model": {"megatron.core.optimizer.muon"},
    "miles.backends.megatron_utils.actor": {"sglang.srt.distributed.parallel_state"},
}

def _rewrite_miles_source(fullname: str, source: str) -> str:
    """Apply minimal source fallbacks for supported Miles import failures."""
    if fullname == "miles.backends.megatron_utils.arguments":
        # Current Megatron removed the legacy training.tokenizer package and
        # exposes the public padding helper under core.tokenizers. Keep the
        # legacy fallback for older Megatron checkouts.
        return replace_source_fragment(
            source=source,
            original_source=(
                "from megatron.training.tokenizer.tokenizer import "
                "_vocab_size_with_padding\n"
            ),
            replacement_source=(
                "try:\n"
                "    from megatron.core.tokenizers.utils.build_tokenizer import "
                "vocab_size_with_padding as _vocab_size_with_padding\n"
                "except ModuleNotFoundError as exc:\n"
                "    if exc.name != 'megatron.core.tokenizers':\n"
                "        raise\n"
                "    from megatron.training.tokenizer.tokenizer import "
                "_vocab_size_with_padding\n"
            ),
            reason=(
                "Use Megatron's public tokenizer padding API while retaining "
                "the legacy tokenizer fallback."
            ),
            authored=_MILES_AUTHORED,
            commits={"miles": _MILES_COMMIT},
            upstream_path=(
                "miles.backends.megatron_utils.arguments:"
                "megatron.training.tokenizer.tokenizer._vocab_size_with_padding"
            ),
            current_path="musa_titan/rl/miles_musa/compat.py",
            commit_branch_verified=True,
        )
    if fullname == "miles.backends.megatron_utils.model":
        # Keep Muon optional when the supported Megatron checkout does not ship it.
        source = replace_source_fragment(
            source=source,
            original_source="from megatron.core.optimizer.muon import get_megatron_muon_optimizer\n",
            replacement_source="try:\n"
            "    from megatron.core.optimizer.muon import get_megatron_muon_optimizer\n"
            "except ModuleNotFoundError as exc:\n"
            "    if exc.name != 'megatron.core.optimizer.muon':\n"
            "        raise\n"
            "    def get_megatron_muon_optimizer(*args, **kwargs):\n"
            "        raise ModuleNotFoundError(\n"
            "            'Megatron Muon optimizer is not available in the current MEGATRON_PATH. '\n"
            "            'Use --optimizer adam/sgd, or switch to a Megatron version that provides '\n"
            "            'megatron.core.optimizer.muon.'\n"
            "        )\n",
            reason="Make Miles' optional Muon import tolerate supported Megatron versions without Muon.",
            authored=_MILES_AUTHORED,
            commits={"miles": _MILES_COMMIT},
            upstream_path=(
                "miles.backends.megatron_utils.model:"
                "megatron.core.optimizer.muon.get_megatron_muon_optimizer"
            ),
            current_path="musa_titan/rl/miles_musa/compat.py",
            commit_branch_verified=True,
        )
        return source
    if fullname == "miles.backends.megatron_utils.actor":
        # Let broadcast/disk-delta modes load when optional P2P dependencies fail.
        return replace_source_fragment(
            source=source,
            original_source="from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P\n",
            replacement_source="try:\n"
            "    from .update_weight.update_weight_from_distributed.p2p import UpdateWeightP2P\n"
            "except ImportError:\n"
            "    UpdateWeightP2P = None\n",
            reason="Make Miles' optional P2P import tolerate supported SGLang versions without the P2P dependency.",
            authored=_MILES_AUTHORED,
            commits={"miles": _MILES_COMMIT},
            upstream_path=(
                "miles.backends.megatron_utils.actor:"
                "update_weight_from_distributed.p2p.UpdateWeightP2P"
            ),
            current_path="musa_titan/rl/miles_musa/compat.py",
            commit_branch_verified=True,
        )
    return source
