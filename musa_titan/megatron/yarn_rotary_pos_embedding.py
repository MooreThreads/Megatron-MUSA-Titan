# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
"""Increase the YaRN cache capacity for large pipeline stages."""

from functools import lru_cache

from megatron.core.models.common.embeddings import yarn_rotary_pos_embedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding

from musa_titan.utils.patching import git_commit, replace_attr


replace_attr(
    YarnRotaryEmbedding, "forward", lru_cache(maxsize=64)(YarnRotaryEmbedding.forward),
    reason="Cache 64 YaRN results for pipeline stages with more than 32 layers",
    commits={"megatron": git_commit(yarn_rotary_pos_embedding.__file__)},
    authored={
        "author": "huang.huang", "modified_date": "unknown",
        "upstream": {"megatron": {
            "branch": "main", "commit": "e855289f", "commit_date": "Oct 08, 2025",
        }},
    },
    upstream_path="megatron.core.models.common.embeddings.yarn_rotary_pos_embedding.YarnRotaryEmbedding.forward",
    current_path="musa_titan/megatron/yarn_rotary_pos_embedding.py",
)
