from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components import ComponentType


@dataclasses.dataclass
class CacheInitParams:
    disable: bool
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int

    is_eagle: bool = False
    tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_cp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    attn_tp_cache_group: Optional[torch.distributed.ProcessGroup] = None
    eviction_policy: str = "lru"
    disable_finished_insert: bool = False

    enable_metrics: bool = False
    enable_kv_cache_events: bool = False

    enable_mamba_extra_buffer: bool = False

    pp_rank: int = 0
    pp_size: int = 1

    attn_cp_rank: int = 0
    attn_cp_size: int = 1

    chunked_prefill_size: Optional[int] = None

    sliding_window_size: Optional[int] = None

    # Time-to-live for cache entries in seconds. If None, TTL is disabled.
    cache_ttl_seconds: Optional[float] = None

    tree_components: Optional[tuple[ComponentType, ...]] = None

    # Draft model's token_to_kv_pool, used by KV connectors that support
    # speculative-decoding piggyback (target + draft KV stored/loaded together).
    # Set by build_kv_cache when ``--speculative-algorithm`` and
    # ``--kv-connector-cls`` are both enabled. ``None`` otherwise — connectors
    # treat ``None`` as "no MTP piggyback" and fall back gracefully.
    #
    # Why this is here (not registered later via a setter):
    # FlexKV's TransferManager doesn't accept re-registration of a GPU
    # device_id, so the draft pool MUST be available BEFORE the connector
    # calls ``_register_to_server``. The earliest hook with both target and
    # draft pools live is the connector's ``__init__``, hence this field.
    draft_token_to_kv_pool: Optional[object] = None
