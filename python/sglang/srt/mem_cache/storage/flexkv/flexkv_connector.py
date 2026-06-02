import logging
import os
import socket
import struct
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.kv_connector import BaseKVConnector, LoadOperation
from sglang.srt.mem_cache.storage.flexkv.flexkv_comm import (
    CMD_PUT_META,
    CMD_LAYERWISE,
    CMD_STORE_COMPLETE,
    FlexKVLayerDoneCounter,
    FlexKVComm,
    send_fds,
)

try:
    from flexkv.common.config import LayerGroupSpec
    from flexkv.common.request import KVResponseStatus
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.integration.config import FlexKVConfig
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVTPClient
    from flexkv.transfer.layerwise import build_layerwise_eventfd_socket_path
    from flexkv.transfer_manager import TransferManagerOnRemote
except ImportError as e:
    raise RuntimeError("FlexKV is not installed. Please install it.") from e

logger = logging.getLogger(__name__)


# ---- FlexKV Connector ----


class FlexKVConnector(BaseKVConnector):
    """KV cache connector backed by FlexKV's distributed cache system.

    Implements ``BaseKVConnector`` so it can be used with
    ``ExtendedRadixCache`` via ``--kv-connector-cls``.
    """

    def __init__(
        self,
        params: Any,
        server_args: Any,
        tp_rank: int = 0,
        dp_rank: Optional[int] = 0,
        attn_cp_rank: Optional[int] = 0,
        pp_group: Any = None,
        attn_tp_group: Any = None,
        attn_cp_group: Any = None,
    ):
        super().__init__(
            params=params,
            server_args=server_args,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )

        # ---- Primitive variables (from params / constructor) ----
        self.page_size = params.page_size
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()

        sglang_model_config = ModelConfig.from_server_args(server_args)

        # ---- Initialize FlexKV config ----
        self.flexkv_config = FlexKVConfig.from_env()
        rank_info = self.flexkv_config.post_init_from_sglang_config(
            sglang_config=sglang_model_config,
            server_args=server_args,
            page_size=self.page_size,
            tp_rank=tp_rank,
            pp_rank=params.pp_rank,
            dp_rank=dp_rank,
            attn_cp_rank=attn_cp_rank,
            kv_cache_pool=kvcache,
        )

        model_config = self.flexkv_config.model_config
        cache_config = self.flexkv_config.cache_config
        self.rank_info = rank_info

        # Structured logging label
        self._rank_label = f" [model_config={model_config}, rank_info={rank_info}]"

        # ---- Communication / sync context ----
        self._sync_ctx = FlexKVComm(
            rank_info=rank_info,
            world_rank=(
                torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            ),
            pp_group=pp_group,
            attn_tp_group=attn_tp_group,
            attn_cp_group=attn_cp_group,
        )
        logger.debug(
            f"[FlexKV] sync_context{self._rank_label}: "
            f"is_sync_leader={self._sync_ctx.is_sync_leader}, "
            f"needs_sync={self._sync_ctx.needs_sync}, "
            f"is_pp_active={self._sync_ctx.is_pp_active}"
        )


        # ---- Align block counts on unified group (single all_reduce MIN) ----
        for _attr in ("num_cpu_blocks", "num_ssd_blocks", "num_remote_blocks"):
            _orig = getattr(self.flexkv_config.cache_config, _attr)
            if _orig is None or _orig <= 0:
                continue
            _aligned = self._sync_ctx.all_reduce_min(_orig)
            logger.debug(
                f"[FlexKV] Block count alignment{self._rank_label}: "
                f"attr={_attr}, {_orig} -> {_aligned}"
            )
            if _aligned != _orig:
                logger.info(
                    f"[FlexKV] Block count MIN alignment '{_attr}': "
                    f"{_orig} -> {_aligned}"
                )
            setattr(self.flexkv_config.cache_config, _attr, _aligned)

        if model_config.nnodes > 1:
            logger.info(
                f"[FlexKV] Multi-node detected{self._rank_label}: "
                f"model_config={model_config}, rank_info={rank_info}"
            )

        # Optional: disable the SWA LayerGroup entirely for debug When SGLANG_FLEXKV_DISABLE_SWA=1
        self._force_disable_swa = bool(int(os.getenv("SGLANG_FLEXKV_DISABLE_SWA", "0")))
        if self._force_disable_swa:
            if (model_config.layer_groups is not None
                    and len(model_config.layer_groups) == 4
                    and model_config.layer_groups[3].compress_ratio == 1):
                logger.info(
                    "[FlexKV-DSV4] SGLANG_FLEXKV_DISABLE_SWA=1: dropping the "
                    "4th (swa) LayerGroupSpec from model_config.layer_groups."
                )
                model_config.layer_groups = model_config.layer_groups[:3]
            if cache_config.swa is not None:
                logger.info(
                    "[FlexKV-DSV4] SGLANG_FLEXKV_DISABLE_SWA=1: disabling "
                    "cache_config.swa (SWA side-channel becomes a no-op)."
                )
                cache_config.swa = None

        # Build unified kv_caches list (MLA vs MHA vs DSV4-SWA)
        logger.info(
            f"[FlexKV-DEBUG] kvcache class={type(kvcache).__name__}, "
            f"hasattr(kv_buffer)={hasattr(kvcache, 'kv_buffer')}, "
            f"hasattr(k_buffer)={hasattr(kvcache, 'k_buffer')}, "
            f"hasattr(swa_kv_pool)={hasattr(kvcache, 'swa_kv_pool')}"
        )

        # DSV4: top-level pool has no kv_buffer / k_buffer; the real KV tensors
        # are split across c4 / c128 / c4_indexer / swa sub-pools with three
        # different compression ratios. Collect them as a multi-group
        # registration so all four sub-pools land in FlexKV's CPU/SSD/Remote
        # cache. For any other BaseSWAKVPool (only swa_kv_pool present), fall
        # back to a SWA-only single-group unwrap.
        self._dsv4_registration: Optional[Dict[str, Any]] = None
        if self._is_dsv4_pool(kvcache):
            self._dsv4_registration = self._collect_dsv4_registration(kvcache)
            reg = self._dsv4_registration
            kv_caches = reg["handles_per_group"][0]
            indexer_buffers = None
            logger.info(
                f"[FlexKV-DSV4] Collected multi-group registration: "
                f"groups={len(reg['layer_groups'])}, "
                f"c4_layers={reg['layer_groups'][0].num_layers}, "
                f"c128_layers={reg['layer_groups'][1].num_layers}, "
                f"indexer_layers={reg['layer_groups'][2].num_layers} "
                f"(SWA handled via side-channel swa_put/swa_get)"
            )
        elif hasattr(kvcache, "swa_kv_pool") and not (
            hasattr(kvcache, "kv_buffer") or hasattr(kvcache, "k_buffer")
        ):
            swa_inner = kvcache.swa_kv_pool
            kv_caches = swa_inner.kv_buffer
            indexer_buffers = None
            logger.info(
                f"[FlexKV-DSV4] Unwrapped swa_kv_pool from {type(kvcache).__name__}: "
                f"layers={len(kv_caches)}, "
                f"shape={tuple(kv_caches[0].shape) if kv_caches else None}, "
                f"dtype={kv_caches[0].dtype if kv_caches else None}"
            )
        elif hasattr(kvcache, "kv_buffer"):
            # MLA: K and V share the same buffer, register once per layer
            kv_caches = kvcache.kv_buffer
            indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        elif hasattr(kvcache, "k_buffer"):
            # MHA: separate K and V buffers, concat as [k_layers..., v_layers...]
            kv_caches = kvcache.k_buffer + kvcache.v_buffer
            indexer_buffers = getattr(kvcache, "index_k_with_scale_buffer", None)
        else:
            raise AttributeError(
                f"Unsupported KV cache type {type(kvcache).__name__}: "
                f"expected swa_kv_pool / kv_buffer / k_buffer."
            )

        if indexer_buffers is not None and len(indexer_buffers) > 0:
            logger.info(
                f"[FlexKV] Detected sparse attention indexer cache with "
                f"{len(indexer_buffers)} indexer layers, "
                f"shape={indexer_buffers[0].shape}"
            )
        logger.info(
            f"[FlexKV-DEBUG] picked kv_caches: len={len(kv_caches)}, "
            f"[0].shape={tuple(kv_caches[0].shape) if len(kv_caches) > 0 else None}, "
            f"[0].dtype={kv_caches[0].dtype if len(kv_caches) > 0 else None}, "
            f"[0].stride={tuple(kv_caches[0].stride()) if len(kv_caches) > 0 else None}"
        )

        # ---- Node B: Launch TransferManagerOnRemote ----
        self._remote_process = None
        if model_config.nnodes > 1 and rank_info.node_rank > 0 and rank_info.local_rank == 0:
            logger.debug(
                f"[FlexKV] Launching TransferManagerOnRemote{self._rank_label}: "
                f"master_host={self.flexkv_config.model_config.master_host}, "
                f"master_ports={self.flexkv_config.model_config.master_ports}")
            self._remote_process = TransferManagerOnRemote.create_process(
                master_host=self.flexkv_config.model_config.master_host,
                master_ports=self.flexkv_config.model_config.master_ports,
            )
            logger.info(
                f"[FlexKV] Launched TransferManagerOnRemote on node_rank={rank_info.node_rank}"
                f"{self._rank_label}"
            )

        if self._sync_ctx.is_sync_leader:
            self.kv_manager = KVManager(
                model_config=model_config,
                cache_config=cache_config,
                dp_client_id=rank_info.dp_client_id,
                server_recv_port=self.flexkv_config.server_recv_port,
                gpu_register_port=self.flexkv_config.gpu_register_port,
            )
            self.kv_manager.start()
            logger.info(
                f"[FlexKV] Creating KVManager{self._rank_label}: "
                f"server_recv_port={self.flexkv_config.server_recv_port}, "
                f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        # ---- GPU Registration Routing ----
        self.dp_client_id = rank_info.dp_client_id
        self.pp_rank = rank_info.pp_rank
        self.tp_client = KVTPClient(
            self.flexkv_config.gpu_register_port,
            dp_client_id=self.dp_client_id,
            pp_rank=self.pp_rank,
            device_id=rank_info.local_rank,
        )

        # ---- GPU Registration (with retry) ----
        self._register_with_retry(kv_caches, indexer_buffers)
        logger.info(
            f"[FlexKV] KVTPClient registered to server{self._rank_label}: "
            f"gpu_register_port={self.flexkv_config.gpu_register_port}")

        self.enable_layerwise_transfer = bool(
            int(os.getenv("FLEXKV_ENABLE_LAYERWISE_TRANSFER", "0"))
        )

        self.layerwise_eventfd_socket = build_layerwise_eventfd_socket_path(
            dp_client_id=self.dp_client_id,
            pp_rank=self.pp_rank,
            model_config=model_config,
        )
        logger.info(
            f"[FlexKV] Eventfd socket path configured{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, "
            f"layerwise_transfer={self.enable_layerwise_transfer}")
        self.layerwise_eventfd_connect_max_retries = max(
            360,
            int(os.getenv("FLEXKV_LAYERWISE_EVENTFD_CONNECT_MAX_RETRIES", "0")),
        )
        self._layer_done_counter: Optional[FlexKVLayerDoneCounter] = None
        self._worker_connected = False

        self._init_layer_transfer_components()

        if self._layer_done_counter is not None and kvcache is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

        # rid -> flexkv_task_id (pending loads awaiting start_load_kv)
        self._pending_loads: Dict[str, int] = {}
        # ext_task_id -> producer_id (layerwise loads in flight)
        self._ongoing_loads: Dict[int, int] = {}
        # ext_task_ids whose load has completed
        self._completed_loads: List[int] = []
        # ext_task_id -> flexkv_task_id (stores in flight)
        self._ongoing_stores: Dict[int, int] = {}
        # ext_task_ids whose store has completed or was skipped
        self._completed_stores: List[int] = []
        # flexkv task ids for periodic drain to prevent pipe deadlock
        self._load_fkv_tids: List[int] = []
        # rid -> flexkv_task_id (prefetch in flight)
        self._ongoing_prefetches: Dict[str, int] = {}
        # rid -> number of tokens prefetched from CPU/SSD/remote (filled in
        # check_prefetch_progress when the task succeeds, drained by
        # pop_prefetch_loaded_tokens). Used by scheduler.py to set
        # ``req.storage_hit_length`` so the L3-hit prefix is reused instead of
        # recomputed on the second request.
        self._prefetch_loaded_tokens: Dict[str, int] = {}
        self._prefetch_enabled = bool(
            cache_config.enable_ssd
            or cache_config.enable_remote
            or cache_config.enable_kv_sharing
        )

        # ---- SWA (Sliding Window Attention) GPU pool detection ----
        self._kvcache = kvcache  # Store full kvcache for translate_loc_from_full_to_swa
        self._swa_kv_pool = (
            None if self._force_disable_swa
            else getattr(kvcache, 'swa_kv_pool', None)
        )
        # Prefer values exposed by the pool itself — DSV4 sets swa_window_size /
        # swa_page_size on the top-level pool, and per-token byte width is
        # available via get_bytes_per_token() on the SWA sub-pool. Fall back to
        # cache_config.swa for legacy non-DSV4 SWA configs.
        self._swa_window_size = int(getattr(kvcache, 'swa_window_size', 0) or 0)
        self._swa_page_size = (
            self._swa_kv_pool.page_size
            if self._swa_kv_pool is not None
            else 0
        )
        if self._swa_kv_pool is not None and hasattr(self._swa_kv_pool, 'get_bytes_per_token'):
            self._swa_bytes_per_token_per_layer = int(self._swa_kv_pool.get_bytes_per_token())
        else:
            self._swa_bytes_per_token_per_layer = 0
        if cache_config.swa is not None and cache_config.swa.enabled:
            if self._swa_window_size == 0:
                self._swa_window_size = cache_config.swa.window_size
            if self._swa_bytes_per_token_per_layer == 0:
                self._swa_bytes_per_token_per_layer = cache_config.swa.bytes_per_token_per_layer
        self._device = kv_caches[0].device if kv_caches else torch.device("cuda")
        # rid -> token_ids prefix for pending SWA loads
        self._pending_swa_token_ids: Dict[str, np.ndarray] = {}
        if self._swa_kv_pool is not None:
            logger.info(
                f"[FlexKV-SWA] Detected SWA KV pool on kvcache, "
                f"window_size={self._swa_window_size}, "
                f"swa_page_size={self._swa_page_size}, "
                f"bytes_per_token_per_layer={self._swa_bytes_per_token_per_layer}"
            )

        if self._sync_ctx.is_sync_leader:
            wait_count = 0
            while not self.kv_manager.is_ready():
                time.sleep(10)
                wait_count += 1
                # Collect diagnostic info for debugging
                diag_parts = []
                # Check IPC socket file existence
                gpu_port = self.flexkv_config.gpu_register_port
                if gpu_port.startswith("ipc://"):
                    ipc_path = gpu_port[len("ipc://"):]
                    ipc_exists = os.path.exists(ipc_path)
                    diag_parts.append(f"ipc_socket={ipc_path} exists={ipc_exists}")
                # Check TransferManager subprocess status
                task_engine = getattr(self.kv_manager, 'kv_task_engine', None)
                if task_engine is not None:
                    for i, th in enumerate(getattr(task_engine, 'transfer_handles', [])):
                        handle = getattr(th, '_handle', None)
                        if handle is not None:
                            parts = []
                            start_evt = getattr(handle, 'start_event', None)
                            ready_evt = getattr(handle, 'ready_event', None)
                            proc = getattr(handle, 'process', None)
                            if start_evt is not None:
                                parts.append(f"started={start_evt.is_set()}")
                            if ready_evt is not None:
                                parts.append(f"ready={ready_evt.is_set()}")
                            if proc is not None:
                                parts.append(f"alive={proc.is_alive()}")
                            if parts:
                                diag_parts.append(f"transfer_handle[{i}]: {', '.join(parts)}")
                diag_str = "; ".join(diag_parts) if diag_parts else "no diagnostics available"
                logger.info(
                    f"[FlexKV] Waiting for FlexKV to be ready{self._rank_label}... "
                    f"(waited {wait_count * 10}s, {diag_str})"
                )
            logger.info(f"[FlexKV] FlexKV is ready{self._rank_label}")
        elif model_config.nnodes > 1 and rank_info.node_rank > 0:
            # Node B: no KVManager to wait for, GPU registration retry handles readiness
            logger.info(f"[FlexKV] Node B skipping is_ready wait{self._rank_label}")

        logger.info(
            f"[FlexKV] Connector initialized{self._rank_label}: "
            f"layerwise_transfer={self.enable_layerwise_transfer}, "
            f"prefetch_enabled={self._prefetch_enabled}, "
            f"model_config={model_config}, rank_info={rank_info}"
        )

    # ---- BaseKVConnector abstract methods ----

    def get_new_hit_length(
        self,
        token_ids: List[int],
        token_mask: torch.Tensor,
        update_state_for_load: bool = False,
        rid: Optional[str] = None,
    ) -> int:
        hit_length = 0
        flexkv_task_id = -1

        # INFO: TP/CP group is strictly synchronous, so TP/CP ranks are symmetric. This means they
        #       have identical dst GPU blocks. Hence, let TP/CP rank 0 do prefix matching on the
        #       TP/CP group's behalf and broadcast the result to the rest of the group.
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            result = self.kv_manager.get_match(
                token_ids=token_ids_np,
                token_mask=token_mask,
            )
            # get_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Guard against unpacking a
            # None result to avoid crashing the scheduler.
            if result is None:
                logger.warning("[FlexKV] get_match returned None, treating as no hit")
                flexkv_task_id = -1
                hit_length = 0
            else:
                flexkv_task_id, matched_mask = result
                hit_length = int(matched_mask.sum()) if matched_mask is not None else 0
            if not update_state_for_load and flexkv_task_id >= 0:
                # Only cancel if the task actually has pending work.  When
                # hit_length == 0 the transfer graph is empty and the task was
                # already marked COMPLETED synchronously inside get_match →
                # _process_empty_graph, so cancelling would be a no-op that
                # triggers a spurious "already completed" warning.
                if hit_length > 0:
                    self.kv_manager.cancel([flexkv_task_id])
            else:
                ## GPU hit length is the zero length of token masks
                gpu_hit_length = torch.logical_not(token_mask).sum()
                logger.debug(f"[FlexKV Connector] gpu hit length: {gpu_hit_length}, Flexkv hit length: {hit_length}")

                # SWA: Check if SWA data is available for the matched prefix.
                # If SWA is not available for the matched prefix, search backward
                # (page-aligned) until we find a point where SWA IS available.
                # If nothing found, reduce hit_length to 0 (force full recompute).
                if hit_length > 0 and self._swa_kv_pool is not None and hasattr(self.kv_manager, 'swa_available'):
                    try:
                        matched_prefix = token_ids_np[:hit_length]
                        swa_avail = self.kv_manager.swa_available(matched_prefix)

                        if swa_avail:
                            logger.debug(
                                f"[FlexKV-SWA] SWA available for prefix hit_length={hit_length}"
                            )
                            # Mark for SWA load-back in start_load_kv
                            if rid is not None:
                                self._pending_swa_token_ids[rid] = token_ids_np[:hit_length].copy()
                        else:
                            # SWA not available — search backward page by page
                            original_hit = hit_length
                            while hit_length > 0:
                                hit_length -= self.page_size
                                if hit_length <= 0:
                                    hit_length = 0
                                    break
                                shorter_prefix = token_ids_np[:hit_length]
                                if self.kv_manager.swa_available(shorter_prefix):
                                    break

                            logger.debug(
                                f"[FlexKV-SWA] Reduced hit_length {original_hit} -> {hit_length} "
                                f"due to missing SWA"
                            )

                            # Cancel the original task since hit_length changed
                            if flexkv_task_id >= 0:
                                self.kv_manager.cancel([flexkv_task_id])
                                flexkv_task_id = -1

                            if hit_length > 0:
                                # Re-do get_match with shorter prefix for correct task setup
                                result = self.kv_manager.get_match(
                                    token_ids=token_ids_np[:hit_length],
                                    token_mask=None,
                                )
                                if result is not None:
                                    flexkv_task_id, _ = result
                                    # Mark for SWA load-back
                                    if rid is not None:
                                        self._pending_swa_token_ids[rid] = token_ids_np[:hit_length].copy()
                                else:
                                    hit_length = 0
                                    flexkv_task_id = -1
                    except Exception as swa_err:
                        logger.debug(f"[FlexKV-SWA] SWA check in get_new_hit_length failed: {swa_err}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"hit_length": hit_length, "task_id": flexkv_task_id},
            )
            hit_length = data["hit_length"]
            flexkv_task_id = data["task_id"]

        # Page-align host_hit_length: ensure GET loads complete pages
        if hit_length > 0 and self.page_size > 1:
            aligned_hit = (hit_length // self.page_size) * self.page_size
            if aligned_hit < hit_length:
                logger.debug(
                    "[FlexKV] get_new_hit_length: host_hit_length page_align %d -> %d (page_size=%d)",
                    hit_length, aligned_hit, self.page_size,
                )
                hit_length = aligned_hit

        if update_state_for_load and rid is not None and hit_length > 0:
            self._pending_loads[rid] = flexkv_task_id
        elif update_state_for_load and flexkv_task_id >= 0 and self._sync_ctx.is_sync_leader:
            # Task was not cancelled earlier, but won't be used — cancel it now
            # to avoid resource leak (e.g. hit_length page-aligned to 0, or rid is None).
            # Skip cancel when hit_length == 0: the task's transfer graph was
            # empty and _process_empty_graph already marked it COMPLETED.
            if hit_length > 0:
                self.kv_manager.cancel([flexkv_task_id])
        return hit_length

    def release_load_state(self, rid: str) -> None:
        fkv_tid = self._pending_loads.pop(rid, -1)
        self._pending_swa_token_ids.pop(rid, None)
        if fkv_tid >= 0 and self._sync_ctx.is_sync_leader:
            self.kv_manager.cancel([fkv_tid])

    def start_load_kv(
        self,
        task_id: int,
        load_ops: List[LoadOperation],
    ) -> None:
        flexkv_task_ids: List[int] = []
        slot_mappings: List[torch.Tensor] = []

        for op in load_ops:
            fkv_tid = self._pending_loads.pop(op.rid, -1)
            if fkv_tid < 0:
                continue
            flexkv_task_ids.append(fkv_tid)
            indices = op.device_indices
            slot_mapping_cpu = indices.cpu() if indices.is_cuda else indices
            slot_mapping_cpu = slot_mapping_cpu.to(torch.int64)
            slot_mappings.append(slot_mapping_cpu)

        # SWA: Restore sliding window data to GPU before main KV load
        # This ensures SWA data is available when the model forward starts.
        if self._swa_kv_pool is not None and hasattr(self, '_pending_swa_token_ids'):
            for op in load_ops:
                self._do_swa_restore_for_op(op)

        logger.debug(f"[FlexKV] start_load_kv: resolved {len(flexkv_task_ids)} flexkv tasks")
        if not flexkv_task_ids:
            self._completed_loads.append(task_id)
            return

        if self._sync_ctx.should_send_slot_mapping_to_remote:
            logger.debug(f"[FlexKV] start_load_kv: sending slot_mapping for cross-node pp_receiver")
            for fkv_tid, slot_map in zip(flexkv_task_ids, slot_mappings):
                self.send_slot_mapping_to_remote(fkv_tid, slot_map)

        if self.enable_layerwise_transfer and self._layer_done_counter is not None:
            # PP1+: receive counter_id from PP0 first
            if self._sync_ctx.is_pp_receiver:
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_LAYERWISE:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_LAYERWISE}, got {payload.get('cmd')}")
                producer_id = payload["counter_id"]
                self._layer_done_counter.register_task_with_explicit_counter_id(task_id, producer_id)
            else:
                # Original logic: every rank independently updates producer
                producer_id = self._layer_done_counter.update_producer()
                self._layer_done_counter.events[producer_id].reset_for_new_transfer()
                self._layer_done_counter.register_task(task_id, producer_id)

            # PP0 sync leader: send counter_id to PP1+
            if self._sync_ctx.is_pp_sender:
                self._sync_ctx.scatter_pp(
                    {"cmd": CMD_LAYERWISE, "fkv_task_id": flexkv_task_ids[0], "counter_id": producer_id},
                )

            if self._sync_ctx.is_sync_leader:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=True,
                    counter_id=producer_id,
                )
                self._load_fkv_tids.extend(flexkv_task_ids)
            self._ongoing_loads[task_id] = producer_id
        else:
            if self._sync_ctx.is_sync_leader:
                self.kv_manager.launch(
                    task_ids=flexkv_task_ids,
                    slot_mappings=slot_mappings,
                    as_batch=True,
                    layerwise_transfer=False,
                )
                response = self.kv_manager.wait(flexkv_task_ids, timeout=30.0)
                if not all(
                    tid in response and response[tid].status == KVResponseStatus.SUCCESS
                    for tid in flexkv_task_ids
                ):
                    logger.warning(
                        "[FlexKV] Some tasks failed in non-layerwise transfer"
                    )

            if self._sync_ctx.needs_sync:
                self._sync_ctx.barrier()

            self._completed_loads.append(task_id)

    def check_completed_load_tasks(self) -> List[int]:
        if self._sync_ctx.is_sync_leader and len(self._load_fkv_tids) >= 100:
            self.kv_manager.try_wait(task_ids=self._load_fkv_tids)
            self._load_fkv_tids.clear()

        if self._layer_done_counter is not None:
            for ext_tid, producer_id in list(self._ongoing_loads.items()):
                if self._layer_done_counter.events[producer_id]._finished:
                    self._completed_loads.append(ext_tid)
                    del self._ongoing_loads[ext_tid]

        result = list(self._completed_loads)
        self._completed_loads.clear()
        return result

    def start_store_kv(
        self,
        task_id: int,
        token_ids: List[int],
        kv_indices: torch.Tensor,
    ) -> None:

        def _send_pp_put_meta(fkv_task_id: int, unmatched_mask):
            if not self._sync_ctx.is_pp_active:
                return
            mask_list = (
                unmatched_mask.cpu().tolist()
                if hasattr(unmatched_mask, "is_cuda") and unmatched_mask.is_cuda
                else (unmatched_mask.tolist() if hasattr(unmatched_mask, "tolist") else [])
            )
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_PUT_META, "fkv_task_id": fkv_task_id, "unmatched_mask": mask_list},
            )

        if not self._sync_ctx.is_sync_leader:
            if self._sync_ctx.is_pp_receiver:
                logger.debug(
                    f"[FlexKV-Connector] start_store_kv: PP1+ scatter recv PUT_META"
                )
                payload = self._sync_ctx.scatter_pp(None)
                if payload.get("cmd") != CMD_PUT_META:
                    raise RuntimeError(f"Tag mismatch: expected {CMD_PUT_META}, got {payload.get('cmd')}")
                fkv_task_id = payload["fkv_task_id"]
                unmatched_mask = torch.tensor(payload["unmatched_mask"])
                if unmatched_mask.sum() > 0 and fkv_task_id >= 0:
                    if self._sync_ctx.should_send_slot_mapping_to_remote:
                        filtered = kv_indices[unmatched_mask]
                        slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                        slot_mapping = slot_mapping.to(torch.int64)
                        self.send_slot_mapping_to_remote(fkv_task_id, slot_mapping)
                    self._ongoing_stores[task_id] = fkv_task_id
                else:
                    self._completed_stores.append(task_id)
            return

        try:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            # Page-align token_ids first. Under EAGLE the caller passes one
            # extra trailing boundary token vs. the already-page-aligned
            # kv_indices, so trim down to the page-aligned length before the
            # 1:1 pairing check.
            if self.page_size != 1:
                aligned_len = len(token_ids_np) // self.page_size * self.page_size
                token_ids_np = token_ids_np[:aligned_len]
            assert len(token_ids_np) == len(kv_indices), (
                f"len(token_ids)={len(token_ids_np)} != len(kv_indices)={len(kv_indices)}, "
                f"task_id={task_id}, page_size={self.page_size}, "
                f"kv_indices_shape={kv_indices.shape if hasattr(kv_indices, 'shape') else 'N/A'}"
            )

            # Page-align token_ids and kv_indices BEFORE put_match so that
            # put_match allocates dst_block_ids consistent with the slot_mapping
            # we will later pass to launch().
            original_len = len(token_ids_np)
            if self.page_size > 1:
                aligned_len = (original_len // self.page_size) * self.page_size
                if aligned_len == 0:
                    _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                    self._completed_stores.append(task_id)
                    return
                if aligned_len < original_len:
                    token_ids_np = token_ids_np[:aligned_len]
                    kv_indices = kv_indices[:aligned_len]
            result = self.kv_manager.put_match(
                token_ids=token_ids_np,
                token_mask=None,
            )
            # put_match returns None when the FlexKV server encounters an
            # error (e.g. in server_client_mode).  Treat as a failed store.
            if result is None:
                logger.warning("[FlexKV] put_match returned None, skipping store for task %d", task_id)
                _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
                self._completed_stores.append(task_id)
                return
            fkv_task_id, unmatched_mask = result

            logger.info(
                f"[FlexKV] start_store_kv: task_id={task_id} tokens={len(token_ids)}, "
                f"kv_indices={len(kv_indices)}, fkv_task_id={fkv_task_id}, "
                f"unmatched={unmatched_mask.sum().item() if hasattr(unmatched_mask, 'sum') else len(unmatched_mask)}"
            )

            _send_pp_put_meta(fkv_task_id, unmatched_mask)

            if unmatched_mask.sum() > 0:
                filtered = kv_indices[unmatched_mask]
                slot_mapping = filtered.cpu() if filtered.is_cuda else filtered
                slot_mapping = slot_mapping.to(torch.int64)

                self.kv_manager.launch(
                    task_ids=[fkv_task_id], slot_mappings=[slot_mapping]
                )
                self._ongoing_stores[task_id] = fkv_task_id
                logger.info(
                    "[FlexKV] start_store_kv: LAUNCHED task_id=%d fkv_task_id=%d "
                    "ongoing_stores_size=%d",
                    task_id, fkv_task_id, len(self._ongoing_stores),
                )
            else:
                self._completed_stores.append(task_id)
                logger.info(
                    "[FlexKV] start_store_kv: ALL_MATCHED task_id=%d -> completed_stores",
                    task_id,
                )

            # SWA: Extract and store sliding window attention data alongside main KV
            if self._swa_kv_pool is not None and hasattr(self.kv_manager, 'swa_put'):
                try:
                    swa_data = self._extract_swa_from_gpu(kv_indices)
                    if swa_data is not None:
                        self.kv_manager.swa_put(token_ids_np, swa_data)
                except Exception as swa_err:
                    logger.debug(f"[FlexKV-SWA] SWA store in start_store_kv failed: {swa_err}")
        except Exception as e:
            logger.error("[FlexKV] start_store_kv failed: %s", e, exc_info=True)
            _send_pp_put_meta(fkv_task_id=-1, unmatched_mask=[])
            self._completed_stores.append(task_id)

    def check_completed_store_tasks(self) -> List[int]:
        completed_ext_ids = list(self._completed_stores)
        self._completed_stores.clear()

        completed_dict = {}
        if self._sync_ctx.is_sync_leader and self._ongoing_stores:
            fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
            completed_dict = self.kv_manager.try_wait(task_ids=list(fk_to_ext.keys()))
            if completed_dict or completed_ext_ids:
                logger.info(
                    "[FlexKV] check_completed_store: pre-pop completed_ext_ids=%s "
                    "ongoing_stores_size=%d try_wait_returned=%s",
                    completed_ext_ids, len(self._ongoing_stores), list(completed_dict.keys()),
                )
            for fk_tid in completed_dict:
                ext_tid = fk_to_ext[fk_tid]
                completed_ext_ids.append(ext_tid)
                del self._ongoing_stores[ext_tid]

        if self._sync_ctx.is_pp_sender:
            self._sync_ctx.scatter_pp(
                {"cmd": CMD_STORE_COMPLETE, "completed_fk_ids": list(completed_dict)},
            )
        elif self._sync_ctx.is_pp_receiver:
            payload = self._sync_ctx.scatter_pp(None)
            if payload.get("cmd") != CMD_STORE_COMPLETE:
                raise RuntimeError(f"Tag mismatch: expected {CMD_STORE_COMPLETE}, got {payload.get('cmd')}")
            fk_ids = payload["completed_fk_ids"]
            if fk_ids and self._ongoing_stores:
                fk_to_ext = {v: k for k, v in self._ongoing_stores.items()}
                for fk_tid in fk_ids:
                    if fk_tid in fk_to_ext:
                        ext_tid = fk_to_ext[fk_tid]
                        completed_ext_ids.append(ext_tid)
                        del self._ongoing_stores[ext_tid]

        if self._sync_ctx.needs_sync:
            completed_ext_ids = self._sync_ctx.scatter(completed_ext_ids)

        return completed_ext_ids

    # ---- Optional overrides ----

    def prefetch(self, rid: str, token_ids: List[int]) -> None:
        if not self._prefetch_enabled:
            return
        if not rid:
            return

        prefetch_task_id = -1
        if self._sync_ctx.is_sync_leader:
            token_ids_np = np.array(token_ids, dtype=np.int64)
            prefetch_task_id = self.kv_manager.prefetch_async(
                token_ids=token_ids_np,
            )
            logger.debug(f"[FlexKV] prefetch: launched task_id={prefetch_task_id}")

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"task_id": prefetch_task_id},
            )
            prefetch_task_id = data["task_id"]

        if prefetch_task_id >= 0:
            self._ongoing_prefetches[rid] = prefetch_task_id

    def check_prefetch_progress(self, rid: str) -> bool:
        if not self._prefetch_enabled:
            return True

        prefetch_task_id = self._ongoing_prefetches.get(rid, -1)
        if prefetch_task_id < 0:
            return True

        is_completed = False
        loaded_tokens = 0
        if self._sync_ctx.is_sync_leader:
            completed = self.kv_manager.try_wait(task_ids=[prefetch_task_id])
            if prefetch_task_id in completed:
                response = completed[prefetch_task_id]
                status = response.status
                if status != KVResponseStatus.SUCCESS:
                    logger.warning(
                        "[FlexKV] prefetch task %d for rid=%s finished with status=%s",
                        prefetch_task_id,
                        rid,
                        status,
                    )
                else:
                    # return_mask is a 1D bool ndarray of length len(token_ids);
                    # True = token was successfully staged to CPU/SSD/remote.
                    mask = response.return_mask
                    if mask is not None and hasattr(mask, "sum"):
                        loaded_tokens = int(mask.sum())
                is_completed = True

        if self._sync_ctx.needs_sync:
            data = self._sync_ctx.scatter(
                {"is_completed": is_completed, "loaded_tokens": loaded_tokens},
            )
            is_completed = data["is_completed"]
            loaded_tokens = data["loaded_tokens"]

        if is_completed:
            self._ongoing_prefetches.pop(rid, None)
            # Stash the loaded count even when it is zero so a subsequent
            # pop_prefetch_loaded_tokens call returns deterministically and
            # we don't double-poll try_wait. Page-align so it reflects what
            # the inner radix cache can actually reuse.
            if loaded_tokens > 0 and self.page_size > 1:
                loaded_tokens = (loaded_tokens // self.page_size) * self.page_size
            self._prefetch_loaded_tokens[rid] = loaded_tokens
            if loaded_tokens > 0:
                logger.debug(
                    "[FlexKV] prefetch for rid=%s staged %d tokens to host",
                    rid,
                    loaded_tokens,
                )
        return is_completed

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        """Return the host-staged token count and clear the per-request entry.

        Populated by :meth:`check_prefetch_progress` once the underlying FlexKV
        prefetch task succeeds. Returns 0 if no prefetch ran for ``rid`` or the
        value was already popped.
        """
        return self._prefetch_loaded_tokens.pop(rid, 0)

    def cancel_prefetch(self, rid: str) -> None:
        self._pending_loads.pop(rid, None)
        self._pending_swa_token_ids.pop(rid, None)
        self._prefetch_loaded_tokens.pop(rid, None)
        prefetch_task_id = self._ongoing_prefetches.pop(rid, -1)
        if self._sync_ctx.is_sync_leader and prefetch_task_id >= 0:
            # Flexkv not support cancel prefetch task yet
            pass

    @property
    def layer_done_counter(self) -> Any:
        return self._layer_done_counter

    def register_layer_transfer_counter(self, kvcache: Any) -> None:
        if self._layer_done_counter is not None:
            kvcache.register_layer_transfer_counter(self._layer_done_counter)

    def reset(self) -> None:
        if self._sync_ctx.is_sync_leader and self._pending_loads:
            pending_tids = [tid for tid in self._pending_loads.values() if tid >= 0]
            if pending_tids:
                self.kv_manager.cancel(pending_tids)
        self._pending_loads.clear()
        self._pending_swa_token_ids.clear()
        self._ongoing_prefetches.clear()
        self._prefetch_loaded_tokens.clear()
        self._ongoing_loads.clear()
        self._completed_loads.clear()
        self._load_fkv_tids.clear()

        if self._sync_ctx.is_sync_leader:
            for fk_tid in list(self._ongoing_stores.values()):
                if fk_tid >= 0:
                    self._wait_flexkv_task(fk_tid)
        self._ongoing_stores.clear()
        self._completed_stores.clear()

        if self._layer_done_counter is not None:
            self._layer_done_counter.reset()

    def shutdown(self) -> None:
        if self._sync_ctx.is_sync_leader:
            self.kv_manager.shutdown()

        # Shutdown TransferManagerOnRemote process on Node B
        if self._remote_process is not None:
            try:
                self._remote_process.terminate()
                self._remote_process.join(timeout=5.0)
                if self._remote_process.is_alive():
                    logger.warning(
                        f"[FlexKV] TransferManagerOnRemote did not terminate gracefully, "
                        f"killing{self._rank_label}")
                    self._remote_process.kill()
                    self._remote_process.join()
            except Exception as e:
                logger.warning(
                    f"[FlexKV] Error shutting down TransferManagerOnRemote{self._rank_label}: {e}")
            self._remote_process = None

    # ---- SWA (Sliding Window Attention) Integration ----

    def _do_swa_restore_for_op(self, op: LoadOperation) -> None:
        """Restore SWA data from FlexKV CPU pool to GPU for a single load operation.

        Retrieves the pending SWA token_ids for this request, fetches the SWA data
        from the KVManager, and writes it back to the GPU SWA pool.
        """
        token_ids = self._pending_swa_token_ids.pop(op.rid, None)
        if token_ids is None:
            return

        if not self._sync_ctx.is_sync_leader:
            return

        if not hasattr(self.kv_manager, 'swa_get'):
            return

        try:
            swa_data = self.kv_manager.swa_get(token_ids)
            if swa_data is None:
                logger.debug(
                    f"[FlexKV-SWA] No SWA data returned for rid={op.rid}, "
                    f"prefix_len={len(token_ids)}"
                )
                return

            self._restore_swa_to_gpu(op.device_indices, swa_data)
            logger.debug(
                f"[FlexKV-SWA] Restored SWA to GPU for rid={op.rid}, "
                f"prefix_len={len(token_ids)}"
            )
        except Exception as e:
            logger.debug(f"[FlexKV-SWA] SWA restore failed for rid={op.rid}: {e}")

    def _translate_full_to_swa(self, full_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """Map full-pool token indices to SWA-pool token indices.

        DSV4 routes through ``kvcache.translate_loc_from_full_to_swa`` (returns
        int32). Legacy SWA pools may expose the same method on the sub-pool.
        Falls back to identity for caches with a 1:1 mapping.
        """
        if hasattr(self._kvcache, 'translate_loc_from_full_to_swa'):
            return self._kvcache.translate_loc_from_full_to_swa(full_indices)
        if hasattr(self._swa_kv_pool, 'translate_loc_from_full_to_swa'):
            return self._swa_kv_pool.translate_loc_from_full_to_swa(full_indices)
        return full_indices

    def _swa_page_view(self, layer_buf: torch.Tensor) -> torch.Tensor:
        """View a 2D page-flat SWA buffer as (num_pages, page_size, bytes_per_token).

        DSV4's ``swa_kv_pool.kv_buffer[layer]`` is shaped
        ``(num_pages, bytes_per_page_padded)`` with the trailing bytes
        zero-padded to a 576-byte boundary. The "unpadded" prefix carries
        ``page_size * bytes_per_token`` bytes, which is what FlexKV stores.
        """
        page_size = self._swa_page_size
        bpt = self._swa_bytes_per_token_per_layer
        return layer_buf[:, : page_size * bpt].view(layer_buf.shape[0], page_size, bpt)

    def _restore_swa_to_gpu(
        self,
        full_indices: torch.Tensor,
        swa_cpu_data: torch.Tensor,
    ) -> None:
        """Restore SWA data from CPU to the GPU SWA sub-pool.

        Expects ``swa_cpu_data`` shaped ``(num_swa_layers, window, bytes_per_token)``
        — the same layout produced by :meth:`_extract_swa_from_gpu`.
        """
        if self._swa_kv_pool is None or self._swa_window_size <= 0:
            return
        if self._swa_page_size <= 0 or self._swa_bytes_per_token_per_layer <= 0:
            return

        window = min(self._swa_window_size, len(full_indices))
        if window <= 0:
            return

        last_full = full_indices[-window:]
        swa_indices = self._translate_full_to_swa(last_full)
        if swa_indices is None:
            return
        swa_indices = swa_indices.to(device=self._device, dtype=torch.long)

        page_size = self._swa_page_size
        bpt = self._swa_bytes_per_token_per_layer
        page_idx = swa_indices // page_size
        intra = swa_indices % page_size

        layers = self._swa_kv_pool.kv_buffer
        num_swa_layers = len(layers)

        # Accept (L, W, bpt) or a flat (L * W * bpt,) byte blob.
        if swa_cpu_data.dim() == 1:
            expected = num_swa_layers * window * bpt
            if swa_cpu_data.numel() < expected:
                logger.debug(
                    f"[FlexKV-SWA] restore size mismatch: got {swa_cpu_data.numel()}, "
                    f"expected {expected}"
                )
                return
            swa_cpu_data = swa_cpu_data[:expected].view(num_swa_layers, window, bpt)
        elif swa_cpu_data.shape != (num_swa_layers, window, bpt):
            logger.debug(
                f"[FlexKV-SWA] restore shape mismatch: got {tuple(swa_cpu_data.shape)}, "
                f"expected {(num_swa_layers, window, bpt)}"
            )
            return

        gpu_data = swa_cpu_data.to(device=self._device, dtype=torch.uint8, non_blocking=True)
        for layer_id in range(num_swa_layers):
            page_view = self._swa_page_view(layers[layer_id])
            page_view[page_idx, intra] = gpu_data[layer_id]
        torch.cuda.synchronize()

    def _extract_swa_from_gpu(self, kv_indices: torch.Tensor) -> Optional["torch.Tensor"]:
        """Extract trailing-window SWA bytes from the DSV4 GPU SWA sub-pool.

        Returns a CPU ``uint8`` tensor shaped
        ``(num_swa_layers, window, bytes_per_token)``.
        """
        if self._swa_kv_pool is None or self._swa_window_size <= 0:
            return None
        if self._swa_page_size <= 0 or self._swa_bytes_per_token_per_layer <= 0:
            return None

        window = min(self._swa_window_size, len(kv_indices))
        if window <= 0:
            return None

        try:
            last_full = kv_indices[-window:]
            swa_indices = self._translate_full_to_swa(last_full)
            if swa_indices is None:
                return None
            swa_indices = swa_indices.to(device=self._device, dtype=torch.long)

            page_size = self._swa_page_size
            page_idx = swa_indices // page_size
            intra = swa_indices % page_size

            per_layer = []
            for layer_buf in self._swa_kv_pool.kv_buffer:
                page_view = self._swa_page_view(layer_buf)
                per_layer.append(page_view[page_idx, intra])
            return torch.stack(per_layer, dim=0).contiguous().cpu()
        except Exception as e:
            logger.debug(f"[FlexKV-SWA] _extract_swa_from_gpu failed: {e}")
            return None

    # ---- Private helpers ----

    def _wait_flexkv_task(self, fk_task_id: int, timeout: float = 20.0) -> bool:
        if fk_task_id < 0 or not self._sync_ctx.is_sync_leader:
            return True
        try:
            response = self.kv_manager.wait([fk_task_id], timeout=timeout)
            return (
                fk_task_id in response
                and response[fk_task_id].status == KVResponseStatus.SUCCESS
            )
        except Exception as e:
            logger.error("[FlexKV] wait task failed: %s", e, exc_info=True)
            return False

    def _register_with_retry(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
        max_retries: int = 360,
    ) -> None:
        """Register GPU with retry for Node B (wait for TransferManagerOnRemote).

        Node B's non-leader ranks may attempt to register before
        TransferManagerOnRemote has finished initializing. This method
        retries the registration up to ``max_retries`` times (default 360,
        i.e. 6 minutes at 1 s intervals).
        """
        for attempt in range(max_retries):
            try:
                self._register_to_server(kv_caches, indexer_buffers)
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                if attempt % 30 == 0:
                    logger.info(
                        f"[FlexKV] GPU register retry{self._rank_label}: "
                        f"attempt={attempt+1}/{max_retries}, error={e}"
                    )
                time.sleep(1.0)

    # ---- DSV4 multi-group helpers ----

    @staticmethod
    def _is_dsv4_pool(kvcache: Any) -> bool:
        """Detect DeepSeekV4TokenToKVPool by structural duck-typing.

        Avoids importing the sglang DSV4 class at module load time (the
        connector is loaded in any backend, not just DSV4).
        """
        return (
            hasattr(kvcache, "c4_kv_pool")
            and hasattr(kvcache, "c128_kv_pool")
            and hasattr(kvcache, "c4_indexer_kv_pool")
            and hasattr(kvcache, "swa_kv_pool")
            and hasattr(kvcache, "compression_ratios")
        )

    def _collect_dsv4_registration(self, kvcache: Any) -> Dict[str, Any]:
        """Collect c4 / c128 / indexer sub-pools as a multi-group spec.

        DSV4 splits its attention layers into three groups by compression
        ratio (CSA 4x, HCA 128x, indexer-only 4x). Each sub-pool keeps its
        own ``(num_pages, bytes_per_page)`` GPU tensor with the same
        ``num_pages`` as the full pool (full_token/(4·page_size/4) ==
        full_token/(128·page_size/128) == full_token/page_size), so a
        single full slot_mapping addresses the right page in every group
        and FlexKV's per-group ``compress_ratio`` shrinks only the
        per-block token count.

        The SWA sub-pool has an independent ``num_pages`` (``swa_size /
        swa_page_size``) that does NOT align with the full num_pages, so
        feeding the full slot_mapping into the main launch path would
        index past the end of the SWA tensor → CUDA illegal memory
        access. SWA is therefore intentionally excluded here and handled
        by the side channel (``swa_put`` / ``swa_get`` +
        ``translate_loc_from_full_to_swa``) in
        ``_extract_swa_from_gpu`` / ``_do_swa_restore_for_op``.

        Returns a dict with three aligned lists (``layer_groups``,
        ``handles_per_group``, ``gpu_layouts``) — index 0 is c4, 1 is
        c128, 2 is c4-indexer.
        """
        stage_start = kvcache._stage_start
        stage_end = kvcache._stage_end
        ratios = kvcache.compression_ratios
        stage_ratios = ratios[stage_start:stage_end]

        c4_local_indices: List[int] = []
        c128_local_indices: List[int] = []
        for local_idx, r in enumerate(stage_ratios):
            if r == 4:
                c4_local_indices.append(local_idx)
            elif r == 128:
                c128_local_indices.append(local_idx)

        c4_pool = kvcache.c4_kv_pool
        c128_pool = kvcache.c128_kv_pool
        idx_pool = kvcache.c4_indexer_kv_pool

        def _kv_layout_from_pool(pool: Any, num_layers: int) -> KVCacheLayout:
            # DSV4 SingleKVPool: kv_buffer[layer].shape == (num_pages, bytes_per_page_padded).
            # Treat the whole flat page as one "head" of size
            # bytes_per_page // page_size so block_stride covers an entire page.
            num_pages, bytes_per_page = pool.kv_buffer[0].shape
            head_size = bytes_per_page // pool.page_size
            return KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=num_layers,
                num_block=num_pages,
                tokens_per_block=pool.page_size,
                num_head=1,
                head_size=head_size,
                is_mla=True,
            )

        c4_layout = _kv_layout_from_pool(c4_pool, num_layers=len(c4_local_indices))
        c128_layout = _kv_layout_from_pool(c128_pool, num_layers=len(c128_local_indices))

        # Indexer: 2D (num_pages, page_stride_bytes). A page holds
        # idx_pool.page_size entries (== c4 entries-per-page), so present it
        # the SAME way as c4/c128: tokens_per_block = page_size and
        # head_size = per-entry bytes. Using tokens_per_block=1 +
        # head_size=whole-page would make head_size carry the entries-per-page
        # factor, which FlexKV's CPU sizing then multiplies again by
        # tpb_g (= global tokens_per_block // compress_ratio), over-allocating
        # the indexer's CPU/SSD region by entries-per-page (64x for DSV4).
        # block_stride (= tokens_per_block * head_size = whole page) is
        # unchanged either way, so GPU addressing is identical.
        idx_buf0 = idx_pool.index_k_with_scale_buffer[0]
        assert idx_buf0.shape[1] % idx_pool.page_size == 0, (
            f"[FlexKV-DSV4] indexer page bytes {idx_buf0.shape[1]} not "
            f"divisible by page_size {idx_pool.page_size}"
        )
        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=len(idx_pool.index_k_with_scale_buffer),
            num_block=idx_buf0.shape[0],
            tokens_per_block=idx_pool.page_size,
            num_head=1,
            head_size=idx_buf0.shape[1] // idx_pool.page_size,
            is_mla=True,
        )

        # FlexKV's LayerGroup protocol requires every group to share the
        # same num_block, because launch() takes a single slot_mapping per
        # task and slot_mapping_to_block_ids reduces it to one block_id
        # vector that is broadcast to all groups. c4 / c128 / indexer
        # satisfy this by construction.

        c128_num_pages = c128_pool.kv_buffer[0].shape[0]
        c4_num_pages = c4_pool.kv_buffer[0].shape[0]
        idx_num_pages = idx_pool.index_k_with_scale_buffer[0].shape[0]
        assert c4_num_pages == c128_num_pages, (
            f"[FlexKV-DSV4] c4_pool num_pages={c4_num_pages} != "
            f"full num_pages={full_num_pages}; FlexKV LayerGroup requires "
            f"identical num_block across groups. Likely cause: "
            f"c4_shrink_factor > 1 (HiSparse host_to_device_ratio)."
        )
        assert idx_num_pages == c128_num_pages, (
            f"[FlexKV-DSV4] indexer_pool num_pages={idx_num_pages} != "
            f"full num_pages={full_num_pages}; FlexKV LayerGroup requires "
            f"identical num_block across groups."
        )

        layer_groups = [
            LayerGroupSpec(
                num_layers=len(c4_local_indices),
                num_kv_heads=1,
                head_size=c4_layout.head_size,
                layer_indices=list(c4_local_indices),
                compress_ratio=4,
                dtype=c4_pool.kv_buffer[0].dtype,
            ),
            LayerGroupSpec(
                num_layers=len(c128_local_indices),
                num_kv_heads=1,
                head_size=c128_layout.head_size,
                layer_indices=list(c128_local_indices),
                compress_ratio=128,
                dtype=c128_pool.kv_buffer[0].dtype,
            ),
            LayerGroupSpec(
                num_layers=len(c4_local_indices),
                num_kv_heads=1,
                head_size=indexer_layout.head_size,
                layer_indices=list(c4_local_indices),
                compress_ratio=4,
                dtype=idx_pool.index_k_with_scale_buffer[0].dtype,
            ),
        ]

        handles_per_group: List[List[torch.Tensor]] = [
            list(c4_pool.kv_buffer),
            list(c128_pool.kv_buffer),
            list(idx_pool.index_k_with_scale_buffer),
        ]
        gpu_layouts: List[KVCacheLayout] = [
            c4_layout,
            c128_layout,
            indexer_layout,
        ]

        for _name, _buf, _ps, _lay in (
            ("c4", c4_pool.kv_buffer[0], c4_pool.page_size, c4_layout),
            ("c128", c128_pool.kv_buffer[0], c128_pool.page_size, c128_layout),
            ("indexer", idx_pool.index_k_with_scale_buffer[0],
             idx_pool.page_size, indexer_layout),
        ):
            _num_pages, _bytes_per_page = _buf.shape
            _block_stride = _lay.tokens_per_block * _lay.head_size
            logger.info(
                "[FlexKV-DSV4-DIAG] group=%s page_size=%d num_pages=%d "
                "bytes_per_page=%d head_size=%d divisible=%s "
                "block_stride=%d real_page_bytes=%d stride_ok=%s",
                _name, _ps, _num_pages, _bytes_per_page,
                _lay.head_size, (_bytes_per_page % _ps == 0) if _ps else False,
                _block_stride, _bytes_per_page, _block_stride == _bytes_per_page,
            )

        return {
            "layer_groups": layer_groups,
            "handles_per_group": handles_per_group,
            "gpu_layouts": gpu_layouts,
        }

    def _register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        indexer_buffers: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """Register GPU KV cache buffers to FlexKV server.

        Args:
            kv_caches: Unified KV cache tensor list.
                - MLA: num_layer tensors (K and V share the same buffer).
                - MHA: 2 * num_layer tensors (K buffers followed by V buffers).
            indexer_buffers: Optional sparse attention indexer buffers.
        """
        assert len(kv_caches) > 0
        logger.info(
            f"[FlexKV-DEBUG] _register_to_server{self._rank_label}: "
            f"len(kv_caches)={len(kv_caches)}, "
            f"kv_caches[0].shape={tuple(kv_caches[0].shape)}, "
            f"kv_caches[0].dtype={kv_caches[0].dtype}, "
            f"kv_caches[0].stride={tuple(kv_caches[0].stride())}, "
            f"kv_caches[0].element_size={kv_caches[0].element_size()}, "
            f"page_size={self.page_size}"
        )

        # DSV4 multi-group path: register c4 / c128 / indexer / swa as
        # separate LayerGroupSpec entries. The legacy single-group code
        # below assumes a homogeneous (num_kv_heads, head_size) shape and
        # cannot describe the four heterogeneous sub-pools.
        if self._dsv4_registration is not None:
            reg = self._dsv4_registration
            self.tp_client.register_to_server(
                kv_caches=reg["handles_per_group"][0],
                kv_layout=reg["gpu_layouts"][0],
                layer_groups=reg["layer_groups"],
                gpu_layouts=reg["gpu_layouts"],
                handles_per_group=reg["handles_per_group"],
            )
            logger.info(
                "[FlexKV-DSV4] Registered %d sub-pool groups to FlexKV server",
                len(reg["layer_groups"]),
            )
            return

        assert kv_caches[0].ndim in (2, 3), (
            f"Expected 2D (DSV4 page-flat) or 3D KV tensor, "
            f"got shape={kv_caches[0].shape}"
        )

        is_mla = self.flexkv_config.model_config.use_mla
        if kv_caches[0].ndim == 3:
            num_blocks, num_kv_heads, head_size = kv_caches[0].shape
        else:
            # DSV4 SingleKVPool: (num_pages, bytes_per_page_padded). Each
            # page already encodes page_size tokens; treat the whole flattened
            # page as one head of size bytes_per_page // page_size so that
            # block_stride covers a full page.
            num_blocks_pages, bytes_per_page = kv_caches[0].shape
            num_kv_heads = 1
            head_size = bytes_per_page // self.page_size
            # Convert to "slots" units used by the LAYERFIRST layout below
            # (num_block = num_blocks // page_size).
            num_blocks = num_blocks_pages * self.page_size

        # GPU layout uses page_size as tokens_per_block so that the transfer
        # engine's block_stride covers an entire page of tokens.  The physical
        # GPU tensor shape is [num_blocks, num_kv_heads, head_size] where each
        # slot stores 1 token, but we present it to FlexKV as
        # [num_blocks/page_size, page_size, num_kv_heads, head_size] so that
        # block_id * block_stride correctly addresses the start of a page.

        gpu_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=self.rank_info.num_layers_per_pp_stage,
            num_block=num_blocks // self.page_size,
            tokens_per_block=self.page_size,
            num_head=num_kv_heads,
            head_size=head_size,
            is_mla=is_mla,
        )

        # Build indexer layout if indexer buffers are present
        indexer_layout = None
        if indexer_buffers is not None and len(indexer_buffers) > 0:
            indexer_tensor = indexer_buffers[0]
            assert indexer_tensor.ndim == 2, (
                f"Expected 2D indexer tensor (num_pages, page_stride_size), "
                f"got shape={indexer_tensor.shape}"
            )
            # sglang's NSA indexer buffer is 2D: (num_pages, page_stride_size),
            # where page_stride_size = page_size * (index_head_dim + scale_bytes).
            # All tokens within a page are flattened into a single contiguous
            # vector, so from FlexKV's perspective each page is one indivisible
            # block with tokens_per_block=1.  The resulting block_stride
            # (= 1 * 1 * page_stride_size) correctly addresses each page.
            indexer_layout = KVCacheLayout(
                type=KVCacheLayoutType.LAYERFIRST,
                num_layer=len(indexer_buffers),
                num_block=indexer_tensor.shape[0],
                tokens_per_block=1,
                num_head=1,
                head_size=indexer_tensor.shape[1],
                is_mla=True,
            )
            logger.debug(
                "[FlexKV] Indexer layout: num_layer=%d, num_block=%d, "
                "tokens_per_block=%d, head_size=%d",
                len(indexer_buffers), indexer_tensor.shape[0],
                1, indexer_tensor.shape[1],
            )
            # Consistency check: indexer num_block should equal main KV num_block
            # (1:1 mapping since tokens_per_block = page_size)
            indexer_config = self.flexkv_config.cache_config.indexer
            if indexer_config is not None:
                expected_indexer_blocks = num_blocks // self.page_size
                assert indexer_tensor.shape[0] == expected_indexer_blocks, (
                    f"[FlexKV] Indexer num_block mismatch: indexer has {indexer_tensor.shape[0]} pages, "
                    f"but main KV has {num_blocks} slots / page_size {self.page_size} "
                    f"= {expected_indexer_blocks} expected blocks"
                )

        # Register KV caches (and optional indexer buffers) to FlexKV server
        self.tp_client.register_to_server(
            kv_caches=kv_caches,
            kv_layout=gpu_layout,
            indexer_buffers=indexer_buffers,
            indexer_layout=indexer_layout,
        )
        logger.info("[FlexKV] Registered KV caches to server")

    def _init_layer_transfer_components(self):
        if not self.enable_layerwise_transfer:
            self._layer_done_counter = None
            self._worker_connected = False
            logger.debug(f"[FlexKV] Layerwise transfer disabled{self._rank_label}")
            return

        self._layer_done_counter = FlexKVLayerDoneCounter(self.rank_info.num_layers_per_pp_stage)
        self._send_eventfds_to_worker()
        logger.info(f"[FlexKV] Initialized layerwise transfer{self._rank_label}")

    def send_slot_mapping_to_remote(self, task_id: int, slot_mapping: torch.Tensor) -> None:
        """Send slot_mapping to TransferManagerOnRemote via existing ZMQ channel (PP1 side only).

        In cross-node PP mode, PP1's FlexKVConnector sends slot_mapping
        via KVTPClient.send_to_server -> TransferManagerOnRemote.command_socket,
        so that the remote side can call set_gpu_blocks() with its own local GPU block_ids.
        """
        slot_mapping_np = slot_mapping.cpu().to(torch.int64).numpy() if slot_mapping.is_cuda else slot_mapping.numpy()
        self.tp_client.set_slot_mapping(task_id, slot_mapping_np)
        logger.debug(
            f"[FlexKV] send_slot_mapping_to_remote: "
            f"sent task_id={task_id} to TransferManagerOnRemote"
        )

    def _send_eventfds_to_worker(
        self, retry_interval: float = 1.0
    ):
        max_retries = self.layerwise_eventfd_connect_max_retries
        # Allow up to 3 full connect+send attempts before giving up.
        max_send_retries = 3
        logger.info(
            f"[FlexKV] Attempting eventfd connection{self._rank_label}: "
            f"socket={self.layerwise_eventfd_socket}, max_retries={max_retries}")

        last_error = None
        for send_attempt in range(max_send_retries):
            sock = None
            try:
                # Phase 1: Connect to the worker socket (retry until ready).
                for attempt in range(max_retries):
                    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    try:
                        sock.connect(self.layerwise_eventfd_socket)
                        logger.info(
                            f"[FlexKV] Eventfd connected{self._rank_label}: "
                            f"socket={self.layerwise_eventfd_socket}, "
                            f"attempts={attempt + 1}"
                            f"{f', send_retry={send_attempt}' if send_attempt > 0 else ''}"
                        )
                        break
                    except (FileNotFoundError, ConnectionRefusedError) as e:
                        sock.close()
                        sock = None
                        if attempt == max_retries - 1:
                            logger.error(
                                f"[FlexKV] Eventfd connection failed{self._rank_label}: "
                                f"socket={self.layerwise_eventfd_socket}, "
                                f"attempts={max_retries}, error={type(e).__name__}"
                            )
                            raise RuntimeError(
                                f"[FlexKV] Failed to connect to eventfd socket "
                                f"{self.layerwise_eventfd_socket} after {max_retries} attempts"
                            )
                        if attempt % 10 == 0:
                            socket_exists = os.path.exists(self.layerwise_eventfd_socket)
                            logger.debug(
                                f"[FlexKV] Eventfd connect retry{self._rank_label}: "
                                f"socket={self.layerwise_eventfd_socket}, "
                                f"attempt={attempt + 1}/{max_retries}, "
                                f"error={type(e).__name__}, socket_exists={socket_exists}"
                            )
                        time.sleep(retry_interval)

                if sock is None:
                    raise RuntimeError(
                        f"[FlexKV] Eventfd socket unavailable after {max_retries} attempts: "
                        f"{self.layerwise_eventfd_socket}"
                    )

                # Phase 2: Send metadata + eventfds over the connected socket.
                # UDS is node-local, so use _per_node TP rank/size so that
                # LayerwiseWorker builds the correct eventfd tensor shape.
                num_counters = self._layer_done_counter.num_counters
                model_config = self.flexkv_config.model_config
                rank_info = self.rank_info
                # Send 16-byte metadata: tp_rank_per_node, tp_size_per_node, num_layers, num_counters
                metadata = struct.pack(
                    "iiii",
                    rank_info.tp_rank_per_node,
                    model_config.tp_size_per_node,
                    rank_info.num_layers_per_pp_stage,
                    num_counters,
                )
                sock.sendall(metadata)
                logger.debug(
                    f"[FlexKV] Eventfd metadata sent{self._rank_label}: "
                    f"tp_rank_per_node={rank_info.tp_rank_per_node}, "
                    f"tp_size_per_node={model_config.tp_size_per_node}, "
                    f"num_layers={rank_info.num_layers_per_pp_stage}, num_counters={num_counters}"
                )

                for counter_id in range(num_counters):
                    fds = self._layer_done_counter.events[counter_id].load_event_fds
                    send_fds(sock, fds, struct.pack("i", counter_id))
                    logger.debug(
                        f"[FlexKV] Eventfd fds sent{self._rank_label}: "
                        f"counter_id={counter_id}, num_fds={len(fds)}"
                    )

                # Wait for ACK from server to confirm fds were received.
                sock.settimeout(30.0)
                try:
                    ack = sock.recv(1)
                except socket.timeout:
                    raise RuntimeError("Timed out waiting for ACK from FlexKV worker")
                if not ack or ack[0] != 1:
                    raise RuntimeError(
                        f"FlexKV worker NACK'd eventfd transfer (ack={ack!r})"
                    )

                self._worker_connected = True
                logger.info(
                    f"[FlexKV] Eventfd setup complete{self._rank_label}: "
                    f"socket={self.layerwise_eventfd_socket}, "
                    f"counters={num_counters}, layers={rank_info.num_layers_per_pp_stage}"
                )
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[FlexKV] Failed to send eventfds{self._rank_label} "
                    f"(send_attempt {send_attempt + 1}/{max_send_retries}): "
                    f"socket={self.layerwise_eventfd_socket}, error={e}. "
                    f"Will reconnect and retry..."
                )
            finally:
                if sock is not None:
                    sock.close()
                # Brief pause before reconnecting.
                time.sleep(retry_interval)

        # All send retries exhausted.
        logger.error(
            f"[FlexKV] Failed to send eventfds{self._rank_label} after "
            f"{max_send_retries} attempts: "
            f"socket={self.layerwise_eventfd_socket}, last_error={last_error}",
            exc_info=True,
        )
        raise RuntimeError(
            f"[FlexKV] Failed to send eventfds to {self.layerwise_eventfd_socket} "
            f"after {max_send_retries} attempts: {last_error}"
        )
