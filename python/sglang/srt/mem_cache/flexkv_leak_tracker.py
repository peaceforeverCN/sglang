"""
FlexKV / SGLang leak tracker.

Records every alloc / free / tree-lock-change / flexkv-task-lifecycle event
with a rid + source label. On leak (invariant checker fires), dumps the
outstanding events so we can pin down which rid + which code path leaked.

Enable via: export SGLANG_FLEXKV_LEAK_TRACE=1
Optional env: SGLANG_FLEXKV_LEAK_TRACE_RING=8000  (ring size, default 5000)
"""

from __future__ import annotations

import contextvars
import logging
import os
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger("flexkv_leak_tracker")


def _enabled() -> bool:
    return os.environ.get("SGLANG_FLEXKV_LEAK_TRACE", "0") == "1"


ENABLED = _enabled()
_RING = int(os.environ.get("SGLANG_FLEXKV_LEAK_TRACE_RING", "5000"))

# ----- current op context (thread-local + contextvar) -----
# Codepath sets this before calling alloc / free so the event carries a rid.
_current_ctx: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "flexkv_leak_current_ctx", default=None
)


class push_ctx:
    """Push a context dict for the duration of a with-block. Nested pushes stack."""

    def __init__(self, **fields):
        self.fields = fields
        self.token = None

    def __enter__(self):
        parent = _current_ctx.get() or {}
        merged = dict(parent)
        merged.update(self.fields)
        self.token = _current_ctx.set(merged)
        return merged

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_ctx.reset(self.token)


def get_ctx() -> Dict[str, Any]:
    return _current_ctx.get() or {}


# ----- ring buffers -----
_lock = threading.Lock()

# alloc events: (t, kind, source, size, page_min, page_max, ctx_dict)
_alloc_events: Deque[Tuple] = deque(maxlen=_RING)
# free events
_free_events: Deque[Tuple] = deque(maxlen=_RING)
# tree lock events: (t, action, node_id, size_delta, tag, ctx)
_lock_events: Deque[Tuple] = deque(maxlen=_RING)
# flexkv events: (t, kind, task_id, fkv_task_id, extra, ctx)
_flexkv_events: Deque[Tuple] = deque(maxlen=_RING)

# aggregated running totals per (kind, source) -> total_pages
_alloc_totals: Dict[str, int] = defaultdict(int)
_free_totals: Dict[str, int] = defaultdict(int)

# per-req allocation/free (rid) -> total pages
_alloc_by_rid: Dict[str, int] = defaultdict(int)
_free_by_rid: Dict[str, int] = defaultdict(int)

# outstanding pages (not yet observed as freed): rough computation from indices
# key: page_idx (int) -> (t, kind, source, ctx snapshot)
_page_owners: Dict[int, Tuple[float, str, str, Dict[str, Any]]] = {}


def _tensor_page_stats(indices: torch.Tensor, page_size: int):
    if indices is None:
        return 0, None, None, []
    if not isinstance(indices, torch.Tensor):
        return 0, None, None, []
    if indices.numel() == 0:
        return 0, None, None, []
    try:
        cpu_ind = indices.detach().to("cpu", copy=False)
    except Exception:
        cpu_ind = indices
    try:
        pages = torch.unique(cpu_ind // page_size)
    except Exception:
        return int(cpu_ind.numel()), None, None, []
    n = int(pages.numel())
    if n == 0:
        return 0, None, None, []
    pmin = int(pages.min().item())
    pmax = int(pages.max().item())
    # Cap the returned page list for _page_owners bookkeeping: 32k pages
    # is enough for post-mortem inspection (max_total_num_tokens/page_size is
    # ~2500 for the user's config).
    return n, pmin, pmax, pages.tolist()


def record_alloc(kind: str, source: str, indices: torch.Tensor, page_size: int, extra: Optional[dict] = None):
    if not ENABLED:
        return
    n, pmin, pmax, pages = _tensor_page_stats(indices, page_size)
    ctx = get_ctx()
    with _lock:
        _alloc_events.append((time.time(), kind, source, n, pmin, pmax, dict(ctx), extra or {}))
        _alloc_totals[f"{kind}:{source}"] += n
        rid = ctx.get("rid")
        if rid:
            _alloc_by_rid[rid] += n
        for p in pages:
            _page_owners[int(p)] = (time.time(), kind, source, dict(ctx))


def record_free(kind: str, source: str, indices: torch.Tensor, page_size: int, extra: Optional[dict] = None):
    if not ENABLED:
        return
    n, pmin, pmax, pages = _tensor_page_stats(indices, page_size)
    ctx = get_ctx()
    with _lock:
        _free_events.append((time.time(), kind, source, n, pmin, pmax, dict(ctx), extra or {}))
        _free_totals[f"{kind}:{source}"] += n
        rid = ctx.get("rid")
        if rid:
            _free_by_rid[rid] += n
        for p in pages:
            _page_owners.pop(int(p), None)


def record_lock(action: str, node_id: Any, size_delta: int, tag: str = "", extra: Optional[dict] = None):
    if not ENABLED:
        return
    ctx = get_ctx()
    with _lock:
        _lock_events.append((time.time(), action, node_id, size_delta, tag, dict(ctx), extra or {}))


def record_flexkv(kind: str, task_id: int = -1, fkv_task_id: int = -1, extra: Optional[dict] = None):
    if not ENABLED:
        return
    ctx = get_ctx()
    with _lock:
        _flexkv_events.append((time.time(), kind, task_id, fkv_task_id, dict(ctx), extra or {}))


def dump_state(
    header: str,
    tree_cache=None,
    allocator=None,
    req_to_token_pool=None,
    running_batch=None,
    last_batch=None,
    connector=None,
    tail: int = 200,
) -> str:
    """Assemble a textual dump of the tracker state. Returns a big string."""
    lines: List[str] = []
    lines.append("========== [FLEXKV LEAK DUMP] " + header + " ==========")
    lines.append(f"time={time.time()} pid={os.getpid()} tid={threading.get_ident()}")

    # Global totals
    lines.append("---- alloc totals (kind:source -> pages) ----")
    for k in sorted(_alloc_totals):
        lines.append(f"  ALLOC {k}: {_alloc_totals[k]}")
    lines.append("---- free totals ----")
    for k in sorted(_free_totals):
        lines.append(f"  FREE  {k}: {_free_totals[k]}")

    # Per-rid net (alloc - free): non-zero rids are candidates
    lines.append("---- per-rid net (alloc-free), pages, only nonzero ----")
    all_rids = set(_alloc_by_rid) | set(_free_by_rid)
    diffs = []
    for rid in all_rids:
        d = _alloc_by_rid.get(rid, 0) - _free_by_rid.get(rid, 0)
        if d != 0:
            diffs.append((d, rid))
    diffs.sort(key=lambda x: -abs(x[0]))
    for d, rid in diffs[:100]:
        lines.append(f"  rid={rid} alloc={_alloc_by_rid.get(rid,0)} free={_free_by_rid.get(rid,0)} net={d}")

    # Outstanding pages (still in _page_owners) grouped by (kind:source, ctx.rid)
    if _page_owners:
        buckets: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for p, (t, kind, source, ctx) in _page_owners.items():
            buckets[(f"{kind}:{source}", ctx.get("rid", "?"))].append(p)
        lines.append(f"---- outstanding pages (not freed yet), total={len(_page_owners)} ----")
        # Show top 40 buckets by size
        top = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        for (src_rid, pages) in top[:40]:
            pages_sorted = sorted(pages)
            lines.append(
                f"  {src_rid[0]} rid={src_rid[1]} count={len(pages)} "
                f"[first10]={pages_sorted[:10]}"
            )

    # Recent events (tail)
    lines.append(f"---- last {tail} alloc events ----")
    with _lock:
        for ev in list(_alloc_events)[-tail:]:
            lines.append(f"  A t={ev[0]:.3f} kind={ev[1]} src={ev[2]} n={ev[3]} pmin={ev[4]} pmax={ev[5]} ctx={ev[6]} extra={ev[7]}")
    lines.append(f"---- last {tail} free events ----")
    with _lock:
        for ev in list(_free_events)[-tail:]:
            lines.append(f"  F t={ev[0]:.3f} kind={ev[1]} src={ev[2]} n={ev[3]} pmin={ev[4]} pmax={ev[5]} ctx={ev[6]} extra={ev[7]}")
    lines.append(f"---- last {tail} lock events ----")
    with _lock:
        for ev in list(_lock_events)[-tail:]:
            lines.append(f"  L t={ev[0]:.3f} action={ev[1]} node_id={ev[2]} delta={ev[3]} tag={ev[4]} ctx={ev[5]} extra={ev[6]}")
    lines.append(f"---- last {tail} flexkv events ----")
    with _lock:
        for ev in list(_flexkv_events)[-tail:]:
            lines.append(f"  X t={ev[0]:.3f} kind={ev[1]} task_id={ev[2]} fkv_task_id={ev[3]} ctx={ev[4]} extra={ev[5]}")

    # Connector / cache state
    if connector is not None:
        try:
            ol = getattr(connector, "_ongoing_loads", None)
            os_ = getattr(connector, "_ongoing_stores", None)
            cl = getattr(connector, "_completed_loads", None)
            cs = getattr(connector, "_completed_stores", None)
            lines.append("---- connector state ----")
            lines.append(f"  _ongoing_loads (n={len(ol) if ol is not None else 'N/A'}): {list(ol.items())[:20] if ol else []}")
            lines.append(f"  _ongoing_stores (n={len(os_) if os_ is not None else 'N/A'}): {list(os_.items())[:20] if os_ else []}")
            lines.append(f"  _completed_loads (n={len(cl) if cl is not None else 'N/A'}): {list(cl)[:20] if cl else []}")
            lines.append(f"  _completed_stores (n={len(cs) if cs is not None else 'N/A'}): {list(cs)[:20] if cs else []}")
        except Exception as e:
            lines.append(f"  <connector dump failed: {e}>")

    if tree_cache is not None:
        try:
            ort = getattr(tree_cache, "_ongoing_load_tasks", None)
            ost = getattr(tree_cache, "_ongoing_store_tasks", None)
            lq = getattr(tree_cache, "_load_queue", None)
            lines.append("---- tree cache (ExtendedRadixCache) state ----")
            if ort is not None:
                lines.append(f"  _ongoing_load_tasks (n={len(ort)}): {[(k, [(getattr(n,'id',None), u) for (n,u) in v[:5]]) for k,v in list(ort.items())[:10]]}")
            if ost is not None:
                lines.append(f"  _ongoing_store_tasks (n={len(ost)}): {[(k, (getattr(v[0],'id',None), v[1]) if isinstance(v, tuple) else getattr(v,'id',None)) for k,v in list(ost.items())[:10]]}")
            if lq is not None:
                lines.append(f"  _load_queue (n={len(lq)})")
        except Exception as e:
            lines.append(f"  <tree cache dump failed: {e}>")

    if allocator is not None:
        try:
            lines.append("---- allocator state ----")
            for name in ("full_attn_allocator", "swa_attn_allocator"):
                a = getattr(allocator, name, None)
                if a is not None:
                    fp = getattr(a, "free_pages", None)
                    rp = getattr(a, "release_pages", None)
                    lines.append(
                        f"  {name}: size={getattr(a, 'size', None)} "
                        f"free_pages={fp.numel() if fp is not None else None} "
                        f"release_pages={rp.numel() if rp is not None else None}"
                    )
            fp = getattr(allocator, "free_pages", None)
            rp = getattr(allocator, "release_pages", None)
            lines.append(
                f"  outer: size={getattr(allocator, 'size', None)} "
                f"free_pages={fp.numel() if fp is not None else None} "
                f"release_pages={rp.numel() if rp is not None else None}"
            )
        except Exception as e:
            lines.append(f"  <allocator dump failed: {e}>")

    if running_batch is not None or last_batch is not None:
        lines.append("---- reqs in batches ----")
        for name, b in (("running_batch", running_batch), ("last_batch", last_batch)):
            if b is None:
                continue
            try:
                if getattr(b, "is_empty", lambda: True)():
                    lines.append(f"  {name}: empty")
                    continue
                for req in b.reqs[:20]:
                    prefix_n = req.prefix_indices.numel() if getattr(req, "prefix_indices", None) is not None else -1
                    lines.append(
                        f"  {name} rid={getattr(req,'rid',None)} "
                        f"pool_idx={getattr(req,'req_pool_idx',None)} "
                        f"kv_committed_len={getattr(req,'kv_committed_len',None)} "
                        f"kv_allocated_len={getattr(req,'kv_allocated_len',None)} "
                        f"kv_committed_freed={getattr(req,'kv_committed_freed',None)} "
                        f"kv_overallocated_freed={getattr(req,'kv_overallocated_freed',None)} "
                        f"prefix_indices.numel={prefix_n} "
                        f"cache_protected_len={getattr(req,'cache_protected_len',None)} "
                        f"swa_evicted_seqlen={getattr(req,'swa_evicted_seqlen',None)} "
                        f"host_hit_length={getattr(req,'host_hit_length',None)} "
                        f"is_retracted={getattr(req,'is_retracted',None)} "
                        f"finished={req.finished() if hasattr(req,'finished') else '?'} "
                    )
            except Exception as e:
                lines.append(f"  <{name} dump failed: {e}>")

    if req_to_token_pool is not None:
        try:
            fs = getattr(req_to_token_pool, "free_slots", None)
            lines.append(
                f"---- req_to_token_pool: free_slots={fs.numel() if fs is not None else None} "
                f"size={getattr(req_to_token_pool,'size',None)} ----"
            )
        except Exception as e:
            lines.append(f"  <req_to_token_pool dump failed: {e}>")

    lines.append("========== [FLEXKV LEAK DUMP END] ==========")
    return "\n".join(lines)


def emit_dump(header: str, **kwargs):
    """Log the dump via logger.error so it goes into scheduler's stderr/log."""
    if not ENABLED:
        return
    try:
        s = dump_state(header, **kwargs)
        # Split into ~4KB chunks so tqdm doesn't drop lines
        for chunk in _split_chunks(s, 3500):
            logger.error(chunk)
    except Exception as e:
        logger.error(f"[flexkv-leak-tracker] emit_dump raised: {e}", exc_info=True)


def _split_chunks(s: str, size: int):
    for i in range(0, len(s), size):
        yield s[i : i + size]
