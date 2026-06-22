# Debug 记录：FlexKV + DSv4 SWA 的 KV pool 内存泄漏

- **分支**：`feature/flexkv_swa_on_main2`
- **基线 commit**：`4c493f194`
- **日期**：2026-06-10
- **涉及文件**：
  - `python/sglang/srt/mem_cache/swa_radix_cache.py`
  - `python/sglang/srt/mem_cache/extended_radix_cache.py`
  - `python/sglang/srt/managers/scheduler.py`（仅诊断脚手架）

---

## 0. 背景：两个 pool 与 full→swa 映射

DSv4 hybrid-SWA 模型有两类注意力层：

- **full-attention 层**：看完整历史 KV。
- **SWA（sliding-window attention）层**：只看最近 `sliding_window_size` 个 token。

对应两个独立显存池：**full pool**（大）和 **SWA pool**（小，只够 ~window 大小）。

每个 token 在 full pool 占一个 slot，索引存在 radix tree node 的 `value` 里。这个 full slot **可能**还对应一个 SWA pool slot，关系存在全局表 `full_to_swa_index_mapping`：

```
full_to_swa_index_mapping[full_slot] = swa_slot   # >0：有 SWA 备份
full_to_swa_index_mapping[full_slot] = 0          # 没有 SWA 备份
```

**关键点**：tree node 的 `value` 存的是 **full slot 索引**。一个节点"占多少 SWA pool slot"取决于它的 value 里有多少 full slot 的 mapping > 0。

### 泄漏检测的恒等式

scheduler 在 idle 时校验（`invariant_checker._check_pool_invariant`）：

```
available + evictable + protected + session_held + uncached == total
```

任一池不满足即报 `pool memory leak detected`。

---

## 1. 触发改动：`_swa_slots_in_value` 改用 `len(value)`

最初的改动把"一个节点占多少 SWA slot"的计算从**查 live mapping** 改成**直接返回 `len(value)`**：

```python
# 改动前（HEAD）
def _swa_slots_in_value(self, value):
    allocator = self.token_to_kv_pool_allocator
    if hasattr(allocator, "count_mapped_swa_slots"):
        return allocator.count_mapped_swa_slots(value)   # 查 mapping，数 >0 的
    return len(value)

# 改动后
def _swa_slots_in_value(self, value):
    return len(value)
```

**动机**：原先在 `inc_lock_ref` / `dec_lock_ref` 之间，mapping 会被 `dec_swa_lock_only` 的 `free_swa` 就地改写（清零），导致同一节点在加锁和解锁时读到的 SWA slot 数不同 → `swa_protected_size_` 加减不配对 → 恒等式失败。改用 `len(value)` 可消除这种"读取时机依赖"。

**代价**：`len(value)` 只在节点**同质**时才等于真实 SWA slot 数。

### 核心不变量：节点 SWA 同质性

> **每个 non-tombstone 节点都是"同质"的**——它的 value 要么整段都有 SWA 映射，要么整段都没有（没有的标成 `swa_tombstone`，从 SWA 计数排除）。

只要成立：`节点 SWA slot 数 == len(value)`。**整个 SWA 计数体系都建立在这个假设上。** 一旦出现"768 长、却只有 256 个 SWA 映射"的 non-tombstone 节点，假设破裂。

---

## 2. 调试方法论：用断言 / 审计把"猜"变成"看"

整个排查过程的核心策略：**先加诊断、用复现日志拿证据，再改代码**。共加了三层脚手架：

### 2.1 同质性断言

```python
def _assert_swa_homogeneous(self, node: TreeNode, where: str) -> None:
    if node.swa_tombstone:
        return
    allocator = self.token_to_kv_pool_allocator
    if not hasattr(allocator, "count_mapped_swa_slots"):
        return
    mapped = allocator.count_mapped_swa_slots(node.value)
    assert mapped == len(node.value), (
        f"[swa-homogeneity] non-tombstone node has {mapped} mapped SWA "
        f"slots but len(value)={len(node.value)} at {where}; "
        f"{node.id=}, {node.swa_lock_ref=}, {node.full_lock_ref=}"
    )
```

调用点：`inc_lock_ref`、`dec_lock_ref`、以及 `evict` 内部节点分支断言 `swa_freed == len(x.value)`。

### 2.2 evictable 审计（区分两类泄漏）

`audit_evictable()` 遍历整棵 tree，输出：

1. **计数器对账**：`counter` vs `walked`（遍历累加）。不等 → 计数漂移；相等 → 计数自洽。
2. **tree-vs-freelist 双重所有权**：节点 slot 是否出现在 allocator 的 free 列表。
3. **tree-internal 重复**：同一物理 slot 是否被 ≥2 个节点引用（建 `slot→owner` 映射）。
4. **free_slots 内部重复**：double-free 会让同一 id 在 free 列表出现两次。
5. 命中 tree-internal 重复时，**dump 全部节点**（parent / key 长度 / value 长度 / tombstone / lock / value 范围）。

在 `scheduler.on_idle` 报 leak 前调用，把报告打进日志。

---

## 3. 四个 bug（按发现顺序）

四个 bug 层层递进——每修一个，断言/审计就把下一个更深的暴露出来。

### Bug #1：revive 路径制造非同质节点

**现象**：`swa_evictable` 涨到 8192，远超 SWA pool 总量 1536（物理不可能）；**无界累积**。

**根因**：`_insert_helper` 的 revive 分支（Branch 1/2）把一个 tombstone 节点重新激活时：

```python
node.value = value[:prefix_len].clone()
node.swa_tombstone = False
self.swa_evictable_size_ += self._swa_slots_in_value(node.value)  # += len(value)
```

但它**从没给这段新 value 重建 SWA 映射**（节点被 tombstone 时 `free_swa` 已把 mapping 清零）。于是真实 mapping=0、却 `+= len` → 非同质 non-tombstone 节点。

**口径不对称导致单向累积**：

| 操作 | 口径 |
|---|---|
| revive / `_add_new_node` 加 evictable | `len(value)` |
| `evict` / `dec_swa_lock_only` 减 | `free_swa()` 真实返回值 |

加 `len`、减真实值 → `swa_evictable_size_` 只增不平 → 累积到 8192。

**修复方向**：revive 不再相信 `len`，**只激活真正有映射的尾段，其余保持 tombstone**。

```python
def _mapped_swa_tail_len(self, value: torch.Tensor) -> int:
    """value 中连续 SWA-mapped 后缀的长度（insert/revive 时刻读一次 mapping）。"""
    allocator = self.token_to_kv_pool_allocator
    mapping = getattr(allocator, "full_to_swa_index_mapping", None)
    if mapping is None or len(value) == 0:
        return len(value)
    mapped = mapping[value] > 0
    total = int(mapped.sum().item())
    if total == 0 or total == len(value):
        return total
    rev = mapped.flip(0)
    first_unmapped = (~rev).nonzero()
    tail = int(first_unmapped[0].item()) if first_unmapped.numel() else len(value)
    assert tail == total, (
        f"[swa] non-suffix SWA mapping in revive: {tail=} {total=} {len(value)=}"
    )
    return tail

def _revive_tombstone_tail(self, node: TreeNode, new_value: torch.Tensor) -> None:
    """只激活 tombstone 节点中 SWA-mapped 的尾段，head 保持 tombstone。"""
    node.value = new_value.clone()
    mapped_tail = self._mapped_swa_tail_len(node.value)
    if mapped_tail == 0:
        return  # 整段无映射，保持 tombstone
    if mapped_tail < len(node.value):
        split_at = len(node.value) - mapped_tail
        self._split_node(node.key, node, split_at)  # head 继承 tombstone=True
    node.swa_tombstone = False
    self.swa_lru_list.insert_mru(node)
    self.swa_evictable_size_ += len(node.value)  # 已同质，len == mapped_tail
```

> **关于"读 live mapping 不安全"**：那特指 inc/dec lock 的**配对计数**——加和减之间 mapping 会变。而在 **insert/revive 时刻读一次决定怎么切**是安全的：切完所有节点同质，后续全用稳定的 `len(value)`。

---

### Bug #2：`cache_unfinished_req` 漏传 `swa_evicted_seqlen`

**现象**：断言在 `cache_unfinished_req → inc_lock_ref` 触发，节点 768 长仅 256 映射。

**根因**：chunked prefill 中 `_evict_swa`（`schedule_batch.py`）随 decode 推进，会对 `req_to_token[old_prefix_len:E]` 调 `free_swa` **抽走这段 SWA 映射**。下一个 chunk 的 `cache_unfinished_req` 重新 insert 整段，但**没传 `swa_evicted_seqlen`**（默认 0），`_insert_helper` 把整段当 non-tombstone 插入 → 非同质节点。

**铁证**：`cache_finished_req` 传了 `req.swa_evicted_seqlen`，`cache_unfinished_req` 没传——对称性缺失。

**修复**：

```python
result = self.insert(
    InsertParams(
        key=radix_key,
        value=values,
        prev_prefix_len=old_prefix_len,
        swa_evicted_seqlen=req.swa_evicted_seqlen,  # 补上，与 cache_finished_req 对称
    )
)
```

---

### Bug #3：FlexKV 加载的 SWA 边界没持久化到 `req`

**现象**：补了 Bug #2 后**断言依然触发，数字一字不变**（768/256, node.id=17）。

**这是决定性反证**：若 insert 端切分有效，补传边界就该消除它。没消除，说明**补传的值本身是 0**。

**根因**：`init_load_back` 算出的 `swa_evicted_seqlen`（=512）是**局部变量**，只传给那一次 insert 就丢弃，**从不写回 `req.swa_evicted_seqlen`**。而该字段只有 `_evict_swa` 会写，刚加载完的 req 仍是 0。于是后续 `cache_unfinished_req` 读到 0（Bug #2 的修复因数据源为空而失效）。

> FlexKV 的 `alloc_extend_swa_tail` 让"前 512 slot 永久无 SWA"成为这条 prefix 的**固有物理属性**，但这个边界只在加载那一刻用了一次。

**修复**：把边界持久化（见 §4 最终代码，与 Bug #4 合并实现，已转绝对坐标）。

---

### Bug #4：`init_load_back` 用相对切片 key 插入 → tree-internal 重复节点

**现象**：断言不再触发（同质性已修），但出现**新的、有界**泄漏：full 多 768、swa 多 256（= 一个完整 FlexKV 节点）。

**audit 定位过程**：

```
counter == walked            → 不是计数漂移
no double-owned (freelist)   → 不是 full double-free
TREE-INTERNAL duplicate full slots: 768 e.g. [(5888,15,17),...]  ← 命中！
```

tree dump 揭示结构：

```
root(0)
├── node 15  key[0:512]   slots 5888..6399  tomb=True   ← 副本A（错误，挂 root）
│   └── node 16 key[512:768] slots 6400..6655
└── node 6   key[0:256]   slots 256..511  (设备前缀, gpu_cached_len=256)
    └── node 17 key[256:768] slots 5888..6399  tomb=True ← 副本B（正确位置）
        └── node 18 key[768:1024] slots 6400..6655
```

**node 15 和 node 17 引用完全相同的物理 slot 5888..6655**。算术闭合：`7168 − 768 = 6400` 真实 slot `+ 59136` free `= 65536` ✓。

**根因**：`init_load_back` 用**从 `gpu_cached_len` 开始的相对切片 key** 调 insert：

```python
key = RadixKey(token_ids=req.fill_ids[gpu_cached_len : gpu_cached_len + host_hit_length], ...)
```

但 `insert` 永远从 **root** 导航。当 `gpu_cached_len > 0` 时，这段相对 key 被当成"从位置 0 开始"插到 root → 错位的 node 15。后续 `cache_*_req` 用**完整 key** 把同一批 slot 插到正确位置（node 6 下）→ node 17。两次落点不同，radix 无法去重 → 重复。

**铁证**：同一函数里，insert 用相对 key，紧接着的 re-match（line 401）却用完整 key——不对称。

**修复**：insert 改用**完整 key + 完整 value**，与 re-match 一致。

---

## 4. 最终修复代码

### 4.1 `extended_radix_cache.py` — `init_load_back`

```python
# 修复前
key = RadixKey(
    token_ids=req.fill_ids[gpu_cached_len : gpu_cached_len + host_hit_length],
    extra_key=req.extra_key,
)
...
swa_evicted_seqlen = 0
if has_swa_tail and window_size > 0 and not is_eagle:
    swa_evicted_seqlen = max(0, effective_len - _swa_tail_len())
self._inner_radixtree.insert(InsertParams(
    key=key, value=device_indices,
    prev_prefix_len=gpu_cached_len, swa_evicted_seqlen=swa_evicted_seqlen,
))
```

```python
# 修复后
# 完整 key：insert 从 root 导航，相对切片会建出平行节点 → tree-internal 重复 → leak
key = RadixKey(
    token_ids=req.fill_ids[: gpu_cached_len + host_hit_length],
    extra_key=req.extra_key,
)
# 完整 value = 设备前缀 slots ++ 新加载的 tail slots，让 insert 能对前缀去重
full_value = torch.cat([req.prefix_indices.to(device_indices), device_indices])
...
# swa_evicted_seqlen 现在是序列起点的绝对位置（key 从 0 起）
swa_evicted_seqlen = 0
if has_swa_tail and window_size > 0 and not is_eagle:
    swa_evicted_seqlen = gpu_cached_len + max(0, effective_len - _swa_tail_len())
self._inner_radixtree.insert(InsertParams(
    key=key, value=full_value,
    prev_prefix_len=gpu_cached_len,   # 让 _insert_helper 跳过(不 free)前缀段
    swa_evicted_seqlen=swa_evicted_seqlen,
))

# 持久化 SWA 边界到 req（Bug #3 修复）：边界是 prefix 的固有属性，需跨调用存活
if swa_evicted_seqlen > 0:
    req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, swa_evicted_seqlen)
```

### 4.2 `swa_radix_cache.py` — `_swa_slots_in_value`

```python
def _swa_slots_in_value(self, value: torch.Tensor) -> int:
    # 每个 non-tombstone 节点都是同质 SWA-mapped 的，故 SWA slot 数 == len(value)。
    # 在此读 live mapping 不安全：dec_swa_lock_only 会在 lock 期间清零它，
    # 使 inc_lock_ref/dec_lock_ref 加减不配对。
    return len(value)
```

### 4.3 `swa_radix_cache.py` — revive helpers

见 §3 Bug #1 的 `_mapped_swa_tail_len` 与 `_revive_tombstone_tail`。`_insert_helper` 的 Branch 1/2 改为调用 `_revive_tombstone_tail`：

```python
if swa_evicted_seqlen <= total_prefix_length:
    # Branch 1: 按真实映射尾段 revive，而非按 len
    self.token_to_kv_pool_allocator.free(node.value[:prefix_len])
    self._revive_tombstone_tail(node, value[:prefix_len])
elif swa_evicted_seqlen < total_prefix_length + prefix_len:
    # Branch 2: caller 边界处先 split，再对尾段按真实映射二次 split
    start_update_idx = swa_evicted_seqlen - total_prefix_length
    self.token_to_kv_pool_allocator.free(node.value[start_update_idx:prefix_len])
    self._split_node(node.key, node, start_update_idx)
    self.token_to_kv_pool_allocator.free(value[:start_update_idx])
    self._revive_tombstone_tail(node, value[start_update_idx:prefix_len])
else:
    # Branch 3 不变
    self.token_to_kv_pool_allocator.free(value[:prefix_len])
```

### 4.4 `swa_radix_cache.py` — `cache_unfinished_req`

见 §3 Bug #2：insert 补传 `swa_evicted_seqlen=req.swa_evicted_seqlen`。

---

## 5. 修复链总览

| # | Bug | 修复 | 解决的现象 |
|---|---|---|---|
| 1 | revive 造非同质节点 | `_revive_tombstone_tail` 只激活真实映射尾段 | swa_evictable 无界涨到 8192 |
| 2 | `cache_unfinished_req` 漏传边界 | 补传 `swa_evicted_seqlen`，与 finished 对称 | 断言 768/256 |
| 3 | FlexKV 边界没持久化 | `init_load_back` 写回 `req.swa_evicted_seqlen` | 补 #2 后断言仍触发 |
| 4 | insert 用相对 key → 重复节点 | 改用完整 key + 完整 value | full +768 / swa +256 有界泄漏 |

`#1–#3` 解决 SWA 同质性，`#4` 解决 tree-internal 重复。`#2` 与 `#3` 严格配对：`#3` 负责"把正确边界存下来"，`#2` 负责"用起来"，缺一不可。

---

## 6. 关键经验

1. **"补了修复仍现象不变、数字不变"是最强的反证**——它把矛头从"逻辑错"转向"数据源/生命周期错"（Bug #3 即由此定位）。
2. **诊断要能区分互斥成因**：evictable 超额有"计数漂移 / tree-vs-freelist double-free / tree-internal 重复 / freelist double-free"四种，修法相反，必须先用审计区分再动手。
3. **insert 时刻读 mapping 安全，lock 配对期间读不安全**——区别在于"是否要求将来某刻读到同一个值"。
4. **对称性缺失是高价值线索**：finished 传、unfinished 不传（#2）；insert 用相对 key、re-match 用完整 key（#4）。

---

## 7. 待清理的调试脚手架

跑稳后应移除或收进 env flag（`count_mapped_swa_slots` 在 inc/dec lock 热路径上 per-node 查 mapping，长期开着有开销）：

- `_assert_swa_homogeneous` 及其三处调用
- `evict` 内部节点分支的 `assert swa_freed == len(x.value)`
- `audit_evictable` 及 `scheduler.on_idle` 里的 audit 调用
- `_mapped_swa_tail_len` 里的 `assert tail == total`（可保留为廉价不变量检查）
- `count_mapped_swa_slots`（若确认无其他用途）

建议包进 `SGLANG_SWA_HOMOGENEITY_CHECK`（默认关）。
