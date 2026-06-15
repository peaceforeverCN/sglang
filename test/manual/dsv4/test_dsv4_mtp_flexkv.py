"""DSV4 + EAGLE MTP + FlexKV connector smoke tests (L1).

This is a level-1 (smoke) test that verifies the *integration boundary*
between three independently-developed pieces:

    DSV4 model  ──────┐
                      │
    EAGLE MTP   ──────┼──> SGLang scheduler ──> FlexKV connector ──> CPU/SSD cache
                      │
    FlexKV      ──────┘

What this test verifies:
  L1.1 server starts: DSV4 + MTP + FlexKV connector all init together.
  L1.2 first request completes; cached_tokens == 0 (cold cache).
  L1.3 same prefix in second request hits FlexKV cache (cached_tokens > 0).
  L1.4 MTP still produces meaningful accept_length under FlexKV (>1.3).

What this test does NOT verify (covered by L2 / L3):
  - logprob equivalence between cold & warm runs (L3 — KL divergence).
  - performance numbers (covered by test_dsv4_flash_mtp_tp8.py without FlexKV).
  - correctness of multi-branch prefix sharing (L2 — UnifiedRadixTreeTestMixin).

Reference tests this combines:
  - test_dsv4_flash_mtp_tp8.py     : DSV4 Flash + MTP server config
  - test_dsv4_hicache_swa_translation_cache.py : HiCache+SWA+DSV4 (the
    HiCache analog of what we're testing with FlexKV).

Manual test (8× H200, 285B FP8 weights). Not registered in CI.

Required environment:
  - FlexKV installed and importable as
    ``sglang.srt.mem_cache.storage.flexkv.flexkv_connector.FlexKVConnector``.
  - ``FLEXKV_CONFIG_PATH`` pointing to a yaml config with at least
    ``cpu_cache_gb`` set (a few GB is enough for this smoke test).
  - ``FLEXKV_SERVER_RECV_PORT`` set to an unused IPC socket path.

Tunables (env vars):
  DSV4_MTP_FLEXKV_MODEL              (default sgl-project/DeepSeek-V4-Flash-FP8)
  DSV4_MTP_FLEXKV_TP_SIZE            (default 8)
  DSV4_MTP_FLEXKV_LAUNCH_TIMEOUT     (default 3600s)
  DSV4_MTP_FLEXKV_ACCEPT_LENGTH_MIN  (default 1.3 — see L1.4 docstring)
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DSV4_MODEL = os.environ.get(
    "DSV4_MTP_FLEXKV_MODEL", "sgl-project/DeepSeek-V4-Flash-FP8"
)
TP_SIZE = int(os.environ.get("DSV4_MTP_FLEXKV_TP_SIZE", "8"))
LAUNCH_TIMEOUT = int(os.environ.get("DSV4_MTP_FLEXKV_LAUNCH_TIMEOUT", "3600"))

# Minimum accept_length we expect MTP to retain under FlexKV. EAGLE on DSV4
# Flash typically achieves ~2.5-3.0 in steady state. We set 1.3 as a low bar:
# anything below that means MTP is broken (every other draft rejected).
ACCEPT_LENGTH_MIN = float(os.environ.get("DSV4_MTP_FLEXKV_ACCEPT_LENGTH_MIN", "1.3"))

FLEXKV_CONNECTOR_CLS = (
    "sglang.srt.mem_cache.storage.flexkv.flexkv_connector.FlexKVConnector"
)


# ---- Long prompt ----------------------------------------------------------
# FlexKV stores at PAGE granularity (page_size=256 for DSv4). A prompt
# shorter than 1 page is dropped at store time:
#
#     aligned_len = (len(token_ids) // page_size) * page_size
#     if aligned_len == 0:
#         self._completed_stores.append(task_id)   # silently skip
#         return
#
# DSv4's tokenizer is dense for Chinese (~1 token per char), so an
# 80-char prompt yields ~30 tokens (well under page_size=256). Empirical:
# our previous ~600-char prompt only produced 173 tokens — still less
# than one page, so FlexKV silently dropped the store.
#
# To safely cross *2 pages* (≥ 512 tokens), this prompt is ~2200 Chinese
# chars. The content is intentionally repetitive/expanded so the request
# stays cheap at decode time but exercises FlexKV across multiple pages,
# which is necessary to test load_back behavior at page boundaries.
LONG_PROMPT = (
    "你是一位资深的人工智能研究员，请用中文非常详细地全面介绍一下机器学习"
    "这个领域。首先讲一下机器学习的定义和它与传统编程的根本区别，传统编程"
    "是程序员把规则手工编码出来，机器学习则是从数据中自动归纳出规律。"
    "然后请系统讲一下机器学习的主要分类，包括监督学习、无监督学习、半监督"
    "学习和强化学习。监督学习里详细讲一下分类和回归两大问题，分类的代表"
    "算法有逻辑回归、决策树、支持向量机、随机森林、梯度提升树等，回归的"
    "代表算法有线性回归、岭回归、Lasso回归等，每种算法的核心思想、训练"
    "流程、优缺点和典型适用场景都要展开讨论。无监督学习里详细讲一下聚类"
    "和降维两大问题，聚类的代表算法有K均值、层次聚类、DBSCAN、谱聚类等，"
    "降维的代表算法有主成分分析、t-SNE、UMAP、自编码器等，每种算法的核心"
    "思想和典型应用场景都要展开。强化学习里讲一下基于价值的方法（Q学习、"
    "深度Q网络）、基于策略的方法（策略梯度、近端策略优化）、Actor-Critic"
    "方法等，并讨论它们在游戏、机器人控制、推荐系统等领域的应用。"
    "接下来请深入讲一下深度学习作为机器学习的一个重要分支，它的核心思想"
    "是什么，为什么需要使用神经网络，浅层网络和深层网络的本质区别在哪里，"
    "梯度消失和梯度爆炸问题如何解决（残差连接、批归一化、层归一化等技术），"
    "常见的网络结构包括全连接网络、卷积神经网络、循环神经网络、长短期记忆"
    "网络、门控循环单元、Transformer等各自的设计动机、关键算子、典型适用"
    "任务是什么。卷积神经网络重点讲一下卷积、池化、感受野等概念，以及"
    "ResNet、VGG、Inception、EfficientNet等代表架构的演进。Transformer"
    "重点讲一下自注意力机制、多头注意力、位置编码、编码器解码器结构等"
    "核心概念，以及它如何彻底改变自然语言处理乃至计算机视觉领域。"
    "请同时讨论一下大规模预训练模型的范式革命，包括BERT、GPT、LLaMA等"
    "代表性模型的训练方法、参数规模、能力涌现现象，以及它们对下游任务"
    "微调的影响。"
    "最后请详细讲一下机器学习当前面临的主要技术挑战，包括但不限于：模型"
    "可解释性问题（黑盒模型在医疗、司法等高风险领域的应用障碍）、数据"
    "隐私和安全问题（联邦学习、差分隐私、同态加密等技术路线）、模型偏见"
    "和公平性问题（数据偏差、表示偏差、评估偏差及其缓解方法）、计算"
    "效率和能耗问题（模型压缩、知识蒸馏、量化、剪枝等技术）、泛化能力"
    "和分布外鲁棒性问题、对抗样本和模型鲁棒性问题、长尾数据和小样本"
    "学习问题、持续学习和灾难性遗忘问题等等。请并结合你对这个领域未来"
    "五到十年发展方向的看法，谈一谈你认为哪些研究方向最有前景，哪些可能"
    "遇到瓶颈。请条理清晰、层次分明，每一部分都展开讲解，不要只是简单"
    "罗列，要给出充分的论据和例子，最好引用一些经典论文或者代表性工作"
    "作为支撑。请确保答案总长度足够长，不少于两千字。"
)


# Server CLI args. Mirrors test_dsv4_flash_mtp_tp8.py but adds FlexKV connector.
DSV4_MTP_FLEXKV_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp",
    str(TP_SIZE),
    # MTP / EAGLE configuration (matches DSV4 Flash defaults).
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--max-running-requests",
    "8",
    # FlexKV connector (the key flag we're testing).
    "--kv-connector-cls",
    FLEXKV_CONNECTOR_CLS,
]


DSV4_MTP_FLEXKV_BASE_ENV = {
    # Same env as test_dsv4_flash_mtp_tp8.py.
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    # FlexKV needs to be enabled. Server-side defaults read from these.
    # FLEXKV_CONFIG_PATH and FLEXKV_SERVER_RECV_PORT are expected to be set
    # in the *outer* environment (or via a test runner wrapper) so that the
    # caller controls cache placement and socket isolation.
    "ENABLE_FLEXKV": "1",
    # Use Copy Engine (independent hardware) instead of SM-based memcpy for
    # H2D/D2H KV transfers. This avoids SM contention with the MTP draft
    # model, which is especially important under multi-pool (DSv4) workloads
    # where 4 sub-pools (c4 / c128 / c4_indexer / SWA) move data in parallel.
    "FLEXKV_USE_CE_TRANSFER_H2D": "1",
    "FLEXKV_USE_CE_TRANSFER_D2H": "1",
}


def _launch_server(extra_env=None):
    """Launch DSV4 + MTP + FlexKV server.

    ``FLEXKV_CONFIG_PATH`` and ``FLEXKV_SERVER_RECV_PORT`` must already be
    set in the caller's environment; we propagate them via popen_launch_server.
    """
    # Sanity check: cache config must be set or FlexKV will fail to start.
    for required in ("FLEXKV_CONFIG_PATH", "FLEXKV_SERVER_RECV_PORT"):
        if required not in os.environ:
            raise unittest.SkipTest(
                f"{required} not set; required for FlexKV connector. "
                f"Example: export FLEXKV_CONFIG_PATH=/path/to/flexkv_config.yml"
            )

    env = dict(DSV4_MTP_FLEXKV_BASE_ENV)
    if extra_env:
        env.update(extra_env)

    return popen_launch_server(
        DSV4_MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=DSV4_MTP_FLEXKV_SERVER_ARGS,
        env=env,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDSV4MTPFlexKVSmoke(CustomTestCase):
    """L1 smoke: DSV4 + EAGLE MTP + FlexKV end-to-end.

    Boots the server once for the whole class and runs four ordered
    sub-tests against the same instance. The order matters: test_03
    relies on test_02 having populated the cache.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = _launch_server()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ---- helpers ----------------------------------------------------------

    def _generate(self, prompt, max_new_tokens=64):
        """POST /generate and return the parsed response.

        Uses temperature=0 for determinism so that a re-issued prompt produces
        the same token stream — important for the cache-hit test.
        """
        r = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    def _flush(self):
        """Flush GPU device cache so the next request must reload from FlexKV.

        WARNING: SGLang's /flush_cache calls ExtendedRadixCache.reset() which
        ALSO calls FlexKVConnector.reset() — wiping the FlexKV CPU pool.
        After /flush_cache, FlexKV cannot serve any cached prefix because its
        CPU pool is empty. So /flush_cache is unsuitable for testing FlexKV
        load_back; use _evict_gpu_only() instead.
        """
        requests.get(self.base_url + "/flush_cache", timeout=30)

    def _evict_gpu_only(self, num_filler_requests=4):
        """Evict GPU radix tree without touching FlexKV's CPU pool.

        Strategy: send several long requests with *unrelated* prompts to
        push the original prompt's KV out of the GPU radix tree (LRU evict).
        FlexKV's CPU pool is unaffected by GPU eviction, so on the next
        same-prefix request, cache must come from FlexKV via load_back.

        ``num_filler_requests`` controls how aggressive the eviction is.
        With ``--max-running-requests 8`` and 768-token filler prompts,
        4 fillers usually suffice to evict any single previous request.
        """
        import random
        for i in range(num_filler_requests):
            # Generate an unrelated long prompt (random content guarantees
            # it doesn't share a prefix with the original test prompt).
            seed = random.randint(0, 999999)
            filler = (
                f"请详细介绍一下随机话题{seed}的相关知识。"
                + "请尽可能详细、长篇大论、面面俱到地展开论述，"
                * 80  # ~ 1000+ chars / ~600+ tokens, enough to fill GPU pool
            )
            try:
                requests.post(
                    self.base_url + "/generate",
                    json={
                        "text": filler,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 8,
                        },
                    },
                    timeout=300,
                )
            except Exception:
                pass  # filler request failures don't fail the test

    @staticmethod
    def _accept_length(meta):
        """accept_length = completion_tokens / spec_verify_ct.

        SGLang reports both fields in meta_info when speculative decoding is
        enabled. Returns None if either field is absent (server may not emit
        spec stats in some build configurations).
        """
        verify_ct = meta.get("spec_verify_ct")
        completion = meta.get("completion_tokens")
        if not verify_ct or not completion:
            return None
        return completion / verify_ct

    # ---- L1.1 -------------------------------------------------------------

    def test_01_server_alive(self):
        """L1.1: server reports healthy and exposes server_info."""
        r = requests.get(self.base_url + "/get_server_info", timeout=30)
        r.raise_for_status()
        info = r.json()
        # The exact schema isn't important here — we just want a non-empty
        # JSON response, which proves the HTTP server, scheduler and
        # connector all came up without crashing.
        self.assertTrue(info, "server_info returned empty payload")

    # ---- L1.2 -------------------------------------------------------------

    def test_02_first_request_cold_cache(self):
        """L1.2: first request completes; cached_tokens == 0 (no prior data)."""
        # Make sure we start cold — flush in case a previous test left cache.
        self._flush()
        result = self._generate(LONG_PROMPT, max_new_tokens=64)
        meta = result["meta_info"]
        self.assertGreater(
            meta["completion_tokens"],
            0,
            "completion_tokens should be > 0 for a successful generation",
        )
        # Cold cache: nothing should hit. Allow a tiny tolerance because
        # EAGLE-2's BOS bookkeeping can sometimes yield cached_tokens=1.
        cached = meta.get("cached_tokens", 0)
        self.assertLessEqual(
            cached,
            1,
            f"expected cold cache (cached_tokens ≤ 1), got {cached}",
        )
        print(
            f"[L1.2 cold] cached={cached} "
            f"completion={meta['completion_tokens']} "
            f"spec_verify_ct={meta.get('spec_verify_ct', 'N/A')}"
        )

    # ---- L1.3 -------------------------------------------------------------

    def test_03_second_request_hits_flexkv(self):
        """L1.3: re-issuing the same prefix hits FlexKV after a device flush.

        Sequence:
          1. send prompt P (this populates GPU + FlexKV CPU cache via store)
          2. /flush_cache to drop the GPU-side radix tree
          3. send the same prompt P + a small suffix; FlexKV should serve the
             prefix from CPU cache → cached_tokens > 0.

        The exact value depends on FlexKV's page alignment, so we only assert
        that *some* prefix was reused. A future L2 test pins the exact count.

        Important: prompt MUST be longer than page_size (256 for DSv4) tokens,
        otherwise FlexKV's store path drops the request entirely as
        ``aligned_len == 0``. See LONG_PROMPT comment for details.
        """
        # Step 1: warm up — populate FlexKV with the long prompt.
        # Need to give async store time to complete before flushing.
        first = self._generate(LONG_PROMPT, max_new_tokens=32)
        first_meta = first["meta_info"]

        # FlexKV store is async; wait for it to finish flushing to host pool.
        import time
        time.sleep(5)

        # Step 2: drop the GPU-side cache. After this, any prefix reuse must
        # come from FlexKV CPU/SSD storage via load_back.
        # CANNOT use /flush_cache — it also wipes FlexKV's CPU pool via
        # ExtendedRadixCache.reset() -> connector.reset(). Instead push the
        # cached prefix out via LRU eviction with unrelated requests.
        self._evict_gpu_only(num_filler_requests=4)
        time.sleep(2)

        # Step 3: re-issue same long prefix with a small suffix tweak. The
        # shared prefix is what we care about; the suffix forces decode.
        second = self._generate(LONG_PROMPT + " 谢谢你的耐心讲解。", max_new_tokens=32)
        second_meta = second["meta_info"]

        cached = second_meta.get("cached_tokens", 0)
        prompt_tokens = second_meta.get("prompt_tokens", 0)

        print(
            f"[L1.3 warm] cached_tokens={cached} prompt_tokens={prompt_tokens} "
            f"(first run cached={first_meta.get('cached_tokens', 0)})"
        )
        self.assertGreater(
            cached,
            0,
            "FlexKV should have served some prefix from CPU cache after flush, "
            "but cached_tokens=0. This usually means the connector didn't "
            "store on the first request, or the load_back path is broken.",
        )

    # ---- L1.4 -------------------------------------------------------------

    def test_04_mtp_accept_length_baseline(self):
        """L1.4: EAGLE accept_length on a cold (no cache) workload.

        Establishes the *no-cache* MTP baseline. FlexKV is enabled but no
        prior request seeded the cache, so this measures pure MTP behavior
        without any KV reuse.

        Threshold rationale:
          - DSV4 Flash + EAGLE on long inputs typically reports accept ≈ 2.5-3.0.
          - We assert > 1.3 here as a *correctness* check (anything below that
            means most draft tokens are rejected, which is suspicious).
          - Performance regressions get caught by test_dsv4_flash_mtp_tp8 and
            by the L3 KL test, not here.

        This baseline pairs with test_05 (cache-hit accept_length) to verify
        the piggyback design (PR #21125 analog): draft KV must be shipped
        alongside target KV on store/load so that decode after a load_back
        retains the same draft accept rate as cold runs.
        """
        # Make sure no prior cache state pollutes the measurement.
        self._flush()
        # Use the long prompt so this baseline is comparable with test_05
        # (which needs a long prompt for FlexKV to actually store anything).
        result = self._generate(LONG_PROMPT, max_new_tokens=256)
        meta = result["meta_info"]

        accept = self._accept_length(meta)
        if accept is None:
            self.skipTest(
                "server didn't report spec_verify_ct/completion_tokens; "
                "cannot compute accept_length"
            )

        # Save the baseline so test_05 can compare against it.
        type(self)._baseline_accept_length = accept

        print(
            f"[L1.4 baseline] accept_length={accept:.2f} "
            f"completion={meta['completion_tokens']} "
            f"spec_verify_ct={meta['spec_verify_ct']} "
            f"cached={meta.get('cached_tokens', 0)}"
        )
        self.assertGreater(
            accept,
            ACCEPT_LENGTH_MIN,
            f"MTP accept_length={accept:.2f} below threshold "
            f"{ACCEPT_LENGTH_MIN}; FlexKV may be interfering with "
            f"speculative decoding.",
        )

    # ---- L1.5 -------------------------------------------------------------

    def test_05_mtp_accept_length_after_load_back(self):
        """L1.5: ★ KEY TEST ★ — Draft KV piggyback works.

        This is the core test for the HiCache PR #21125 analog: when the
        target KV cache hits FlexKV's CPU pool and is loaded back to GPU,
        the *draft* KV cache must also be loaded back, otherwise the first
        few decode steps after the hit will see stale or zero draft KV and
        accept_length will collapse.

        Test sequence:
          1. Send a long prompt P. Both target and draft KV land in GPU,
             then on request finish, FlexKV stores them to CPU pool.
          2. flush_cache: drop the GPU radix tree. Now the only place the
             KV exists is FlexKV's CPU pool.
          3. Send P again. FlexKV loads the prefix from CPU back to GPU.
          4. Measure accept_length on the decode that follows.

        Expected behavior:
          - test_04 baseline:           accept_length L4 (cold, e.g. 2.5)
          - this test (after load_back): accept_length L5

          Without piggyback (draft KV dropped on store):
            L5 << L4   (e.g. 1.0-1.5; first decode step sees empty draft KV)

          With piggyback (PR #21125-style):
            L5 ≈ L4    (within a few percent; draft KV restored intact)

          We assert L5 >= 80% of L4. The 20% slack absorbs run-to-run jitter
          but rejects an actually broken piggyback path.

        Reference numbers (from PR #21125, HiCache):
          - Without piggyback: accept_length ≈ 3.07
          - With piggyback:    accept_length ≈ 6.94 (≈2.3x improvement on
            long-context workloads where prefix reuse dominates)
        """
        if not hasattr(type(self), '_baseline_accept_length'):
            self.skipTest(
                "test_04_mtp_accept_length_baseline must run first to establish baseline"
            )

        baseline = type(self)._baseline_accept_length

        # Same long prompt as test_04 to keep the comparison apples-to-apples.
        # MUST be > page_size (256) tokens or FlexKV silently drops the store.
        # Step 1: warm up — populate FlexKV.
        first = self._generate(LONG_PROMPT, max_new_tokens=64)

        # Give async store some time to complete (target + draft KV).
        import time
        time.sleep(5)

        # Step 2: drop the GPU-side cache. After this, prefix reuse must
        # come from FlexKV via load_back.
        # CANNOT use /flush_cache — it wipes FlexKV's CPU pool too.
        self._evict_gpu_only(num_filler_requests=4)
        time.sleep(2)

        # Step 3: re-issue the same prompt. FlexKV should hit and load_back
        # both target AND draft KV (if piggyback is wired up correctly).
        result = self._generate(LONG_PROMPT, max_new_tokens=256)
        meta = result["meta_info"]

        cached = meta.get("cached_tokens", 0)
        accept_after_load = self._accept_length(meta)
        if accept_after_load is None:
            self.skipTest(
                "server didn't report spec_verify_ct/completion_tokens"
            )

        # First sanity: the load_back must have happened. If cached_tokens=0,
        # the test isn't even measuring what we think it is.
        self.assertGreater(
            cached,
            0,
            f"Expected FlexKV to hit on second request (cached_tokens > 0), "
            f"got {cached}. test_03 should have already verified this; if "
            f"test_03 passed and this didn't, it's a flaky cache hit.",
        )

        ratio = accept_after_load / baseline
        print(
            f"[L1.5 piggyback] baseline={baseline:.2f} "
            f"after_load_back={accept_after_load:.2f} ratio={ratio:.2%} "
            f"cached={cached} prompt_tokens={meta.get('prompt_tokens')}"
        )

        # Core assertion: accept_length after load_back should match baseline
        # within 20%. Anything significantly lower indicates draft KV was lost.
        self.assertGreaterEqual(
            ratio,
            0.80,
            f"accept_length collapsed from {baseline:.2f} (cold) to "
            f"{accept_after_load:.2f} after FlexKV load_back "
            f"(ratio={ratio:.2%}). This strongly suggests draft KV is NOT "
            f"being piggybacked through FlexKV — see HiCache PR #21125 "
            f"for the analog implementation (set_draft_kv_pool + "
            f"backup_from_device_all_layer / load_to_device_per_layer "
            f"on draft pool).",
        )


if __name__ == "__main__":
    unittest.main()
