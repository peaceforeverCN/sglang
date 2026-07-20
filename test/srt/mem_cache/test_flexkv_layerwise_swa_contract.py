import unittest
from pathlib import Path


class TestFlexKVLayerwiseSWAContract(unittest.TestCase):
    @staticmethod
    def _connector_source():
        repo_root = Path(__file__).resolve().parents[3]
        return (
            repo_root / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py"
        ).read_text(encoding="utf-8")

    def test_connector_does_not_prefire_dense_layer_eventfds(self):
        connector_source = self._connector_source()

        self.assertNotIn("_signal_dense_layers_ready", connector_source)

    def test_connector_does_not_depend_on_flexkv_debug_logging(self):
        connector_source = self._connector_source()

        self.assertNotIn("flexkv.common.debug", connector_source)
        self.assertNotIn("SEGV-DEBUG", connector_source)
        self.assertNotIn("FLEXKV-DEBUG", connector_source)
        self.assertNotIn("print(", connector_source)

    def test_compress_state_sidecars_default_on_and_allow_swa_only(self):
        connector_source = self._connector_source()

        self.assertIn(
            'self.flexkv_config.user_config, "swa_multi_group", None',
            connector_source,
        )
        self.assertIn("if swa_multi_group is not False:", connector_source)
        self.assertIn("using SWA-only I/O", connector_source)


if __name__ == "__main__":
    unittest.main()
