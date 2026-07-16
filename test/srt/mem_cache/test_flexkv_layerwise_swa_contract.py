import unittest
from pathlib import Path


class TestFlexKVLayerwiseSWAContract(unittest.TestCase):
    def test_connector_does_not_prefire_dense_layer_eventfds(self):
        repo_root = Path(__file__).resolve().parents[3]
        connector_source = (
            repo_root / "python/sglang/srt/mem_cache/storage/flexkv/flexkv_connector.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("_signal_dense_layers_ready", connector_source)


if __name__ == "__main__":
    unittest.main()
