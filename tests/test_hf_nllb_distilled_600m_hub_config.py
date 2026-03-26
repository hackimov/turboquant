"""
Hub config smoke for ``facebook/nllb-200-distilled-600M`` (``M2M100Config``, ``model_type: m2m_100``).

Loads only ``config.json`` from the Hub (no weight download). Skips if offline / no cache.
Full model forward is not run here: allocating ~600M random parameters would be heavy for CI.
"""

import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "requires transformers")
class TestHFNllbDistilled600mHubConfig(unittest.TestCase):
    def test_hub_config_head_dim_and_cache_layers(self):
        try:
            from transformers import AutoConfig

            from turboquant.hf_cache import TurboQuantEncoderDecoderCache, turboquant_encoder_decoder_cache
        except ImportError as e:
            self.skipTest(str(e))

        try:
            cfg = AutoConfig.from_pretrained("facebook/nllb-200-distilled-600M")
        except Exception as e:
            self.skipTest(f"Hub config unavailable (offline or no cache): {e}")

        self.assertEqual(cfg.model_type, "m2m_100")
        head_dim = int(cfg.d_model) // int(cfg.decoder_attention_heads)
        self.assertEqual(head_dim, 64)
        self.assertEqual(int(cfg.decoder_layers), 12)

        quantizer = TurboQuantProd(bits=3, head_dim=head_dim, device="cpu", dtype=torch.float32, seed=0)
        past = turboquant_encoder_decoder_cache(cfg, quantizer)
        self.assertIsInstance(past, TurboQuantEncoderDecoderCache)
        self.assertEqual(len(past.cross_attention_cache.layers), int(cfg.decoder_layers))
        self.assertEqual(len(past.self_attention_cache.layers), int(cfg.decoder_layers))


if __name__ == "__main__":
    unittest.main()
