import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "requires transformers")
class TestHFTurboQuantCache(unittest.TestCase):
    def test_turboquant_cache_updates_like_dynamic(self):
        from turboquant.hf_cache import TurboQuantDynamicCache

        class _Dec:
            num_hidden_layers = 2
            sliding_window = None
            layer_types = None

        class _Cfg:
            def get_text_config(self, decoder=True):
                return _Dec()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantizer = TurboQuantProd(bits=3, head_dim=16, device=device, dtype=torch.float32, seed=0)
        cache = TurboQuantDynamicCache(_Cfg(), quantizer)
        self.assertEqual(len(cache.layers), 2)

        B, H, T, D = 1, 2, 1, 16
        k = torch.randn(B, H, T, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, T, D, device=device, dtype=torch.float32)

        for layer_idx in range(2):
            k_out, v_out = cache.update(k, v, layer_idx, cache_kwargs=None)
            self.assertEqual(k_out.shape[-2], T)
            self.assertEqual(v_out.shape[-2], T)

        self.assertEqual(cache.get_seq_length(0), T)
        self.assertEqual(cache.get_seq_length(1), T)

        k2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        v2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        for layer_idx in range(2):
            k_out, v_out = cache.update(k2, v2, layer_idx, cache_kwargs=None)
        self.assertEqual(k_out.shape[-2], T + 1)
        self.assertEqual(cache.get_seq_length(0), T + 1)
        self.assertEqual(cache.get_seq_length(1), T + 1)

        layer0 = cache.layers[0]
        self.assertIsNotNone(layer0.compressed_kv)
        self.assertEqual(layer0.compressed_kv["k_idx"].shape[2], T + 1)

    def test_sliding_layer_without_window_uses_quant_cache(self):
        """layer_types sliding but sliding_window None — avoid int(None); use TurboQuant layer."""
        from turboquant.hf_cache import TurboQuantCacheLayer, TurboQuantDynamicCache

        class _Dec:
            num_hidden_layers = 1
            sliding_window = None
            layer_types = ["sliding_attention"]

        class _Cfg:
            def get_text_config(self, decoder=True):
                return _Dec()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantizer = TurboQuantProd(bits=3, head_dim=16, device=device, dtype=torch.float32, seed=0)
        cache = TurboQuantDynamicCache(_Cfg(), quantizer)
        self.assertEqual(len(cache.layers), 1)
        self.assertIsInstance(cache.layers[0], TurboQuantCacheLayer)

    def test_hybrid_float_cache_non_strict_appends_native_kv(self):
        """hybrid_float_cache + strict_reencode=False: return value is cat of incoming K/V (no full decompress)."""
        from turboquant.hf_cache import TurboQuantCacheLayer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantizer = TurboQuantProd(bits=3, head_dim=16, device=device, dtype=torch.float32, seed=1)
        layer = TurboQuantCacheLayer(quantizer, strict_reencode=False, hybrid_float_cache=True)
        B, H, D = 1, 2, 16
        k1 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        v1 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        ko1, vo1 = layer.update(k1, v1, None)
        torch.testing.assert_close(ko1, k1)
        torch.testing.assert_close(vo1, v1)
        k2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        v2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        ko2, vo2 = layer.update(k2, v2, None)
        torch.testing.assert_close(ko2, torch.cat([k1, k2], dim=-2))
        torch.testing.assert_close(vo2, torch.cat([v1, v2], dim=-2))
        self.assertEqual(layer.get_seq_length(), 2)
        torch.testing.assert_close(layer.keys, ko2)
        torch.testing.assert_close(layer.values, vo2)

    def test_hybrid_float_cache_strict_runs(self):
        from turboquant.hf_cache import TurboQuantCacheLayer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantizer = TurboQuantProd(bits=3, head_dim=16, device=device, dtype=torch.float32, seed=2)
        layer = TurboQuantCacheLayer(quantizer, strict_reencode=True, hybrid_float_cache=True)
        B, H, D = 1, 2, 16
        k = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        ko, vo = layer.update(k, v, None)
        self.assertEqual(ko.shape[-2], 1)
        k2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        v2 = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
        ko2, vo2 = layer.update(k2, v2, None)
        self.assertEqual(ko2.shape[-2], 2)
        self.assertEqual(layer.get_seq_length(), 2)

    def test_hybrid_float_cache_crop_syncs_float_buffer(self):
        from turboquant.hf_cache import TurboQuantCacheLayer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        quantizer = TurboQuantProd(bits=3, head_dim=16, device=device, dtype=torch.float32, seed=3)
        layer = TurboQuantCacheLayer(quantizer, strict_reencode=False, hybrid_float_cache=True)
        B, H, D = 1, 2, 16
        for _ in range(3):
            kn = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
            vn = torch.randn(B, H, 1, D, device=device, dtype=torch.float32)
            layer.update(kn, vn, None)
        layer.crop(1)
        self.assertEqual(layer.get_seq_length(), 1)
        self.assertEqual(layer.keys.shape[-2], 1)

    def test_cross_attention_mask_broadcasts_from_b_1_m_n(self):
        """M2M-100 / NLLB-style cross padding mask often looks like [B, 1, tgt, src] (last two = score dims)."""
        from turboquant.kernels.attention_mask import broadcast_additive_attn_mask

        B, H, M, N = 2, 16, 1, 64
        m = torch.ones(B, 1, M, N, dtype=torch.bool)
        out = broadcast_additive_attn_mask(m, B, H, M, N, device=m.device)
        self.assertEqual(tuple(out.shape), (B, H, M, N))


if __name__ == "__main__":
    unittest.main(verbosity=2)
