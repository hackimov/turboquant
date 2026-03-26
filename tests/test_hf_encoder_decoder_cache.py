"""Encoder–decoder (cross-attention) KV cache: T5, M2M-100 / NLLB-style models."""

import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "requires transformers")
class TestHFEncoderDecoderCache(unittest.TestCase):
    def test_t5_cross_attention_compressed_kv(self):
        from transformers.models.t5.configuration_t5 import T5Config
        from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration

        from turboquant.hf_cache import TurboQuantEncoderDecoderCache, turboquant_encoder_decoder_cache

        c = T5Config(
            vocab_size=128,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
        )
        quantizer = TurboQuantProd(bits=3, head_dim=8, device="cpu", dtype=torch.float32, seed=0)
        past = turboquant_encoder_decoder_cache(c, quantizer)
        self.assertIsInstance(past, TurboQuantEncoderDecoderCache)
        model = T5ForConditionalGeneration(c)
        model.eval()
        enc = torch.randint(0, 128, (1, 4))
        dec = torch.randint(0, 128, (1, 3))
        out = model(
            input_ids=enc,
            decoder_input_ids=dec,
            past_key_values=past,
            use_cache=True,
        )
        pkv = out.past_key_values
        self.assertEqual(len(pkv.cross_attention_cache.layers), 1)
        ck = pkv.cross_attention_cache.layers[0].compressed_kv
        self.assertIsNotNone(ck)
        self.assertIn("k_idx", ck)
        self.assertEqual(int(ck["k_idx"].shape[2]), 4)

    def test_m2m100_cross_attention_compressed_kv(self):
        from transformers.models.m2m_100.configuration_m2m_100 import M2M100Config
        from transformers.models.m2m_100.modeling_m2m_100 import M2M100ForConditionalGeneration

        from turboquant.hf_cache import turboquant_encoder_decoder_cache

        c = M2M100Config(
            vocab_size=128,
            d_model=32,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
        )
        head_dim = c.d_model // c.decoder_attention_heads
        quantizer = TurboQuantProd(bits=3, head_dim=head_dim, device="cpu", dtype=torch.float32, seed=1)
        past = turboquant_encoder_decoder_cache(c, quantizer)
        model = M2M100ForConditionalGeneration(c)
        model.eval()
        enc = torch.randint(0, 128, (1, 5))
        dec = torch.randint(0, 128, (1, 2))
        enc_mask = torch.ones(1, 5, dtype=torch.long)
        dec_mask = torch.ones(1, 2, dtype=torch.long)
        out = model(
            input_ids=enc,
            attention_mask=enc_mask,
            decoder_input_ids=dec,
            decoder_attention_mask=dec_mask,
            past_key_values=past,
            use_cache=True,
        )
        ck = out.past_key_values.cross_attention_cache.layers[0].compressed_kv
        self.assertIsNotNone(ck)
        self.assertEqual(int(ck["k_idx"].shape[2]), 5)

    def test_mt5_cross_attention_compressed_kv(self):
        from transformers.models.mt5.configuration_mt5 import MT5Config
        from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

        from turboquant.hf_cache import turboquant_encoder_decoder_cache

        c = MT5Config(
            vocab_size=128,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
        )
        quantizer = TurboQuantProd(bits=3, head_dim=8, device="cpu", dtype=torch.float32, seed=3)
        past = turboquant_encoder_decoder_cache(c, quantizer)
        model = MT5ForConditionalGeneration(c)
        model.eval()
        enc = torch.randint(0, 128, (1, 4))
        dec = torch.randint(0, 128, (1, 3))
        out = model(
            input_ids=enc,
            decoder_input_ids=dec,
            past_key_values=past,
            use_cache=True,
        )
        ck = out.past_key_values.cross_attention_cache.layers[0].compressed_kv
        self.assertIsNotNone(ck)
        self.assertEqual(int(ck["k_idx"].shape[2]), 4)

    def test_nllb_moe_cross_attention_compressed_kv(self):
        from transformers.models.nllb_moe.configuration_nllb_moe import NllbMoeConfig
        from transformers.models.nllb_moe.modeling_nllb_moe import NllbMoeForConditionalGeneration

        from turboquant.hf_cache import turboquant_encoder_decoder_cache

        c = NllbMoeConfig(
            vocab_size=128,
            d_model=32,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=4,
            decoder_attention_heads=4,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            num_experts=2,
            expert_capacity=4,
            router_z_loss_coef=0.0,
            router_aux_loss_coef=0.0,
        )
        head_dim = c.d_model // c.decoder_attention_heads
        quantizer = TurboQuantProd(bits=3, head_dim=head_dim, device="cpu", dtype=torch.float32, seed=4)
        past = turboquant_encoder_decoder_cache(c, quantizer)
        model = NllbMoeForConditionalGeneration(c)
        model.eval()
        enc = torch.randint(0, 128, (1, 5))
        dec = torch.randint(0, 128, (1, 2))
        out = model(
            input_ids=enc,
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            decoder_input_ids=dec,
            decoder_attention_mask=torch.ones(1, 2, dtype=torch.long),
            past_key_values=past,
            use_cache=True,
        )
        ck = out.past_key_values.cross_attention_cache.layers[0].compressed_kv
        self.assertIsNotNone(ck)
        self.assertEqual(int(ck["k_idx"].shape[2]), 5)

    def test_self_side_stays_float_dynamic_when_quantize_self_false(self):
        from transformers.cache_utils import DynamicLayer
        from transformers.models.t5.configuration_t5 import T5Config

        from turboquant.hf_cache import turboquant_encoder_decoder_cache

        c = T5Config(
            vocab_size=128,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
        )
        quantizer = TurboQuantProd(bits=3, head_dim=8, device="cpu", dtype=torch.float32, seed=2)
        past = turboquant_encoder_decoder_cache(c, quantizer, quantize_self=False)
        self.assertIsInstance(past.self_attention_cache.layers[0], DynamicLayer)
        from turboquant.hf_cache import TurboQuantCacheLayer

        self.assertIsInstance(past.cross_attention_cache.layers[0], TurboQuantCacheLayer)


if __name__ == "__main__":
    unittest.main()
