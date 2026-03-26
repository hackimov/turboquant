"""Smoke: Mistral / Qwen2 / Qwen3 + TurboQuant fused attention (same contract as Llama-style)."""

from __future__ import annotations

import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


def _have_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _triton_cuda() -> bool:
    return importlib.util.find_spec("triton") is not None and torch.cuda.is_available()


def _tiny_mistral(device: str):
    from transformers import MistralConfig, MistralForCausalLM

    cfg = MistralConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        sliding_window=None,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    m = MistralForCausalLM(cfg)
    m.to(device=device, dtype=torch.float32)
    m.eval()
    return m


def _tiny_granite(device: str):
    from transformers import GraniteConfig, GraniteForCausalLM

    cfg = GraniteConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    m = GraniteForCausalLM(cfg)
    m.to(device=device, dtype=torch.float32)
    m.eval()
    return m


def _tiny_gemma2(device: str):
    from transformers import Gemma2Config, Gemma2ForCausalLM

    cfg = Gemma2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        attn_logit_softcapping=50.0,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    m = Gemma2ForCausalLM(cfg)
    m.to(device=device, dtype=torch.float32)
    m.eval()
    return m


def _tiny_qwen2(device: str):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_sliding_window=False,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    m = Qwen2ForCausalLM(cfg)
    m.to(device=device, dtype=torch.float32)
    m.eval()
    return m


def _tiny_qwen3(device: str):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        use_sliding_window=False,
    )
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "eager"
    m = Qwen3ForCausalLM(cfg)
    m.to(device=device, dtype=torch.float32)
    m.eval()
    return m


@unittest.skipUnless(_have_transformers(), "requires transformers")
class TestHFDecoderFusedSmoke(unittest.TestCase):
    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_mistral_fused_prefill_cuda(self):
        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_fused_attention import (
            TurboQuantMistralAttention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        device = "cuda"
        model = _tiny_mistral(device)
        hd = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=hd, device=device, dtype=torch.float32, seed=11)
        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_fused_attention(model, q, architecture="mistral")
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 5), device=device)
            attn = torch.ones(1, 5, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantMistralAttention)
        finally:
            uninstall_turboquant_fused_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_qwen2_fused_prefill_cuda(self):
        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_fused_attention import (
            TurboQuantQwen2Attention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        device = "cuda"
        model = _tiny_qwen2(device)
        hd = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=hd, device=device, dtype=torch.float32, seed=12)
        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_fused_attention(model, q, architecture="qwen2")
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
            attn = torch.ones(1, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantQwen2Attention)
        finally:
            uninstall_turboquant_fused_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_qwen3_fused_prefill_cuda(self):
        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_fused_attention import (
            TurboQuantQwen3Attention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        device = "cuda"
        model = _tiny_qwen3(device)
        hd = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        q = TurboQuantProd(bits=3, head_dim=hd, device=device, dtype=torch.float32, seed=120)
        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_fused_attention(model, q, architecture="qwen3")
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
            attn = torch.ones(1, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantQwen3Attention)
        finally:
            uninstall_turboquant_fused_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_granite_fused_prefill_cuda(self):
        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_fused_attention import (
            TurboQuantGraniteAttention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        device = "cuda"
        model = _tiny_granite(device)
        hd = model.config.hidden_size // model.config.num_attention_heads
        q = TurboQuantProd(bits=3, head_dim=hd, device=device, dtype=torch.float32, seed=13)
        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_fused_attention(model, q, architecture="granite")
        try:
            inp = torch.randint(0, model.config.vocab_size, (1, 5), device=device)
            attn = torch.ones(1, 5, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantGraniteAttention)
        finally:
            uninstall_turboquant_fused_attention(model)

    @unittest.skipUnless(_triton_cuda(), "requires CUDA and Triton")
    def test_gemma2_softcap_uses_stock_attention_path_cuda(self):
        """Gemma2 softcap / query scaling → stock forward; wrapper installed but no Triton logits path."""
        from turboquant.hf_cache import turboquant_dynamic_cache
        from turboquant.hf_fused_attention import (
            TurboQuantGemma2Attention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        device = "cuda"
        model = _tiny_gemma2(device)
        hd = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
        q = TurboQuantProd(bits=3, head_dim=hd, device=device, dtype=torch.float32, seed=14)
        cache = turboquant_dynamic_cache(model.config, q, triton_fused_layers=True, strict_reencode=False)
        install_turboquant_fused_attention(model, q, architecture="gemma2")
        try:
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantGemma2Attention)
            inp = torch.randint(0, model.config.vocab_size, (1, 4), device=device)
            attn = torch.ones(1, 4, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(input_ids=inp, attention_mask=attn, past_key_values=cache, use_cache=True)
            self.assertTrue(torch.isfinite(out.logits).all())
        finally:
            uninstall_turboquant_fused_attention(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
