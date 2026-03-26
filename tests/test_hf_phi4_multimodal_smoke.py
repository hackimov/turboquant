"""Smoke: Phi-4 Multimodal text attention + TurboQuant fused wrapper (CPU, no full multimodal model)."""

from __future__ import annotations

import importlib.util
import unittest

import torch
import torch.nn as nn

from turboquant import TurboQuantProd


def _have_phi4mm() -> bool:
    try:
        import transformers.models.phi4_multimodal.modeling_phi4_multimodal  # noqa: F401

        return True
    except ImportError:
        return False


def _tiny_phi4mm_attn():
    from transformers import Phi4MultimodalConfig
    from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalAttention

    cfg = Phi4MultimodalConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        sliding_window=None,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=[2],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
    )
    return Phi4MultimodalAttention(cfg, layer_idx=0)


class _DecoderOnlyShell(nn.Module):
    """Minimal ``model.layers[0].self_attn`` stack for ``install_turboquant_fused_attention``."""

    def __init__(self, self_attn: nn.Module) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].self_attn = self_attn


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None, "requires transformers")
@unittest.skipUnless(_have_phi4mm(), "requires transformers phi4_multimodal")
class TestHFPhi4MultimodalSmoke(unittest.TestCase):
    def test_install_uninstall_restores_phi4mm_attention_type(self):
        from transformers.models.phi4_multimodal.modeling_phi4_multimodal import Phi4MultimodalAttention

        from turboquant.hf_fused_attention import (
            TurboQuantPhi4MultimodalAttention,
            install_turboquant_fused_attention,
            uninstall_turboquant_fused_attention,
        )

        attn = _tiny_phi4mm_attn()
        model = _DecoderOnlyShell(attn)
        hd = int(attn.head_dim)
        q = TurboQuantProd(bits=3, head_dim=hd, device="cpu", dtype=torch.float32, seed=0)
        install_turboquant_fused_attention(model, q, architecture="phi4_multimodal")
        try:
            self.assertIsInstance(model.model.layers[0].self_attn, TurboQuantPhi4MultimodalAttention)
            uninstall_turboquant_fused_attention(model)
            self.assertIs(type(model.model.layers[0].self_attn), Phi4MultimodalAttention)
        finally:
            uninstall_turboquant_fused_attention(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
