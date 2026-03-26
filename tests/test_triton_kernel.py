import importlib.util
import unittest

import torch

from turboquant import TurboQuantProd


class TestTritonQuantizedAttentionKernel(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA (install triton-windows on Windows; official triton on Linux)",
    )
    def test_attention_scores_match_dequantize(self):
        device = "cuda"
        quantizer = TurboQuantProd(bits=3, head_dim=32, device=device, dtype=torch.float32)

        # Small shapes for fast correctness check.
        B, H, M, N, D = 1, 1, 2, 4, quantizer.head_dim
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        # Kernel scores
        scores_triton = quantizer.quantized_attention_scores_triton(q, kv)

        # Reference: decompress K fully and compute attention scores
        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        scores_ref = torch.matmul(q, k_recon.transpose(-2, -1)) / (D**0.5)

        self.assertEqual(scores_triton.shape, scores_ref.shape)

        # Floating point tolerance: kernel accumulates in fp32.
        max_abs_err = torch.max(torch.abs(scores_triton - scores_ref)).item()
        self.assertLess(max_abs_err, 1e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_attention_scores_causal(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=0)
        B, H, S = 1, 2, 11
        q = torch.randn(B, H, S, D, device=device)
        k = torch.randn(B, H, S, D, device=device)
        kv = quantizer.quantize_kv(k, k, return_compressed=True)
        scores_t = quantizer.quantized_attention_scores_triton(q, kv, causal=True)
        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        scores_r = torch.matmul(q, k_recon.transpose(-2, -1)) / (D**0.5)
        mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        scores_r = scores_r.masked_fill(mask, float("-inf"))
        tril_m = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool))
        tril_m = tril_m.view(1, 1, S, S)
        max_err = torch.max(torch.abs(scores_t - scores_r).masked_select(tril_m)).item()
        self.assertLess(max_err, 1e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_attention_scores_mqa(self):
        """MQA: one KV head, ``num_kv_heads=1`` (Falcon / older GPT-J style)."""
        device = "cuda"
        D = 32
        H_q, H_kv = 4, 1
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=1)
        B, M, N = 1, 3, 9
        q = torch.randn(B, H_q, M, D, device=device)
        k_kv = torch.randn(B, H_kv, N, D, device=device)
        kv = quantizer.quantize_kv(k_kv, k_kv, return_compressed=True)
        scores_t = quantizer.quantized_attention_scores_triton(q, kv, num_kv_heads=H_kv)
        k_rep = k_kv.repeat_interleave(H_q // H_kv, dim=1)
        kv2 = quantizer.quantize_kv(k_rep, k_rep, return_compressed=True)
        scores_r = quantizer.quantized_attention_scores_triton(q, kv2)
        max_err = torch.max(torch.abs(scores_t - scores_r)).item()
        self.assertLess(max_err, 1e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_attention_scores_mqa_falcon_like_16_query_1_kv(self):
        """High fan-out MQA (many Q heads, one K/V head) — scores path."""
        device = "cuda"
        D = 32
        H_q, H_kv = 16, 1
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=6)
        B, M, N = 1, 4, 11
        torch.manual_seed(7)
        q = torch.randn(B, H_q, M, D, device=device)
        k_kv = torch.randn(B, H_kv, N, D, device=device)
        kv = quantizer.quantize_kv(k_kv, k_kv, return_compressed=True)
        scores_t = quantizer.quantized_attention_scores_triton(q, kv, num_kv_heads=H_kv)
        k_rep = k_kv.repeat_interleave(H_q // H_kv, dim=1)
        kv2 = quantizer.quantize_kv(k_rep, k_rep, return_compressed=True)
        scores_r = quantizer.quantized_attention_scores_triton(q, kv2)
        max_err = torch.max(torch.abs(scores_t - scores_r)).item()
        self.assertLess(max_err, 1e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_attention_scores_gqa_llama_style_8_query_2_kv(self):
        """Scores kernel: same GQA head grouping as fused attention (Llama-3 style ratios)."""
        device = "cuda"
        D = 32
        H_q, H_kv = 8, 2
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=2)
        B, M, N = 1, 5, 12
        torch.manual_seed(3)
        q = torch.randn(B, H_q, M, D, device=device)
        k_kv = torch.randn(B, H_kv, N, D, device=device)
        kv = quantizer.quantize_kv(k_kv, k_kv, return_compressed=True)
        scores_t = quantizer.quantized_attention_scores_triton(q, kv, num_kv_heads=H_kv)
        k_rep = k_kv.repeat_interleave(H_q // H_kv, dim=1)
        kv2 = quantizer.quantize_kv(k_rep, k_rep, return_compressed=True)
        scores_r = quantizer.quantized_attention_scores_triton(q, kv2)
        max_err = torch.max(torch.abs(scores_t - scores_r)).item()
        self.assertLess(max_err, 1e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_attention_scores_custom_mask(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=4)
        B, H, M, N = 1, 2, 7, 15
        torch.manual_seed(5)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, k, return_compressed=True)

        mask_2d = torch.rand(M, N, device=device) > 0.55
        scores_t = quantizer.quantized_attention_scores_triton(q, kv, attention_mask=mask_2d)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        scores_ref = torch.matmul(q, k_recon.transpose(-2, -1)) / (D**0.5)
        add = torch.where(
            mask_2d,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32),
        ).view(1, 1, M, N)
        scores_ref = scores_ref + add

        fin = torch.isfinite(scores_ref)
        max_err = torch.max(torch.abs(scores_t[fin] - scores_ref[fin])).item()
        self.assertLess(max_err, 1e-3)
        self.assertTrue(torch.all(torch.isinf(scores_t[~fin])))


if __name__ == "__main__":
    unittest.main(verbosity=2)

