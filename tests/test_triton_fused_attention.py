import importlib.util
import math
import unittest

import torch
import torch.nn.functional as F

from turboquant import TurboQuantProd
from turboquant.kernels.fused_attention import pack_dense_kv_to_paged


def _fused_gqa_matches_repeated_kv(
    quantizer: TurboQuantProd,
    *,
    B: int,
    H_q: int,
    H_kv: int,
    M: int,
    N: int,
    D: int,
    device: str,
    seed: int,
    **fused_kw,
) -> float:
    """
    Max abs error between shared-KV path (KV ``[B, H_kv, N, D]``) and reference
    (repeat KV heads to ``H_q`` then quantize). Covers **GQA** (``H_kv``>1) and **MQA**
    (``H_kv==1``, e.g. Falcon, some GPT-J).
    """
    assert H_q % H_kv == 0
    g = H_q // H_kv
    torch.manual_seed(seed)
    q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float32)
    k_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
    v_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
    kv = quantizer.quantize_kv(k_kv, v_kv, return_compressed=True)
    out_gqa = quantizer.quantized_attention_fused_triton(q, kv, num_kv_heads=H_kv, **fused_kw)
    k_rep = k_kv.repeat_interleave(g, dim=1)
    v_rep = v_kv.repeat_interleave(g, dim=1)
    kv_rep = quantizer.quantize_kv(k_rep, v_rep, return_compressed=True)
    out_ref = quantizer.quantized_attention_fused_triton(q, kv_rep, **fused_kw)
    return torch.max(torch.abs(out_gqa - out_ref)).item()


class TestTritonFusedAttention(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_dense_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=0)

        B, H, M, N = 1, 2, 3, 17
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_triton = quantizer.quantized_attention_fused_triton(q, kv)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        self.assertEqual(out_triton.shape, out_ref.shape)
        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_matches_dense_fused(self):
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=1)

        B, H, M, N = 1, 1, 2, 24
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_dense = quantizer.quantized_attention_fused_triton(q, kv)

        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )

        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_causal_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=2)

        B, H, S = 1, 2, 19
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        out_triton = quantizer.quantized_attention_fused_triton(q, kv, causal=True)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / (D**0.5)
        causal_mask = torch.triu(
            torch.ones(S, S, device=device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        out_ref = torch.matmul(torch.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_mqa_matches_repeated_kv(self):
        """MQA: ``num_key_value_heads==1``, multiple Q heads (``num_kv_heads=1`` in API)."""
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=3)
        err = _fused_gqa_matches_repeated_kv(
            quantizer, B=1, H_q=4, H_kv=1, M=5, N=14, D=D, device=device, seed=20
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_mqa_falcon_like_16_query_1_kv(self):
        """Falcon-style MQA: many query heads, single KV head (``KV_HEAD_GROUPS=16``)."""
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=50)
        err = _fused_gqa_matches_repeated_kv(
            quantizer, B=1, H_q=16, H_kv=1, M=7, N=20, D=D, device=device, seed=51
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_mqa_paged_matches_dense(self):
        """Paged kernel with MQA: same mapping as dense (``h_kv=0`` for all ``h_q``)."""
        device = "cuda"
        D = 32
        block_size = 8
        H_q, H_kv = 8, 1
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=52)
        B, M, N = 1, 4, 24
        torch.manual_seed(53)
        q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float32)
        k_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        v_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k_kv, v_kv, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv, num_kv_heads=H_kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
            num_kv_heads=H_kv,
        )
        self.assertLess(torch.max(torch.abs(out_dense - out_paged)).item(), 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_mqa_causal_matches_repeated_kv(self):
        device = "cuda"
        D = 32
        S = 14
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=54)
        err = _fused_gqa_matches_repeated_kv(
            quantizer,
            B=1,
            H_q=8,
            H_kv=1,
            M=S,
            N=S,
            D=D,
            device=device,
            seed=55,
            causal=True,
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_llama_style_8_query_2_kv_heads(self):
        """Llama-3 family: num_attention_heads=8, num_key_value_heads=2 → 4 groups."""
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=30)
        err = _fused_gqa_matches_repeated_kv(
            quantizer, B=1, H_q=8, H_kv=2, M=6, N=13, D=D, device=device, seed=31
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_mistral_style_4_query_2_kv_heads(self):
        """Typical small Mistral-like GQA: 4 Q heads, 2 KV heads."""
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=32)
        err = _fused_gqa_matches_repeated_kv(
            quantizer, B=1, H_q=4, H_kv=2, M=4, N=16, D=D, device=device, seed=33
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_paged_matches_dense(self):
        """Paged Triton kernel must use the same h_q → h_kv mapping as dense (KV_HEAD_GROUPS)."""
        device = "cuda"
        D = 32
        block_size = 8
        H_q, H_kv = 8, 2
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=34)
        B, M, N = 1, 3, 24
        torch.manual_seed(35)
        q = torch.randn(B, H_q, M, D, device=device, dtype=torch.float32)
        k_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        v_kv = torch.randn(B, H_kv, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k_kv, v_kv, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv, num_kv_heads=H_kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
            num_kv_heads=H_kv,
        )
        self.assertLess(torch.max(torch.abs(out_dense - out_paged)).item(), 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_causal_matches_reference(self):
        device = "cuda"
        D = 32
        H_q, H_kv = 6, 2
        S = 15
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=36)
        err = _fused_gqa_matches_repeated_kv(
            quantizer,
            B=1,
            H_q=H_q,
            H_kv=H_kv,
            M=S,
            N=S,
            D=D,
            device=device,
            seed=37,
            causal=True,
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_custom_mask_matches_reference(self):
        device = "cuda"
        D = 32
        H_q, H_kv = 8, 2
        B, M, N = 1, 5, 18
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=38)
        torch.manual_seed(39)
        mask = torch.rand(M, N, device=device) > 0.5
        err = _fused_gqa_matches_repeated_kv(
            quantizer,
            B=B,
            H_q=H_q,
            H_kv=H_kv,
            M=M,
            N=N,
            D=D,
            device=device,
            seed=40,
            attention_mask=mask,
        )
        self.assertLess(err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_gqa_wrong_kv_head_count_raises(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=41)
        q = torch.randn(1, 8, 3, D, device=device, dtype=torch.float32)
        k = torch.randn(1, 2, 10, D, device=device, dtype=torch.float32)
        v = torch.randn(1, 2, 10, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        with self.assertRaises(ValueError) as ctx:
            quantizer.quantized_attention_fused_triton(q, kv, num_kv_heads=4)
        self.assertIn("num_kv_heads", str(ctx.exception).lower())

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_custom_mask_matches_reference(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=6)
        B, H, M, N = 1, 2, 4, 21
        torch.manual_seed(7)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)

        mask_bn = torch.rand(B, M, N, device=device) > 0.45
        out_triton = quantizer.quantized_attention_fused_triton(q, kv, attention_mask=mask_bn)

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        add = torch.where(
            mask_bn.unsqueeze(1).expand(B, H, M, N),
            torch.zeros((), device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32),
        )
        scores = scores + add
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_matches_dense_with_mask(self):
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=8)
        B, H, M, N = 1, 1, 3, 24
        torch.manual_seed(9)
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        mask = torch.rand(M, N, device=device) > 0.5

        out_dense = quantizer.quantized_attention_fused_triton(q, kv, attention_mask=mask)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
            attention_mask=mask,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_causal_and_custom_mask(self):
        device = "cuda"
        D = 32
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=10)
        S = 12
        torch.manual_seed(11)
        q = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        k = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        v = torch.randn(1, 1, S, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        allow = torch.ones(S, S, dtype=torch.bool, device=device)
        allow[0, S - 1] = False
        allow[3, 2] = False

        out_triton = quantizer.quantized_attention_fused_triton(
            q, kv, causal=True, attention_mask=allow
        )

        k_recon = quantizer.dequantize(kv["k_idx"], kv["k_norm"], kv["k_sign"], kv["k_gamma"])
        v_recon = quantizer.dequantize(kv["v_idx"], kv["v_norm"], kv["v_sign"], kv["v_gamma"])
        scores = torch.matmul(q, k_recon.transpose(-2, -1)) / math.sqrt(D)
        add = torch.where(
            allow,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32),
        ).view(1, 1, S, S)
        scores = scores + add
        causal_mask = torch.triu(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        out_ref = torch.matmul(F.softmax(scores, dim=-1), v_recon)

        max_err = torch.max(torch.abs(out_triton - out_ref)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_tiled_query_len_not_multiple_of_block_m(self):
        """Paged kernel uses BLOCK_M=16; query length need not be a multiple."""
        device = "cuda"
        D = 32
        block_size = 8
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=12)
        B, H, M, N = 1, 2, 17, 20
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            block_tables,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)

    @unittest.skipUnless(
        importlib.util.find_spec("triton") is not None and torch.cuda.is_available(),
        "requires triton and CUDA",
    )
    def test_fused_paged_vllm_style_padded_block_table(self):
        """Trailing ``-1`` logical slots (allocator padding) must not break attention."""
        device = "cuda"
        D = 32
        block_size = 4
        quantizer = TurboQuantProd(bits=3, head_dim=D, device=device, dtype=torch.float32, seed=13)
        B, H, M, N = 1, 1, 3, 10
        q = torch.randn(B, H, M, D, device=device, dtype=torch.float32)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
        kv = quantizer.quantize_kv(k, v, return_compressed=True)
        out_dense = quantizer.quantized_attention_fused_triton(q, kv)
        paged, block_tables, context_lens = pack_dense_kv_to_paged(kv, block_size=block_size)
        pad_slots = 6
        bt = torch.cat(
            [
                block_tables,
                torch.full((B, pad_slots), -1, device=device, dtype=torch.int32),
            ],
            dim=1,
        )
        out_paged = quantizer.quantized_attention_fused_triton_paged(
            q,
            paged,
            bt,
            context_lens,
            block_size=block_size,
            max_seq_len=N,
        )
        max_err = torch.max(torch.abs(out_dense - out_paged)).item()
        self.assertLess(max_err, 5e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
