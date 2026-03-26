import tempfile
import unittest
from pathlib import Path

import torch

from turboquant import TurboQuantProd
from turboquant.llama_cpp_pack import (
    deserialize_quantizer_metadata,
    read_quantizer_metadata,
    serialize_quantizer_metadata,
    write_quantizer_metadata,
)
from turboquant.vllm_pack import TurboQuantPageLayout, scatter_one_token, uint8_pages_to_paged_dict


class TestLlamaCppPack(unittest.TestCase):
    def test_metadata_roundtrip_tensors(self):
        d, bits = 64, 3
        q0 = TurboQuantProd(bits=bits, head_dim=d, device="cpu", seed=42)
        blob = serialize_quantizer_metadata(q0)
        q1 = deserialize_quantizer_metadata(blob, device="cpu")
        self.assertTrue(torch.allclose(q0.Pi, q1.Pi))
        self.assertTrue(torch.allclose(q0.S, q1.S))
        self.assertTrue(torch.allclose(q0._centroids, q1._centroids))
        self.assertEqual(q0.bits, q1.bits)
        self.assertEqual(q0.head_dim, q1.head_dim)

    def test_metadata_roundtrip_ternary_codebook(self):
        d, bits = 32, 3
        q0 = TurboQuantProd(bits=bits, head_dim=d, device="cpu", seed=5, codebook="ternary")
        blob = serialize_quantizer_metadata(q0)
        q1 = deserialize_quantizer_metadata(blob, device="cpu")
        self.assertEqual(q1.codebook, "ternary")
        self.assertEqual(q1._centroids.numel(), 3)
        self.assertTrue(torch.allclose(q0._centroids, q1._centroids))

    def test_metadata_roundtrip_fractional_bits(self):
        d, bits = 64, 1.5
        q0 = TurboQuantProd(bits=bits, head_dim=d, device="cpu", seed=42, codebook="paper")
        blob = serialize_quantizer_metadata(q0)
        q1 = deserialize_quantizer_metadata(blob, device="cpu")
        self.assertEqual(q1.codebook, "paper")
        self.assertEqual(q0.bits, q1.bits)
        self.assertEqual(q0._centroids.numel(), q1._centroids.numel())
        self.assertTrue(torch.allclose(q0._centroids, q1._centroids))

        k = torch.randn(1, 2, 5, d)
        v = torch.randn(1, 2, 5, d)
        c0 = q0.quantize_kv(k, v, return_compressed=True)
        c1 = q1.quantize_kv(k, v, return_compressed=True)
        for name in c0:
            a, b = c0[name], c1[name]
            if a.dtype in (torch.int32, torch.int64):
                self.assertTrue(torch.equal(a, b), msg=name)
            else:
                self.assertTrue(torch.allclose(a, b, rtol=1e-5, atol=1e-5), msg=name)

    def test_metadata_file_roundtrip_quantize_match(self):
        d, bits = 32, 2
        q0 = TurboQuantProd(bits=bits, head_dim=d, device="cpu", seed=7)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "m.tqmeta"
            write_quantizer_metadata(p, q0)
            q1 = read_quantizer_metadata(p, device="cpu")
        k = torch.randn(1, 2, 5, d)
        v = torch.randn(1, 2, 5, d)
        c0 = q0.quantize_kv(k, v, return_compressed=True)
        c1 = q1.quantize_kv(k, v, return_compressed=True)
        for name in c0:
            a, b = c0[name], c1[name]
            if a.dtype in (torch.int32, torch.int64):
                self.assertTrue(torch.equal(a, b), msg=name)
            else:
                self.assertTrue(
                    torch.allclose(a, b, rtol=1e-5, atol=1e-5),
                    msg=f"{name} max_abs={(a - b).abs().max().item()}",
                )

    def test_fixed_pi_s_constructor(self):
        d = 16
        q0 = TurboQuantProd(bits=3, head_dim=d, device="cpu", seed=0)
        q1 = TurboQuantProd(
            bits=3,
            head_dim=d,
            device="cpu",
            Pi=q0.Pi.clone(),
            S=q0.S.clone(),
            centroids=q0._centroids.clone(),
        )
        x = torch.randn(1, 1, 4, d)
        c0 = q0.quantize_kv(x, x, return_compressed=True)
        c1 = q1.quantize_kv(x, x, return_compressed=True)
        self.assertTrue(torch.equal(c0["k_idx"], c1["k_idx"]))

    def test_paged_layout_unchanged_vs_vllm(self):
        block_size, h, d = 4, 2, 64
        aux = torch.float16
        layout = TurboQuantPageLayout.build(block_size, h, d, aux)
        pages = torch.zeros(2, layout.page_bytes, dtype=torch.uint8)
        q = TurboQuantProd(bits=3, head_dim=d, device="cpu", seed=123)
        key = torch.randn(h, d)
        val = torch.randn(h, d)
        scatter_one_token(pages, layout, 0, 1, q, key, val)
        paged = uint8_pages_to_paged_dict(pages, layout)
        ck = q.quantize_kv(
            key.unsqueeze(0).unsqueeze(2),
            val.unsqueeze(0).unsqueeze(2),
            return_compressed=True,
        )
        bi, pos = 0, 1
        self.assertTrue(torch.equal(paged["k_idx_phys"][bi, pos], ck["k_idx"][0, :, 0, :]))
        # scatter stores aux fields in layout.aux_dtype (fp16); ck tensors stay quantizer dtype (fp32).
        self.assertTrue(
            torch.allclose(
                paged["k_norm_phys"][bi, pos].float(),
                ck["k_norm"][0, :, 0, :].float(),
                rtol=1e-3,
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
