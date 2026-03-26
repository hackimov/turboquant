# TurboQuant to llama.cpp (C++/CUDA)

This directory documents how to wire **TurboQuant-compressed KV** into [llama.cpp](https://github.com/ggerganov/llama.cpp). The Python package already defines:

- **Paged byte layout** - identical to vLLM: `turboquant.vllm_pack` (`TurboQuantPageLayout`, `scatter_one_token`, `uint8_pages_to_paged_dict`).
- **Quantizer sidecar** - `turboquant.llama_cpp_pack` writes a portable `*.tqmeta` blob (`Pi`, `S`, centroids, `qjl_factor`) so C++ does not depend on PyTorch RNG.

A **reference CPU decoder** (no ggml dependency) lives in [`reference/turboquant_sidecar_loader.cpp`](reference/turboquant_sidecar_loader.cpp). Use it to validate layouts and as a starting point for ggml kernels.

## Requirements

- Model: decoder attention with `head_size` in `{16, 32, 64, 128, 256}` (same constraint as our Triton fused path).
- **CUDA path**: either port the math from `turboquant.kernels.fused_attention` / Triton, or call into a small CUDA library; this repo does not ship a llama.cpp binary.
- **Pi / S / centroids** must match between training/serving and inference: distribute the `.tqmeta` next to the GGUF or embed the tensors as custom GGUF keys (see [`UPSTREAM_EDITS.md`](UPSTREAM_EDITS.md)).

## Python: emit metadata and packed pages

```python
from turboquant import TurboQuantProd
from turboquant.llama_cpp_pack import write_quantizer_metadata
from turboquant.vllm_pack import TurboQuantPageLayout, scatter_one_token
import torch

bits, head_dim, n_kv = 3, 128, 8
q = TurboQuantProd(bits=bits, head_dim=head_dim, device="cpu")
write_quantizer_metadata("model.tqmeta", q)

block_size, aux = 16, torch.float16
layout = TurboQuantPageLayout.build(block_size, n_kv, head_dim, aux)
pages = torch.zeros(num_blocks, layout.page_bytes, dtype=torch.uint8)
# fill with scatter_one_token(..., q, key_h_d, value_h_d) per token slot
```

## Export: tqmeta sidecar next to GGUF

If your `llama.cpp` fork expects a `*.tqmeta` sidecar next to `*.gguf`, generate it like this:

```bash
python integrations/llama_cpp/export_tqmeta_sidecar.py \
  --gguf /path/to/model.gguf \
  --bits 3 \
  --head-dim 128 \
  --seed 42
```

The script will create `model.tqmeta` next to the GGUF.

Note: `Pi` and `S` depend on `--seed` (while `centroids` depend on `bits/head_dim/codebook`).
So `--seed` must match the one used when compressing KV with TurboQuant.

## Bring-up checklist (fork / patch)

1. Add a KV cache storage mode (or reuse a raw `uint8` paged buffer) sized with `layout.page_bytes` from `TurboQuantPageLayout.build(...)`.
2. On cache write, run the same quantization as `scatter_one_token` (native C++ or bind Python for prototyping).
3. Load `*.tqmeta` at model load; pass `Pi`, `S`, centroids into attention.
4. For decode, implement **dequant K/V** (see reference `.cpp`) or a fused attention kernel that consumes packed indices/norms/signs/gamma (mirror Triton).
5. Optional: extend GGUF with tensor keys `turboquant.pi`, `turboquant.s`, `turboquant.centroids`, `turboquant.bits` instead of a sidecar.

## Limitations

- **Sliding-window / MLA / shared KV / sinks**: not covered; same caveats as the vLLM overlay.
- **Performance**: reference code is correctness-oriented; production needs SIMD/CUDA fused paths.
- **ROCm / Vulkan**: same porting effort as any new KV dtype in llama.cpp.

## Related

- vLLM overlay: [`integrations/vllm_upstream/`](../vllm_upstream/README.md)
- Paged layout implementation: `turboquant/vllm_pack.py`
- HF export to paged tensors: `turboquant.hf_cache.export_cache_to_paged_per_layer`
