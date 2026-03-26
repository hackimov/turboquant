# TurboQuant → vLLM (upstream) integration

End-to-end wiring for **vLLM v1** with packed TurboQuant KV pages and Triton fused decode (`turboquant` + `turboquant.vllm_pack`). After applying the patch below you get:

- `--kv-cache-dtype turboquant` and `--turboquant-bits {1.5,2,2.5,3,4}` (default 3)
- CUDA backend selection → `TURBOQUANT_ATTN` when the KV dtype is `turboquant`
- `TurboQuantAttentionSpec` for correct `real_page_size_bytes` in the KV allocator
- Backend module `vllm/v1/attention/backends/turboquant_attn.py` (same content as `overlay/...` in this repo)

## Requirements

- GPU with Triton support (same class of devices as the rest of TurboQuant CUDA paths)
- `pip install turboquant-kv[triton]` in the **same** environment as vLLM
- Models with **decoder** attention, **no sliding window**, **no sinks**, **no KV sharing**, **no MLA**, `head_size` in `{16,32,64,128,256}`, and `head_size_v == head_size`

## Install (recommended: git patch)

1. Clone [vLLM](https://github.com/vllm-project/vllm) and check out a revision close to patch base commit **`e38817f`** (vLLM `main` when [`patches/vllm_turboquant_e38817f.patch`](patches/vllm_turboquant_e38817f.patch) was generated). From the vLLM repo root you can `git apply` that file; the script below is recommended.
2. From the **turboquant-kv** repo:

   ```bash
   python integrations/vllm_upstream/apply_to_vllm.py /path/to/vllm
   ```

   Dry-run: add `--check`. If hunks fail on a newer main, try `--3way`.

3. `pip install turboquant-kv[triton]` then install vLLM in editable mode, e.g. `pip install -e ./vllm`.
4. Smoke: see [Smoke checklist (TURBOQUANT_ATTN)](#smoke-checklist-turboquant_attn) below.

## Smoke checklist (TURBOQUANT_ATTN)

Use this when you need to confirm that **TurboQuant’s Triton path** is active for attention, not FlashAttention-2/3 paged KV.

### 1. CLI (minimum)

```bash
vllm serve <model> \
  --kv-cache-dtype turboquant \
  --max-model-len 4096
# optional: --turboquant-bits 4   # default 3; allowed 1.5–4 (incl. 1.5/2.5 outlier allocation)
```

First bring-up (CUDA graph issues): add `--enforce-eager` (see [Limitations](#limitations)).

### 2. Log line added by the patch (reliable)

When `cache_dtype` is validated, **`vllm/config/cache.py`** logs once (INFO):

> `Using TurboQuant packed KV cache; install the optional dependency: pip install 'turboquant-kv[triton]'.`

Capture and grep:

```bash
vllm serve <model> --kv-cache-dtype turboquant --max-model-len 4096 2>&1 | tee /tmp/vllm-tq.log
grep -F "Using TurboQuant packed KV cache" /tmp/vllm-tq.log
```

That confirms the engine accepted **`turboquant`** KV dtype, which is the gate for **`TURBOQUANT_ATTN`** on CUDA (`vllm/platforms/cuda.py` returns only `TURBOQUANT_ATTN` when `kv_cache_dtype == "turboquant"` and not MLA).

### 3. Log: selected attention backend (vLLM v1 CUDA)

After workers initialize, the CUDA platform typically logs which backend was chosen (wording may vary slightly by revision), including the string **`TURBOQUANT_ATTN`**:

```bash
grep -E "TURBOQUANT_ATTN|attention backend" /tmp/vllm-tq.log
```

If you see a line like `Using TURBOQUANT_ATTN attention backend out of potential backends: ...`, that is direct confirmation—not FlashAttention-2/3—for decoder attention using this integration.

### 4. Optional: DEBUG / module path

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ... --kv-cache-dtype turboquant ...
```

Then search the log for **`turboquant_attn`** (backend module path). Exact DEBUG lines vary by vLLM revision; the messages in §2–3 are the practical signals.

### 5. Python checks (no weights / no server)

Backend name:

```bash
python -c "from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend as T; print(T.get_name())"
# expect: TURBOQUANT_ATTN
```

Optional: assert CUDA priority list via **`vllm.platforms.cuda._get_backend_priorities`** (private API; **signature differs between vLLM revisions**—inspect before calling):

```python
import inspect
from vllm.platforms import cuda
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.registry import AttentionBackendEnum

print(inspect.signature(cuda._get_backend_priorities))
# Example (add/remove kwargs to match your installed vLLM):
out = cuda._get_backend_priorities(
    use_mla=False,
    device_capability=DeviceCapability(9, 0),
    kv_cache_dtype="turboquant",
)
assert out == [AttentionBackendEnum.TURBOQUANT_ATTN]
```

### 6. Contrast with FA2 / FA3 default KV

For the **same model**, a baseline without TurboQuant KV, e.g. `--kv-cache-dtype auto` (or FP16/BF16 KV), may select **FlashAttention** on supported GPUs. That path is **orthogonal** to `--kv-cache-dtype turboquant` and does not read TurboQuant’s `uint8` pages. See [Flash Attention 3, Hopper (H100), and FP8 KV paths](#flash-attention-3-hopper-h100-and-fp8-kv-paths).

## Install (manual / porting)

If the patch does not apply, use [`UPSTREAM_EDITS.md`](UPSTREAM_EDITS.md) as a checklist (same edits as the patch, with file anchors). Copy `overlay/vllm/v1/attention/backends/turboquant_attn.py` if that file is missing.

## Layout and tests (this repository)

- Page packing and scatter helpers: `turboquant.vllm_pack` — tests in `tests/test_vllm_pack.py` and CUDA paged-attention checks in `tests/test_triton_fused_attention.py`.
- vLLM stores one `uint8` row per paged block. For Triton decode, use **`paged_kv_views_from_allocator_buffer(kv_cache, layout)`** (recommended in the overlay) or **`uint8_pages_to_paged_dict`**: both build zero-copy `*_phys` views `[P, block_size, H_kv, D]` for `turboquant_fused_attention_paged`.
- The paged fused kernel tiles query rows (`BLOCK_M = 16`), masks invalid physical page ids (`pb < 0` or `pb >= P`) and out-of-range logical block columns, and avoids reading `block_tables` past the tensor width—so allocator padding (`-1` slots, wide `block_table`) matches real vLLM usage.

## Flash Attention 3, Hopper (H100), and FP8 KV paths

**TL;DR:** TurboQuant **paged KV is not** FlashAttention-2/3 paged KV. On Hopper, **FP8 attention / FA3** uses vendor layouts and kernels that assume **native FP8 (or FP16/BF16) K/V** in the format those kernels expect. TurboQuant stores **PolarQuant + QJL** state in **`uint8` pages** (`turboquant.vllm_pack.TurboQuantPageLayout`) and decodes with **Triton** (`turboquant_fused_attention_paged`). There is **no** drop-in mapping between FA3’s Hopper FP8 KV layout and TurboQuant’s page bytes—treat them as **separate backends**.

### What vLLM does when `kv_cache_dtype=turboquant`

- Platform code (see `UPSTREAM_EDITS.md`, `vllm/platforms/cuda.py`) returns **`TURBOQUANT_ATTN`** when `kv_cache_dtype == "turboquant"` (and not MLA).
- The overlay backend (`turboquant_attn.py`) uses **`TurboQuantAttentionImpl`**: KV **write** = Python scatter into the `uint8` buffer; **read** = Triton fused attention on `*_phys` views. **FlashAttention 3 is not used for those attention steps.**

### Hopper GPU (sm_90)

- The TurboQuant backend declares support from **compute capability ≥ 7.5** (see `TurboQuantAttentionBackend.supports_compute_capability`), so **H100 is in range**.
- **Verification:** run with `--kv-cache-dtype turboquant` and confirm in logs / config that the active attention backend is **`TURBOQUANT_ATTN`**, not a FlashAttention variant. If a newer vLLM revision prefers **FA3** for default FP16/BF16 KV, that path applies only when **not** using TurboQuant KV.

### FP8 (two different meanings)

1. **Weights / activations FP8** (e.g. W8A8, FP8 quantization of linear layers): orthogonal to KV layout; may still use `--kv-cache-dtype turboquant` for **KV** as long as the TurboQuant backend is selected.
2. **KV cache stored as FP8** (framework “FP8 KV” / FA-style): **not** the same as `turboquant`. You typically choose **either** FP8 KV **or** TurboQuant `uint8` KV, not both for the same buffer.

### Paged KV compatibility checklist (Hopper / FA3 context)

| Question | TurboQuant expectation |
|----------|-------------------------|
| Physical page format | One row per block: `torch.uint8` of length `TurboQuantPageLayout(...).page_bytes` |
| Logical tensors for Triton | `paged_kv_views_from_allocator_buffer` / `uint8_pages_to_paged_dict` → `k_idx_phys`, …, `v_gamma_phys` |
| Same as FA3 FP8 paged layout? | **No** — different semantics and packing |
| Action if integrating FA3 elsewhere | Keep TurboQuant on **`TURBOQUANT_ATTN`**; do not point FA3 kernels at TurboQuant buffers |

For layout invariants and tests, see **`turboquant.vllm_pack`**, `tests/test_vllm_pack.py`, and paged fused tests in `tests/test_triton_fused_attention.py`.

## Limitations

- **ROCm / CPU / XPU**: not wired; CUDA `_get_backend_priorities` only selects TurboQuant when `kv_cache_dtype == "turboquant"`.
- **Prefix caching / CUDA graphs**: not fully validated; prefer `--enforce-eager` for first bring-up if you see capture errors.
- **Performance**: per-token cache updates run in Python; a future step is a fused CUDA/Triton cache writer.

## Upstream PR checklist

1. Apache-2.0 headers on new files (already in `turboquant_attn.py`).
2. Optional dependency: document `turboquant-kv[triton]` in vLLM docs or extras.
3. CI: smoke test gated on CUDA + optional import (pattern used for other Triton backends).
4. Keep `TurboQuantAttentionSpec` and worker `ZeroBlockIds` / `init_meta` handling in sync (`type(spec) is TurboQuantAttentionSpec` branch in `vllm/v1/worker/utils.py`).
