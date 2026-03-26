"""
Binary layout for TurboQuant-compressed paged KV (one uint8 row per physical page).

This is **not** FlashAttention / FlashAttention-3 / Hopper FP8 KV storage layout: those stacks
use their own packed K/V formats for vendor kernels. TurboQuant pages are only consumed by
``turboquant.kernels.fused_attention.turboquant_fused_attention_paged`` (Triton) via ``*_phys`` views.

Used by the vLLM upstream overlay in ``integrations/vllm_upstream/``: one logical
``[num_blocks, page_bytes]`` tensor matches the fused Triton paged tensors
``[P, block_size, H, D]`` / ``[P, block_size, H, 1]`` laid out back-to-back.

``bits`` does not change tensor sizes (indices always int64 in storage); the
quantizer codebook depends on ``bits`` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from .core import TurboQuantProd


def _esize(dt: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dt).element_size())


def align_up(n: int, a: int) -> int:
    return (int(n) + int(a) - 1) // int(a) * int(a)


def _int64_scalar_as_u8(src: torch.Tensor) -> torch.Tensor:
    """0-dim int64 (and 1-element tensors) → length-8 uint8 for packed page rows."""
    return src.to(torch.int64).reshape(1).view(torch.uint8)


def _scalar_as_u8(src: torch.Tensor) -> torch.Tensor:
    """0-dim float/half/bfloat → raw element bytes (length = element size)."""
    return src.contiguous().reshape(1).view(torch.uint8)


@dataclass(frozen=True)
class TurboQuantPageLayout:
    """Byte layout of one paged block (``block_size`` tokens)."""

    block_size: int
    num_kv_heads: int
    head_dim: int
    aux_dtype: torch.dtype

    k_idx_offset: int
    k_idx_bytes: int
    k_norm_offset: int
    k_norm_bytes: int
    k_sign_offset: int
    k_sign_bytes: int
    k_gamma_offset: int
    k_gamma_bytes: int
    v_idx_offset: int
    v_idx_bytes: int
    v_norm_offset: int
    v_norm_bytes: int
    v_sign_offset: int
    v_sign_bytes: int
    v_gamma_offset: int
    v_gamma_bytes: int
    page_bytes: int

    @staticmethod
    def build(
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        aux_dtype: torch.dtype,
        *,
        align: int = 8,
    ) -> "TurboQuantPageLayout":
        b, h, d = int(block_size), int(num_kv_heads), int(head_dim)
        es = _esize(aux_dtype)
        idx_count = b * h * d
        norm_count = b * h * 1
        sign_count = b * h * d
        gamma_count = b * h * 1

        off = 0

        def pack_int64_elems(n: int) -> Tuple[int, int]:
            nonlocal off
            nbytes = align_up(n * 8, align)
            start = off
            off += nbytes
            return start, nbytes

        def pack_aux_elems(n: int) -> Tuple[int, int]:
            nonlocal off
            nbytes = align_up(n * es, align)
            start = off
            off += nbytes
            return start, nbytes

        k_idx_offset, k_idx_bytes = pack_int64_elems(idx_count)
        k_norm_offset, k_norm_bytes = pack_aux_elems(norm_count)
        k_sign_offset, k_sign_bytes = pack_aux_elems(sign_count)
        k_gamma_offset, k_gamma_bytes = pack_aux_elems(gamma_count)
        v_idx_offset, v_idx_bytes = pack_int64_elems(idx_count)
        v_norm_offset, v_norm_bytes = pack_aux_elems(norm_count)
        v_sign_offset, v_sign_bytes = pack_aux_elems(sign_count)
        v_gamma_offset, v_gamma_bytes = pack_aux_elems(gamma_count)
        page_bytes = off
        return TurboQuantPageLayout(
            block_size=b,
            num_kv_heads=h,
            head_dim=d,
            aux_dtype=aux_dtype,
            k_idx_offset=k_idx_offset,
            k_idx_bytes=k_idx_bytes,
            k_norm_offset=k_norm_offset,
            k_norm_bytes=k_norm_bytes,
            k_sign_offset=k_sign_offset,
            k_sign_bytes=k_sign_bytes,
            k_gamma_offset=k_gamma_offset,
            k_gamma_bytes=k_gamma_bytes,
            v_idx_offset=v_idx_offset,
            v_idx_bytes=v_idx_bytes,
            v_norm_offset=v_norm_offset,
            v_norm_bytes=v_norm_bytes,
            v_sign_offset=v_sign_offset,
            v_sign_bytes=v_sign_bytes,
            v_gamma_offset=v_gamma_offset,
            v_gamma_bytes=v_gamma_bytes,
            page_bytes=page_bytes,
        )


def turboquant_paged_block_bytes(
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    aux_dtype: torch.dtype,
) -> int:
    """Total bytes per physical page (one KV cache block)."""
    return TurboQuantPageLayout.build(block_size, num_kv_heads, head_dim, aux_dtype).page_bytes


def uint8_pages_to_paged_dict(
    pages_u8: torch.Tensor,
    layout: TurboQuantPageLayout,
) -> Dict[str, torch.Tensor]:
    """
    View ``pages_u8`` ``[num_blocks, page_bytes]`` as the eight physical tensors
    expected by :func:`turboquant.kernels.fused_attention.turboquant_fused_attention_paged`.
    """
    if pages_u8.dtype != torch.uint8:
        raise TypeError("pages_u8 must be uint8")
    if pages_u8.dim() != 2:
        raise ValueError("pages_u8 must be [num_blocks, page_bytes]")
    nb = pages_u8.shape[0]
    b, h, d = layout.block_size, layout.num_kv_heads, layout.head_dim

    def view_region(
        offset: int,
        nbytes: int,
        dtype: torch.dtype,
        per_block_elems: int,
        tail_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
        chunk = pages_u8.narrow(1, offset, nbytes)
        es = _esize(dtype)
        cols = nbytes // es
        if cols < per_block_elems:
            raise RuntimeError("layout region too small")
        typed = chunk.view(dtype)
        typed = typed.narrow(1, 0, per_block_elems)
        return typed.view(nb, *tail_shape)

    out: Dict[str, torch.Tensor] = {}
    idx_el = b * h * d
    norm_el = b * h
    out["k_idx_phys"] = view_region(
        layout.k_idx_offset, layout.k_idx_bytes, torch.int64, idx_el, (b, h, d)
    )
    out["k_norm_phys"] = view_region(
        layout.k_norm_offset, layout.k_norm_bytes, layout.aux_dtype, norm_el, (b, h, 1)
    )
    out["k_sign_phys"] = view_region(
        layout.k_sign_offset, layout.k_sign_bytes, layout.aux_dtype, idx_el, (b, h, d)
    )
    out["k_gamma_phys"] = view_region(
        layout.k_gamma_offset, layout.k_gamma_bytes, layout.aux_dtype, norm_el, (b, h, 1)
    )
    out["v_idx_phys"] = view_region(
        layout.v_idx_offset, layout.v_idx_bytes, torch.int64, idx_el, (b, h, d)
    )
    out["v_norm_phys"] = view_region(
        layout.v_norm_offset, layout.v_norm_bytes, layout.aux_dtype, norm_el, (b, h, 1)
    )
    out["v_sign_phys"] = view_region(
        layout.v_sign_offset, layout.v_sign_bytes, layout.aux_dtype, idx_el, (b, h, d)
    )
    out["v_gamma_phys"] = view_region(
        layout.v_gamma_offset, layout.v_gamma_bytes, layout.aux_dtype, norm_el, (b, h, 1)
    )
    return out


def paged_kv_views_from_allocator_buffer(
    kv_cache: torch.Tensor,
    layout: TurboQuantPageLayout,
) -> Dict[str, torch.Tensor]:
    """
    Zero-copy mapping from a vLLM-style KV buffer ``[num_physical_blocks, page_bytes]`` (any dtype
    storage, typically ``torch.uint8`` or ``torch.int8``) to the ``*_phys`` tensors used by
    :func:`turboquant.kernels.fused_attention.turboquant_fused_attention_paged`.

    The allocator owns ``kv_cache``; this only builds typed strided views into the same storage.
    """
    if kv_cache.dim() != 2:
        raise ValueError("kv_cache must be [num_blocks, page_bytes]")
    return uint8_pages_to_paged_dict(kv_cache.view(torch.uint8), layout)


def num_physical_blocks(kv_cache: torch.Tensor) -> int:
    """Leading dimension of the paged KV buffer (number of allocatable pages)."""
    if kv_cache.dim() != 2:
        raise ValueError("kv_cache must be [num_blocks, page_bytes]")
    return int(kv_cache.shape[0])


def _linear_token_index(pos_in_block: int, head: int, dim: int, *, H: int, D: int) -> int:
    return (pos_in_block * H + head) * D + dim


def scatter_one_token(
    pages_u8: torch.Tensor,
    layout: TurboQuantPageLayout,
    physical_block: int,
    pos_in_block: int,
    quantizer: TurboQuantProd,
    key_1: torch.Tensor,
    value_1: torch.Tensor,
) -> None:
    """
    Write one token into a paged uint8 buffer.

    ``key_1`` / ``value_1``: shape ``[num_kv_heads, head_dim]``.
    """
    if key_1.shape != (layout.num_kv_heads, layout.head_dim):
        raise ValueError(f"key_1 shape {key_1.shape} expected {(layout.num_kv_heads, layout.head_dim)}")
    if value_1.shape != key_1.shape:
        raise ValueError("value_1 must match key_1")

    k_batch = key_1.unsqueeze(0).unsqueeze(2)
    v_batch = value_1.unsqueeze(0).unsqueeze(2)
    ck = quantizer.quantize_kv(k_batch, v_batch, return_compressed=True)

    row = pages_u8[physical_block]
    _, h, d = layout.block_size, layout.num_kv_heads, layout.head_dim
    es = _esize(layout.aux_dtype)

    def idx_linear(pos: int, hi: int, di: int) -> int:
        return _linear_token_index(pos, hi, di, H=h, D=d)

    def norm_linear(pos: int, hi: int) -> int:
        return pos * h + hi

    for hi in range(h):
        for di in range(d):
            li = idx_linear(pos_in_block, hi, di)
            src = ck["k_idx"][0, hi, 0, di]
            off = layout.k_idx_offset + li * 8
            row.narrow(0, off, 8).copy_(_int64_scalar_as_u8(src))
            ks = ck["k_sign"][0, hi, 0, di].to(layout.aux_dtype)
            off = layout.k_sign_offset + li * es
            row.narrow(0, off, es).copy_(_scalar_as_u8(ks))
        ln = norm_linear(pos_in_block, hi)
        kn = ck["k_norm"][0, hi, 0, 0].to(layout.aux_dtype)
        row.narrow(0, layout.k_norm_offset + ln * es, es).copy_(_scalar_as_u8(kn))
        kg = ck["k_gamma"][0, hi, 0, 0].to(layout.aux_dtype)
        row.narrow(0, layout.k_gamma_offset + ln * es, es).copy_(_scalar_as_u8(kg))

    for hi in range(h):
        for di in range(d):
            li = idx_linear(pos_in_block, hi, di)
            src = ck["v_idx"][0, hi, 0, di]
            off = layout.v_idx_offset + li * 8
            row.narrow(0, off, 8).copy_(_int64_scalar_as_u8(src))
            vs = ck["v_sign"][0, hi, 0, di].to(layout.aux_dtype)
            off = layout.v_sign_offset + li * es
            row.narrow(0, off, es).copy_(_scalar_as_u8(vs))
        ln = norm_linear(pos_in_block, hi)
        vn = ck["v_norm"][0, hi, 0, 0].to(layout.aux_dtype)
        row.narrow(0, layout.v_norm_offset + ln * es, es).copy_(_scalar_as_u8(vn))
        vg = ck["v_gamma"][0, hi, 0, 0].to(layout.aux_dtype)
        row.narrow(0, layout.v_gamma_offset + ln * es, es).copy_(_scalar_as_u8(vg))


def scatter_tokens_from_cache_update(
    pages_u8: torch.Tensor,
    layout: TurboQuantPageLayout,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    quantizer: TurboQuantProd,
    block_size: int,
) -> None:
    """
    vLLM-style cache update: ``key``/``value`` are ragged token-major
    ``[>=num_tokens, H, D]``; ``slot_mapping`` has one slot per token (same length
    as leading dimension used by the caller).
    """
    sm = slot_mapping.flatten().long()
    n = sm.numel()
    key_flat = key[:n]
    val_flat = value[:n]
    for i in range(n):
        slot = int(sm[i].item())
        pb = slot // block_size
        pos = slot % block_size
        scatter_one_token(
            pages_u8,
            layout,
            pb,
            pos,
            quantizer,
            key_flat[i],
            val_flat[i],
        )
