"""
Binary sidecar and layout notes for **llama.cpp** (C++/CUDA) integration.

llama.cpp does not ship TurboQuant; this module defines a **portable sidecar**
(``*.tqmeta``) with ``Π``, ``S``, centroids, and ``qjl_factor`` so native code can
decode the same packed KV pages as :mod:`turboquant.vllm_pack`.

**Paged KV bytes** per physical block are identical to vLLM:
:class:`~turboquant.vllm_pack.TurboQuantPageLayout`, :func:`~turboquant.vllm_pack.scatter_one_token`,
:func:`~turboquant.vllm_pack.uint8_pages_to_paged_dict`.

See ``integrations/llama_cpp/README.md`` for upstream hook points and a reference C++ decoder.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import BinaryIO, Union

import torch

from .core import TurboQuantProd, _centroid_levels_for

# File format: portable binary sidecar header + payload.
#
# v1 (legacy): integer `bits` in the header, and `codebook` inferred from `(bits, k)` in Python.
_MAGIC = b"TURBOQT1"
_VERSION_1 = 1
_HEADER_STRUCT_V1 = struct.Struct("<8sIIIId")  # magic, ver, bits(u32), head_dim(u32), k(u32), qjl(d)

# v2: store `bits` as float64 and include codebook kind explicitly.
# Needed because fractional bits can collide with ternary in centroid count `k`.
_VERSION_2 = 2
_HEADER_STRUCT_V2 = struct.Struct("<8sIdIIId")  # magic, ver, bits(f64), head_dim(u32), k(u32), codebook(u32), qjl(d)


def _infer_codebook_from_header_v1(bits: int, k: int) -> str:
    ek = _centroid_levels_for(int(bits), "paper")
    if int(k) == ek:
        return "paper"
    if int(k) == 3:
        return "ternary"
    raise ValueError(
        f"turboquant metadata (v1): bits={bits} and k={k} do not match paper (expect k={ek}) or ternary (k=3)"
    )


def _codebook_to_int(codebook: str) -> int:
    if codebook == "paper":
        return 0
    if codebook == "ternary":
        return 1
    raise ValueError(f"unsupported codebook kind: {codebook!r}")


def _codebook_from_int(v: int) -> str:
    if v == 0:
        return "paper"
    if v == 1:
        return "ternary"
    raise ValueError(f"unsupported codebook id: {v}")


def serialize_quantizer_metadata(quantizer: TurboQuantProd) -> bytes:
    """
    Pack ``TurboQuantProd`` state into bytes (CPU float32 Π/S/centroids + double qjl_factor).

    Use with the same ``bits``/``head_dim`` when allocating KV pages; load in C++ or
    :func:`deserialize_quantizer_metadata`.
    """
    d = int(quantizer.head_dim)
    bits = float(quantizer.bits)
    k = int(quantizer._centroids.numel())
    pi_f = quantizer.Pi.detach().float().cpu().contiguous().numpy().tobytes()
    s_f = quantizer.S.detach().float().cpu().contiguous().numpy().tobytes()
    c_f = quantizer._centroids.detach().float().cpu().contiguous().numpy().tobytes()
    if len(pi_f) != d * d * 4 or len(s_f) != d * d * 4:
        raise RuntimeError("internal shape error for Pi/S")
    if len(c_f) != k * 4:
        raise RuntimeError("internal shape error for centroids")
    codebook_id = _codebook_to_int(quantizer.codebook)
    header = _HEADER_STRUCT_V2.pack(
        _MAGIC,
        _VERSION_2,
        float(bits),
        d,
        k,
        int(codebook_id),
        float(quantizer._qjl_factor),
    )
    return header + c_f + pi_f + s_f


def deserialize_quantizer_metadata(
    data: bytes,
    *,
    device: Union[str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> TurboQuantProd:
    """Rebuild :class:`~turboquant.TurboQuantProd` from :func:`serialize_quantizer_metadata` output."""
    # Need at least the legacy header prefix to read version.
    if len(data) < _HEADER_STRUCT_V1.size:
        raise ValueError("truncated turboquant metadata")

    magic = data[0:8]
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}")

    ver = struct.unpack_from("<I", data, 8)[0]
    if ver == _VERSION_1:
        hlen = _HEADER_STRUCT_V1.size
        _, ver1, bits_i, head_dim_i, k_i, qjl = _HEADER_STRUCT_V1.unpack_from(data, 0)
        assert ver1 == _VERSION_1
        codebook = _infer_codebook_from_header_v1(int(bits_i), int(k_i))
        bits = float(int(bits_i))
        head_dim = int(head_dim_i)
        k = int(k_i)
    elif ver == _VERSION_2:
        hlen = _HEADER_STRUCT_V2.size
        _, ver2, bits_f, head_dim_i, k_i, codebook_id, qjl = _HEADER_STRUCT_V2.unpack_from(data, 0)
        assert ver2 == _VERSION_2
        codebook = _codebook_from_int(int(codebook_id))
        bits = float(bits_f)
        head_dim = int(head_dim_i)
        k = int(k_i)
    else:
        raise ValueError(f"unsupported version {ver}")

    need = hlen + k * 4 + head_dim * head_dim * 8
    if len(data) < need:
        raise ValueError(f"truncated payload: need {need} bytes, got {len(data)}")

    off = hlen
    centroids = torch.frombuffer(bytearray(data[off : off + k * 4]), dtype=torch.float32).clone()
    off += k * 4
    d2 = head_dim * head_dim * 4
    pi = torch.frombuffer(bytearray(data[off : off + d2]), dtype=torch.float32).reshape(head_dim, head_dim).clone()
    off += d2
    s = torch.frombuffer(bytearray(data[off : off + d2]), dtype=torch.float32).reshape(head_dim, head_dim).clone()

    expected_qjl = math.sqrt(math.pi / 2.0) / float(head_dim)
    if not math.isclose(qjl, expected_qjl, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"corrupt qjl_factor in header: {qjl} vs expected {expected_qjl}")

    dev = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    return TurboQuantProd(
        bits=float(bits),
        head_dim=head_dim,
        device=dev,
        dtype=dtype,
        codebook=codebook,  # type: ignore[arg-type]
        Pi=pi,
        S=s,
        centroids=centroids,
    )


def write_quantizer_metadata(path: Union[str, Path], quantizer: TurboQuantProd) -> None:
    Path(path).write_bytes(serialize_quantizer_metadata(quantizer))


def read_quantizer_metadata(
    path: Union[str, Path],
    *,
    device: Union[str, None] = None,
    dtype: torch.dtype = torch.float32,
) -> TurboQuantProd:
    return deserialize_quantizer_metadata(Path(path).read_bytes(), device=device, dtype=dtype)


def append_metadata_to_file(fp: BinaryIO, quantizer: TurboQuantProd) -> None:
    """Write one metadata blob (for multi-quantizer archives; caller defines outer framing)."""
    fp.write(serialize_quantizer_metadata(quantizer))
