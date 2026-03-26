#!/usr/bin/env python3
"""
Export TurboQuant quantizer metadata to a llama.cpp-compatible sidecar.

llama.cpp integration expects a binary `*.tqmeta` blob next to the `*.gguf`.
This script generates that blob using the same portable format as:
`turboquant.llama_cpp_pack.write_quantizer_metadata`.

Usage example:
    python integrations/llama_cpp/export_tqmeta_sidecar.py \
      --gguf /path/to/model.gguf --bits 3 --head-dim 128 --seed 42
-> writes `/path/to/model.tqmeta`
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from turboquant import TurboQuantProd
from turboquant.llama_cpp_pack import read_quantizer_metadata, write_quantizer_metadata


def _load_tensor_auto(path: Path, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    # Keep this minimal: accept torch tensors saved with torch.save, or NumPy .npy.
    suffix = path.suffix.lower()
    if suffix in (".pt", ".pth"):
        t = torch.load(str(path), map_location="cpu")
        if isinstance(t, dict):
            # Common convention: {"tensor": ...}
            if "tensor" in t:
                t = t["tensor"]
            else:
                raise ValueError(f"Unsupported dict payload in {path}")
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor in {path}, got: {type(t)}")
        return t.to(dtype=dtype).contiguous()
    if suffix == ".npy":
        import numpy as np

        arr = np.load(str(path))
        return torch.from_numpy(arr).to(dtype=dtype).contiguous()
    raise ValueError(f"Unsupported tensor format for {path} (suffix={suffix!r})")


def build_quantizer_from_args(args: argparse.Namespace) -> TurboQuantProd:
    device = "cpu"
    dtype = torch.float32
    codebook = args.codebook

    if args.seed is not None:
        return TurboQuantProd(
            bits=float(args.bits),
            head_dim=int(args.head_dim),
            device=device,
            dtype=dtype,
            seed=int(args.seed),
            codebook=codebook,
        )

    if args.pi_path is None or args.s_path is None:
        raise SystemExit(
            "Error: provide either --seed or both --pi-path and --s-path (centroids optional)."
        )

    pi = _load_tensor_auto(Path(args.pi_path), dtype=dtype)
    s = _load_tensor_auto(Path(args.s_path), dtype=dtype)
    centroids = None
    if args.centroids_path is not None:
        centroids = _load_tensor_auto(Path(args.centroids_path), dtype=dtype).view(-1)

    return TurboQuantProd(
        bits=float(args.bits),
        head_dim=int(args.head_dim),
        device=device,
        dtype=dtype,
        codebook=codebook,
        Pi=pi,
        S=s,
        centroids=centroids,
    )


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gguf",
        type=str,
        default=None,
        help="Optional: GGUF path. If provided and --tqmeta-out is omitted, writes <gguf>.tqmeta.",
    )
    p.add_argument(
        "--tqmeta-out",
        type=str,
        default=None,
        help="Output path for the sidecar (defaults to derived from --gguf).",
    )

    p.add_argument(
        "--bits",
        type=float,
        required=True,
        choices=(1.5, 2.0, 2.5, 3.0, 4.0),
        help="TurboQuant bits (supports 1.5/2.5 outlier channel allocation).",
    )
    p.add_argument("--head-dim", type=int, required=True, help="KV head dim (per attention head).")
    p.add_argument(
        "--codebook",
        type=str,
        default="paper",
        choices=("paper", "ternary"),
        help="TurboQuant codebook kind.",
    )

    seed_group = p.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seed", type=int, default=None, help="Seed used to generate Pi/S (must match training).")

    p.add_argument(
        "--pi-path",
        type=str,
        default=None,
        help="Optional: path to saved Pi tensor (.pt/.pth or .npy). Used if --seed is not provided.",
    )
    p.add_argument(
        "--s-path",
        type=str,
        default=None,
        help="Optional: path to saved S tensor (.pt/.pth or .npy). Used if --seed is not provided.",
    )
    p.add_argument(
        "--centroids-path",
        type=str,
        default=None,
        help="Optional: path to saved centroids tensor (.pt/.pth or .npy).",
    )

    p.add_argument(
        "--verify",
        action="store_true",
        help="Read back the generated file and print bits/head_dim check.",
    )

    args = p.parse_args(argv)

    if args.tqmeta_out is None:
        if args.gguf is None:
            raise SystemExit("Error: provide --tqmeta-out or provide --gguf (then derived output is used).")
        args.tqmeta_out = str(Path(args.gguf).with_suffix(".tqmeta"))

    out_path = Path(args.tqmeta_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q = build_quantizer_from_args(args)
    write_quantizer_metadata(out_path, q)

    if args.verify:
        q2 = read_quantizer_metadata(out_path, device="cpu")
        print(f"[tqmeta] wrote: {out_path}")
        print(f"[tqmeta] verify: bits={q2.bits} head_dim={q2.head_dim} qjl_factor={float(q2._qjl_factor):.8g}")
    else:
        print(f"[tqmeta] wrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

