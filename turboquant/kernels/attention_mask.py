"""Broadcast additive attention masks to [B, H, M, N] for Triton attention."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch


def broadcast_additive_attn_mask(
    attention_mask: torch.Tensor,
    B: int,
    H: int,
    M: int,
    N: int,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Float mask added to logits (before softmax / same as stored scores).

    - ``bool``: ``True`` → 0, ``False`` → ``-inf``.
    - ``float`` / half: values are added directly (use ``0`` / ``-inf`` for keep/mask).

    Accepted shapes (then broadcast to ``(B, H, M, N)``):

    - ``[M, N]`` — shared batch and heads
    - ``[B, M, N]`` — shared heads
    - ``[B, H, M, N]`` — full

    Encoder–decoder **cross-attention** often supplies a dense padding mask with last dims ``(M, N)`` =
    ``(decoder_query_len, encoder_key_len)``, e.g. ``[batch, 1, tgt_len, src_len]`` — valid as long as the
    trailing ``(M, N)`` matches the score matrix.

    The result may use stride ``0`` on broadcast dimensions (no copy).
    """
    dev = device if device is not None else attention_mask.device
    if attention_mask.device != dev:
        attention_mask = attention_mask.to(dev)

    if attention_mask.dtype == torch.bool:
        z = torch.zeros((), device=dev, dtype=torch.float32)
        neg = torch.tensor(float("-inf"), device=dev, dtype=torch.float32)
        m = torch.where(attention_mask, z, neg)
    else:
        m = attention_mask.to(dtype=torch.float32)

    if m.dim() == 2:
        if m.shape[0] != M or m.shape[1] != N:
            raise ValueError(f"2D attention_mask must be [M, N] = ({M}, {N}), got {tuple(m.shape)}")
        m = m.view(1, 1, M, N)
    elif m.dim() == 3:
        if m.shape[0] != B or m.shape[1] != M or m.shape[2] != N:
            raise ValueError(
                f"3D attention_mask must be [B, M, N] = ({B}, {M}, {N}), got {tuple(m.shape)}"
            )
        m = m.unsqueeze(1)
    elif m.dim() == 4:
        if m.shape[2] != M or m.shape[3] != N:
            raise ValueError(
                f"4D attention_mask must end with [M, N] = ({M}, {N}), got {tuple(m.shape)}"
            )
    else:
        raise ValueError(
            f"attention_mask must be 2D, 3D, or 4D, got shape {tuple(attention_mask.shape)}"
        )

    try:
        out = torch.broadcast_to(m, (B, H, M, N))
    except RuntimeError as e:
        raise ValueError(
            f"Cannot broadcast attention_mask to (B, H, M, N) = ({B}, {H}, {M}, {N})"
        ) from e
    return out


def mask_add_arg(
    attention_mask: Union[torch.Tensor, None],
    B: int,
    H: int,
    M: int,
    N: int,
    *,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, int, int, int, int]]:
    """Return ``None`` or ``(tensor, s0, s1, s2, s3)`` strides for kernel."""
    if attention_mask is None:
        return None
    t = broadcast_additive_attn_mask(attention_mask, B, H, M, N, device=device)
    return t, t.stride(0), t.stride(1), t.stride(2), t.stride(3)
