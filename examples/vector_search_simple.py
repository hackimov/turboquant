"""
TurboQuant native vector search API example.

FAISS-like, in-memory ANN using `turboquant.search.VectorIndex`.

Run (CPU):
  python examples/vector_search_simple.py --device cpu
"""

import argparse
import math

import torch

from turboquant.search import VectorIndex


def _compute_scores(xq: torch.Tensor, xb: torch.Tensor, metric: str) -> torch.Tensor:
    """
    Returns scores where higher is better.

    Shapes:
      xq: [Q, d]
      xb: [N, d]
    """
    if metric == "ip":
        return xq @ xb.T
    if metric == "cosine":
        xq_n = torch.nn.functional.normalize(xq.float(), p=2, dim=-1)
        xb_n = torch.nn.functional.normalize(xb.float(), p=2, dim=-1)
        return xq_n @ xb_n.T
    if metric == "l2":
        # Higher is better => use negative squared L2 distance
        q_norm2 = torch.sum(xq.float() * xq.float(), dim=-1, keepdim=True)  # [Q,1]
        x_norm2 = torch.sum(xb.float() * xb.float(), dim=-1).unsqueeze(0)  # [1,N]
        dot = xq.float() @ xb.float().T  # [Q,N]
        return -(q_norm2 + x_norm2 - 2.0 * dot)
    raise ValueError(f"Unknown metric: {metric}")


def main() -> int:
    p = argparse.ArgumentParser(description="Vector search example for turboquant.search.VectorIndex")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--bits", type=float, default=3)
    p.add_argument("--metric", choices=("ip", "cosine", "l2"), default="ip")
    p.add_argument("--nb", type=int, default=2000, help="database size")
    p.add_argument("--nq", type=int, default=5, help="number of queries")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.02, help="query noise scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--search-chunk-size", type=int, default=4096)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create database vectors.
    xb = torch.randn(args.nb, args.dim, dtype=torch.float32)

    # Make queries correlated with some database entries (so recall is meaningful).
    pick = torch.randint(0, args.nb, (args.nq,))
    xq = xb[pick] + args.noise * torch.randn(args.nq, args.dim, dtype=torch.float32)

    # Ground truth (brute force).
    scores_true = _compute_scores(xq, xb, args.metric)  # [Q,N]
    topk_true = torch.topk(scores_true, k=min(args.k, args.nb), dim=1).indices  # [Q,K]

    index = VectorIndex(
        dim=args.dim,
        bits=args.bits,
        metric=args.metric,
        device=device,
        seed=args.seed,
        search_chunk_size=args.search_chunk_size,
    )

    index.add(xb)  # ids default to 0..N-1
    scores_pred, ids_pred = index.search(xq, k=args.k)

    # Recall@k: whether predicted id is in true top-k set for each query.
    # If k > nb, VectorIndex internally caps k, but we use ids_pred width.
    kk = ids_pred.shape[1]
    true_sets = topk_true[:, :kk]
    hit = (ids_pred.unsqueeze(-1) == true_sets.unsqueeze(1)).any(dim=-1)  # [Q,K] -> any across set
    recall_at_k = hit.float().mean().item()

    print("VectorIndex demo")
    print(f"  device={device} dim={args.dim} bits={args.bits} metric={args.metric}")
    print(f"  nb={args.nb} nq={args.nq} k={args.k} (used k={kk})")
    print(f"  recall@k ~= {recall_at_k:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

