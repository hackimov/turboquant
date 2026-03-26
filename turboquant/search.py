from typing import Optional, Tuple

import torch

from .core import TurboQuantProd


class VectorIndex:
    """
    TurboQuant-based vector search index (FAISS-like API surface).

    Index stores vectors in TurboQuant compressed form and performs approximate search
    by chunk-wise dequantization during query time.
    """

    def __init__(
        self,
        dim: int,
        *,
        bits: float = 3,
        metric: str = "ip",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        search_chunk_size: int = 4096,
    ) -> None:
        if dim <= 1:
            raise ValueError("dim must be > 1")
        if metric not in ("ip", "cosine", "l2"):
            raise ValueError("metric must be one of: ip, cosine, l2")
        if search_chunk_size <= 0:
            raise ValueError("search_chunk_size must be > 0")

        self.dim = int(dim)
        self.metric = metric
        self.search_chunk_size = int(search_chunk_size)
        self.quantizer = TurboQuantProd(bits=bits, head_dim=dim, device=device, dtype=dtype, seed=seed)

        self._idx: Optional[torch.Tensor] = None
        self._norm: Optional[torch.Tensor] = None
        self._sign: Optional[torch.Tensor] = None
        self._gamma: Optional[torch.Tensor] = None
        self._ids = torch.empty(0, dtype=torch.long)
        self._next_id = 0

    @property
    def ntotal(self) -> int:
        return int(self._ids.numel())

    def reset(self) -> None:
        self._idx = None
        self._norm = None
        self._sign = None
        self._gamma = None
        self._ids = torch.empty(0, dtype=torch.long)
        self._next_id = 0

    def add(self, vectors: torch.Tensor, ids: Optional[torch.Tensor] = None) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"vectors must have shape [N, {self.dim}]")
        if vectors.shape[0] == 0:
            return

        vectors = vectors.to(self.quantizer.device, dtype=self.quantizer.dtype)
        _, idx, norm, sign, gamma = self.quantizer.quantize(vectors)

        n = vectors.shape[0]
        if ids is None:
            ids = torch.arange(self._next_id, self._next_id + n, dtype=torch.long)
        else:
            ids = ids.reshape(-1).to(dtype=torch.long).cpu()
            if ids.numel() != n:
                raise ValueError("ids length must match number of vectors")
            if torch.unique(ids).numel() != ids.numel():
                raise ValueError("ids must be unique inside one add() call")

        if self._idx is None:
            self._idx = idx
            self._norm = norm
            self._sign = sign
            self._gamma = gamma
        else:
            self._idx = torch.cat([self._idx, idx], dim=0)
            self._norm = torch.cat([self._norm, norm], dim=0)
            self._sign = torch.cat([self._sign, sign], dim=0)
            self._gamma = torch.cat([self._gamma, gamma], dim=0)

        self._ids = torch.cat([self._ids, ids], dim=0)
        self._next_id = int(max(self._next_id, int(self._ids.max().item()) + 1))

    def search(self, queries: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.ntotal == 0:
            raise ValueError("index is empty; call add() first")
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"queries must have shape [Q, {self.dim}]")
        if k <= 0:
            raise ValueError("k must be > 0")

        k = min(k, self.ntotal)
        q = queries.to(self.quantizer.device, dtype=self.quantizer.dtype)
        if self.metric == "cosine":
            q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(dtype=self.quantizer.dtype)

        best_scores = torch.full((q.shape[0], k), float("-inf"), device=self.quantizer.device, dtype=torch.float32)
        best_ids = torch.full((q.shape[0], k), -1, device=self.quantizer.device, dtype=torch.long)

        for start in range(0, self.ntotal, self.search_chunk_size):
            end = min(start + self.search_chunk_size, self.ntotal)
            idx_chunk = self._idx[start:end]
            norm_chunk = self._norm[start:end]
            sign_chunk = self._sign[start:end]
            gamma_chunk = self._gamma[start:end]
            db_chunk = self.quantizer.dequantize(idx_chunk, norm_chunk, sign_chunk, gamma_chunk)

            if self.metric == "cosine":
                db_chunk = torch.nn.functional.normalize(db_chunk.float(), p=2, dim=-1).to(dtype=self.quantizer.dtype)

            dot = torch.matmul(q, db_chunk.T).float()
            if self.metric == "l2":
                q_norm2 = torch.sum(q.float() * q.float(), dim=-1, keepdim=True)
                x_norm2 = torch.sum(db_chunk.float() * db_chunk.float(), dim=-1).unsqueeze(0)
                score_chunk = -(q_norm2 + x_norm2 - 2.0 * dot)
            else:
                score_chunk = dot

            ids_chunk = self._ids[start:end].to(device=self.quantizer.device)
            ids_chunk = ids_chunk.unsqueeze(0).expand(q.shape[0], -1)

            merged_scores = torch.cat([best_scores, score_chunk], dim=1)
            merged_ids = torch.cat([best_ids, ids_chunk], dim=1)
            top_scores, top_pos = torch.topk(merged_scores, k=k, dim=1)
            top_ids = torch.gather(merged_ids, 1, top_pos)
            best_scores, best_ids = top_scores, top_ids

        return best_scores.cpu(), best_ids.cpu()
