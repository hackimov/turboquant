import unittest

import torch

from turboquant.search import VectorIndex


class TestVectorIndex(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 32

    def test_add_and_search_ip(self):
        torch.manual_seed(0)
        index = VectorIndex(dim=self.dim, bits=3, metric="ip", device=self.device, seed=1, search_chunk_size=16)

        base = torch.randn(50, self.dim)
        queries = base[:5].clone()
        index.add(base)

        scores, ids = index.search(queries, k=5)
        self.assertEqual(scores.shape, (5, 5))
        self.assertEqual(ids.shape, (5, 5))
        self.assertEqual(index.ntotal, 50)

    def test_add_with_ids(self):
        torch.manual_seed(1)
        index = VectorIndex(dim=self.dim, bits=3, metric="cosine", device=self.device, seed=2)
        vectors = torch.randn(10, self.dim)
        ids = torch.arange(100, 110)
        index.add(vectors, ids=ids)
        _, got_ids = index.search(vectors[:2], k=3)
        self.assertTrue(torch.all(got_ids >= 100))

    def test_reset(self):
        index = VectorIndex(dim=self.dim, bits=3, metric="l2", device=self.device, seed=3)
        index.add(torch.randn(8, self.dim))
        self.assertEqual(index.ntotal, 8)
        index.reset()
        self.assertEqual(index.ntotal, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
