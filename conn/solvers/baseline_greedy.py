from __future__ import annotations

import torch

from conn.solvers.base import BaseSolver


class BaselineGreedySolver(BaseSolver):
    """Greedily form 4 groups of 4 by max avg cosine similarity using static embeddings."""

    def __init__(self, encoder):
        super().__init__(encoder)

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        vecs = torch.stack([self.encoder.embed_phrase(w) for w in words16], dim=0)
        return self._greedy_group(vecs, words16, self.encoder)
