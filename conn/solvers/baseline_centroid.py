from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from conn.search import greedy_group_search
from conn.solvers.base import (
    BaseSolver,
    ExampleGroupsLike,
    example_centroids,
    tier_weight,
)

if TYPE_CHECKING:
    from data_loader import ExampleGroup


class BaselineCentroidSolver(BaseSolver):
    """Greedy grouping blending static embedding coherence and centroid similarity."""

    def __init__(
        self,
        encoder,
        example_groups: ExampleGroupsLike,
        alpha: float = 0.5,
        use_label_in_centroid: bool = True,
        use_tier_weighting: bool = True,
    ):
        super().__init__(encoder)
        self.example_groups = example_groups
        self.alpha = alpha
        self.use_label_in_centroid = use_label_in_centroid
        self.use_tier_weighting = use_tier_weighting

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        vecs = torch.stack([self.encoder.embed_phrase(w) for w in words16], dim=0)
        ex_centroids, ex_meta = example_centroids(
            self.encoder, self.example_groups, use_label=self.use_label_in_centroid
        )
        has_examples = ex_centroids.numel() > 0
        alpha = self.alpha
        use_tier = self.use_tier_weighting

        def score_fn(combo: list[int], slot: int) -> float:
            emb = vecs[combo]
            coherence = self.encoder.group_similarity(emb)
            if not has_examples:
                return coherence

            centroid = emb.mean(dim=0)
            centroid = centroid / (centroid.norm(p=2) + 1e-12)
            sims = (centroid.unsqueeze(0) @ ex_centroids.T).squeeze(0)

            if use_tier and ex_meta:
                weights = torch.tensor(
                    [tier_weight(level, slot) for _, level in ex_meta],
                    dtype=sims.dtype,
                    device=sims.device,
                )
                sims = sims * weights

            example_sim = sims.max().item()
            return (1 - alpha) * coherence + alpha * example_sim

        groups_idx = greedy_group_search(16, 4, score_fn)
        return [[words16[i] for i in idxs] for idxs in groups_idx]
