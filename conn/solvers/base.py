from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from conn.search import greedy_group_search

if TYPE_CHECKING:
    from data_loader import ExampleGroup

ExampleGroupsLike = list["ExampleGroup"] | list[list[str]]


class BaseSolver(ABC):
    """Abstract base class for all Connections puzzle solvers."""

    def __init__(self, encoder=None):
        self.encoder = encoder

    @abstractmethod
    def solve(self, words16: list[str]) -> list[list[str]]:
        ...

    @staticmethod
    def _validate_board(words16: list[str]) -> None:
        if len(words16) != 16:
            raise ValueError(f"Expected 16 words, got {len(words16)}")

    @staticmethod
    def _greedy_group(
        vecs: torch.Tensor,
        words16: list[str],
        encoder,
    ) -> list[list[str]]:
        def score_fn(combo: list[int], slot: int) -> float:
            return encoder.group_similarity(vecs[combo])

        groups_idx = greedy_group_search(16, 4, score_fn)
        return [[words16[i] for i in idxs] for idxs in groups_idx]


def example_words(eg: ExampleGroup | list[str]) -> list[str]:
    return eg.words if hasattr(eg, "words") else list(eg)


def example_centroids(
    encoder,
    example_groups: ExampleGroupsLike,
    use_label: bool = True,
) -> tuple[torch.Tensor, list[tuple[str, int | None]]]:
    """Return [num_groups, dim] L2-normalized centroids and (label, level) metadata."""
    from data_loader import ExampleGroup

    if not example_groups:
        return torch.empty(0, 0), []
    centroids = []
    metadata: list[tuple[str, int | None]] = []
    for group in example_groups:
        words = example_words(group)
        if len(words) != 4:
            continue
        vecs = torch.stack([encoder.embed_phrase(w) for w in words], dim=0)
        c = vecs.mean(dim=0)
        if use_label and isinstance(group, ExampleGroup) and group.label:
            label_vec = encoder.embed_phrase(group.label)
            c = c + label_vec
        c = c / (c.norm(p=2) + 1e-12)
        centroids.append(c)
        label = getattr(group, "label", "") if hasattr(group, "label") else ""
        level = getattr(group, "level", None) if hasattr(group, "level") else None
        metadata.append((label, level))
    if not centroids:
        return torch.empty(0, 0), []
    return torch.stack(centroids, dim=0), metadata


def tier_weight(example_level: int | None, slot: int) -> float:
    if example_level is None:
        return 1.0
    return 1.0 if example_level == slot else 0.5
