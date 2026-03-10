from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from conn.search import greedy_group_search

if TYPE_CHECKING:
    from conn.data import ExampleGroup

ExampleGroupsLike = list["ExampleGroup"] | list[list[str]]


def _example_words(eg: ExampleGroup | list[str]) -> list[str]:
    return eg.words if hasattr(eg, "words") else list(eg)


def solve_baseline_greedy(words16: list[str], encoder) -> list[list[str]]:
    """Greedily form 4 groups of 4 by max avg cosine similarity using static embeddings."""
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
    
    vecs = torch.stack([encoder.embed_phrase(w) for w in words16], dim=0)
    
    def score_fn(combo: list[int], slot: int) -> float:
        return encoder.group_similarity(vecs[combo])

    groups_idx = greedy_group_search(16, 4, score_fn)
    return [[words16[i] for i in idxs] for idxs in groups_idx]


def _example_centroids(
    encoder,
    example_groups: ExampleGroupsLike,
    use_label: bool = True,
) -> tuple[torch.Tensor, list[tuple[str, int | None]]]:
    """Return [num_groups, dim] L2-normalized centroids and list of (label, level) per example."""
    from conn.data import ExampleGroup

    if not example_groups:
        return torch.empty(0, 0), []
    centroids = []
    metadata: list[tuple[str, int | None]] = []
    for group in example_groups:
        words = _example_words(group)
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


def _tier_weight(example_level: int | None, slot: int) -> float:
    if example_level is None:
        return 1.0
    return 1.0 if example_level == slot else 0.5


def solve_baseline_centroid(
    words16: list[str],
    encoder,
    example_groups: ExampleGroupsLike,
    alpha: float = 0.5,
    use_label_in_centroid: bool = True,
    use_tier_weighting: bool = True,
) -> list[list[str]]:
    """Greedy grouping blending static embedding coherence and centroid similarity."""
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
        
    vecs = torch.stack([encoder.embed_phrase(w) for w in words16], dim=0)
    example_centroids, example_meta = _example_centroids(
        encoder, example_groups, use_label=use_label_in_centroid
    )
    use_examples = example_centroids.numel() > 0

    def score_fn(combo: list[int], slot: int) -> float:
        emb = vecs[combo]
        coherence = encoder.group_similarity(emb)
        if not use_examples:
            return coherence
            
        centroid = emb.mean(dim=0)
        centroid = centroid / (centroid.norm(p=2) + 1e-12)
        sims = (centroid.unsqueeze(0) @ example_centroids.T).squeeze(0)
        
        if use_tier_weighting and example_meta:
            weights = torch.tensor(
                [_tier_weight(level, slot) for _, level in example_meta],
                dtype=sims.dtype,
                device=sims.device,
            )
            sims = sims * weights
            
        example_sim = sims.max().item()
        return (1 - alpha) * coherence + alpha * example_sim

    groups_idx = greedy_group_search(16, 4, score_fn)
    return [[words16[i] for i in idxs] for idxs in groups_idx]


def solve_zero_shot(
    words16: list[str],
    encoder,
) -> list[list[str]]:
    """
    Properly uses DeBERTa by passing the entire board as context (no prior examples).
    Extracts contextualized embeddings and applies greedy grouping.
    """
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
        
    prompt = "\nNow group the following words based on their context:\n"
    
    vecs = encoder.embed_board(words16, prompt_prefix=prompt)
    
    def score_fn(combo: list[int], slot: int) -> float:
        # With contextualized embeddings, the coherence of the vectors
        # is much more robust as the attention has already resolved ambiguity.
        return encoder.group_similarity(vecs[combo])
        
    groups_idx = greedy_group_search(16, 4, score_fn)
    return [[words16[i] for i in idxs] for idxs in groups_idx]


def solve_few_shot(
    words16: list[str],
    encoder,
    example_groups: ExampleGroupsLike = None,
) -> list[list[str]]:
    """
    Properly uses DeBERTa by passing the entire board as context, injecting examples into the prompt.
    Extracts contextualized embeddings and applies greedy grouping.
    """
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
        
    prompt = ""
    if example_groups:
        prompt = "Here are examples of valid semantic groups:\n"
        # Use a reasonable subset if there are many examples to prevent truncation
        subset = example_groups[:10] if len(example_groups) > 10 else example_groups
        for eg in subset:
            words = _example_words(eg)
            label = getattr(eg, "label", "") if hasattr(eg, "label") else ""
            if label:
                prompt += f"- {label}: {', '.join(words)}\n"
            else:
                prompt += f"- {', '.join(words)}\n"
        prompt += "\nNow group the following words based on their context:\n"
    else:
        prompt = "\nNow group the following words based on their context:\n"
    
    vecs = encoder.embed_board(words16, prompt_prefix=prompt)
    
    def score_fn(combo: list[int], slot: int) -> float:
        return encoder.group_similarity(vecs[combo])
        
    groups_idx = greedy_group_search(16, 4, score_fn)
    return [[words16[i] for i in idxs] for idxs in groups_idx]


def solve_puzzle_random_grouping(words16: list[str], seed: int | None = None) -> list[list[str]]:
    """Randomly form 4 groups of 4"""
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")

    shuffled = words16[:]
    random.Random(seed).shuffle(shuffled)

    return [
        shuffled[0:4],
        shuffled[4:8],
        shuffled[8:12],
        shuffled[12:16],
    ]
