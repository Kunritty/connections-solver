from itertools import combinations

import torch
from torch.nn.functional import normalize


def solve_puzzle_zero_shot(words16: list[str], encoder) -> list[list[str]]:
    """Greedily form 4 groups of 4 by max avg cosine similarity."""
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
    vecs = torch.stack([encoder.embed_phrase(w) for w in words16], dim=0)
    remaining = list(range(16))
    groups_idx: list[list[int]] = []
    for _ in range(3):
        best_score = float("-inf")
        best_combo = None
        for combo in combinations(remaining, 4):
            emb = vecs[list(combo)]
            score = encoder.group_similarity(emb)
            if score > best_score:
                best_score = score
                best_combo = list(combo)
        groups_idx.append(best_combo)
        remaining = [i for i in remaining if i not in best_combo]
    groups_idx.append(remaining)
    return [[words16[i] for i in idxs] for idxs in groups_idx]


def _example_centroids(encoder, example_groups: list[list[str]]) -> torch.Tensor:
    """Return [num_groups, dim] L2-normalized centroids."""
    if not example_groups:
        return torch.empty(0, 0)
    centroids = []
    for group in example_groups:
        if len(group) != 4:
            continue
        vecs = torch.stack([encoder.embed_phrase(w) for w in group], dim=0)
        c = vecs.mean(dim=0)
        c = c / (c.norm(p=2) + 1e-12)
        centroids.append(c)
    if not centroids:
        return torch.empty(0, 0)
    return torch.stack(centroids, dim=0)


def solve_puzzle_few_shot(
    words16: list[str],
    encoder,
    example_groups: list[list[str]],
    alpha: float = 0.5,
) -> list[list[str]]:
    """Same greedy loop; score = (1-alpha)*group_similarity + alpha*max_similarity_to_example_centroids."""
    if len(words16) != 16:
        raise ValueError(f"Expected 16 words, got {len(words16)}")
    vecs = torch.stack([encoder.embed_phrase(w) for w in words16], dim=0)
    example_centroids = _example_centroids(encoder, example_groups)
    use_examples = example_centroids.numel() > 0

    remaining = list(range(16))
    groups_idx: list[list[int]] = []

    for _ in range(3):
        best_score = float("-inf")
        best_combo = None
        for combo in combinations(remaining, 4):
            emb = vecs[list(combo)]
            coherence = encoder.group_similarity(emb)
            if use_examples:
                centroid = emb.mean(dim=0)
                centroid = centroid / (centroid.norm(p=2) + 1e-12)
                example_sim = (centroid.unsqueeze(0) @ example_centroids.T).squeeze(0).max().item()
                score = (1 - alpha) * coherence + alpha * example_sim
            else:
                score = coherence
            if score > best_score:
                best_score = score
                best_combo = list(combo)
        groups_idx.append(best_combo)
        remaining = [i for i in remaining if i not in best_combo]
    groups_idx.append(remaining)
    return [[words16[i] for i in idxs] for idxs in groups_idx]
