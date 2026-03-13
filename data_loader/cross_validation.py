from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from data_loader.loader import load_huggingface_dataset


@dataclass
class CVFold:
    fold: int
    train: Any
    val: Any


def get_cv_folds(
    n_folds: int = 5,
    seed: int = 175,
) -> list[CVFold]:
    """Return k stratified cross-validation folds from the full dataset.

    Each fold has a disjoint validation set of size 1/n_folds, with the
    remaining puzzles as training data. Folds are consistent across calls
    with the same seed.

    Args:
        n_folds: Number of folds (k in k-fold CV).
        seed: Random seed — use the same value across all notebooks for
              reproducible, comparable splits.

    Returns:
        List of CVFold objects, each containing .fold, .train, and .val splits.
    """
    from datasets import concatenate_datasets

    ds = load_huggingface_dataset()["train"]
    ds_shuffled = ds.shuffle(seed=seed)
    n = len(ds_shuffled)

    # Compute contiguous index ranges for each val fold.
    fold_sizes = [n // n_folds + (1 if i < n % n_folds else 0) for i in range(n_folds)]
    boundaries = []
    start = 0
    for size in fold_sizes:
        boundaries.append((start, start + size))
        start += size

    val_sets = [ds_shuffled.select(range(lo, hi)) for lo, hi in boundaries]

    folds: list[CVFold] = []
    for i in range(n_folds):
        train = concatenate_datasets([val_sets[j] for j in range(n_folds) if j != i])
        folds.append(CVFold(fold=i, train=train, val=val_sets[i]))

    return folds


def get_cv_fold(
    fold: int,
    n_folds: int = 5,
    seed: int = 175,
) -> CVFold:
    """Return a single cross-validation fold by index.

    Args:
        fold: Zero-based fold index (0 to n_folds-1).
        n_folds: Total number of folds.
        seed: Random seed.

    Returns:
        CVFold with .train and .val HuggingFace dataset splits.
    """
    if not (0 <= fold < n_folds):
        raise ValueError(f"fold must be in [0, {n_folds - 1}], got {fold}")
    return get_cv_folds(n_folds=n_folds, seed=seed)[fold]
