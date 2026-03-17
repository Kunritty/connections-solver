"""
Generate a distribution JSON for the random grouping baseline on the test set.
Output matches the format of LLM/DeBERTa test_outputs.json for use in the ridge plot.
Run from repo root: python scripts/generate_random_baseline_outputs.py [seed]
"""
from __future__ import annotations

import json
import random
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import get_train_test_split, gold_groups_from_row, gold_example_groups_from_row


def _norm(g: list) -> frozenset:
    return frozenset(w.strip().upper() for w in g)


def accuracy_zero_one(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    if not isinstance(pred_groups, list) or len(pred_groups) != 4 or len(gold_groups) != 4:
        return 0.0
    if any(not isinstance(g, list) or len(g) != 4 for g in pred_groups):
        return 0.0
    all_words_pred = set(word.strip().upper() for group in pred_groups for word in group)
    all_words_gold = set(word.strip().upper() for group in gold_groups for word in group)
    if all_words_pred != all_words_gold:
        return 0.0
    pred_sets = {_norm(g) for g in pred_groups}
    gold_sets = {_norm(g) for g in gold_groups}
    return 1.0 if pred_sets == gold_sets else 0.0


def accuracy_min_swaps(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> int:
    SWAP_MAX = 12
    if not isinstance(pred_groups, list) or len(pred_groups) != 4 or len(gold_groups) != 4:
        return SWAP_MAX
    if any(not isinstance(g, list) or len(g) != 4 for g in pred_groups):
        return SWAP_MAX
    all_words_pred = set(word.strip().upper() for group in pred_groups for word in group)
    all_words_gold = set(word.strip().upper() for group in gold_groups for word in group)
    if all_words_pred != all_words_gold:
        return SWAP_MAX
    pred_sets = [_norm(g) for g in pred_groups]
    gold_sets = [_norm(g) for g in gold_groups]
    best_misplaced = 16
    for perm in permutations(range(4)):
        misplaced = sum(4 - len(pred_sets[i] & gold_sets[perm[i]]) for i in range(4))
        best_misplaced = min(best_misplaced, misplaced)
    return (best_misplaced + 1) // 2


def n_correct_groups(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> int:
    if not pred_groups or len(pred_groups) != 4 or len(gold_groups) != 4:
        return 0
    if any(len(g) != 4 for g in pred_groups) or any(len(g) != 4 for g in gold_groups):
        return 0
    pred_sets = [_norm(g) for g in pred_groups]
    gold_sets = [_norm(g) for g in gold_groups]
    return max(
        sum(1 for j in range(4) if pred_sets[j] == gold_sets[pi[j]])
        for pi in permutations(range(4))
    )


def correct_word_count(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> int:
    if not pred_groups or len(pred_groups) != 4 or len(gold_groups) != 4:
        return 0
    if any(len(g) != 4 for g in pred_groups) or any(len(g) != 4 for g in gold_groups):
        return 0
    pred_sets = [_norm(g) for g in pred_groups]
    gold_sets = [_norm(g) for g in gold_groups]
    return max(
        sum(len(pred_sets[j] & gold_sets[pi[j]]) for j in range(4))
        for pi in permutations(range(4))
    )


def random_grouping(words16: list[str], rng: random.Random) -> list[list[str]]:
    shuffled = words16[:]
    rng.shuffle(shuffled)
    return [shuffled[0:4], shuffled[4:8], shuffled[8:12], shuffled[12:16]]


def main(seed: int = 42, out_path: Path | None = None) -> None:
    if out_path is None:
        out_path = ROOT / "reports" / "random_baseline" / "random_baseline_test_outputs.json"

    _, ds_test = get_train_test_split()
    rng = random.Random(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    for i, row in enumerate(ds_test):
        words16 = row.get("words", [])
        if len(words16) != 16:
            continue
        gold = gold_groups_from_row(row)
        if len(gold) != 4:
            continue

        pred = random_grouping(words16, rng)
        z = accuracy_zero_one(pred, gold)
        s = accuracy_min_swaps(pred, gold)
        n_correct = n_correct_groups(pred, gold)
        correct_count = correct_word_count(pred, gold)
        gold_egs = gold_example_groups_from_row(row)
        levels = [eg.level for eg in gold_egs if eg.level is not None]
        valid = len(pred) == 4 and all(len(g) == 4 for g in pred)

        outputs.append({
            "index": i,
            "words16": words16,
            "pred_groups": pred,
            "gold_groups": gold,
            "zero_one": z,
            "min_swaps": s,
            "date": str(row.get("date", "")),
            "levels": levels,
            "n_correct_groups": n_correct,
            "correct_word_count": correct_count,
            "is_valid": valid,
        })

    out_path.write_text(json.dumps(outputs, indent=2))
    print(f"Wrote {len(outputs)} rows to {out_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    main(seed=seed)
