import time
from itertools import permutations

from data_loader import gold_groups_from_row

SWAP_MAX_VALUE = 12

def _norm(g: list) -> frozenset:
    return frozenset(w.strip().upper() for w in g)

def _is_valid_prediction(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> bool:
    if not isinstance(pred_groups, list) or len(pred_groups) != 4 or len(gold_groups) != 4:
        return False
    if any(not isinstance(g, list) or len(g) != 4 for g in pred_groups):
        return False
    all_words_pred = set(word.strip().upper() for group in pred_groups for word in group)
    all_words_gold = set(word.strip().upper() for group in gold_groups for word in group)
    return all_words_pred == all_words_gold

def accuracy_zero_one(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    # If the output does not match the 4-group structure (or the correct subset of words),
    # punish by assigning 0 accuracy.
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


def n_correct_groups(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> int:
    """Number of predicted groups that exactly match some gold group (best permutation)."""
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
    """Max number of words in correct position over all 4! alignments of pred to gold."""
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


def accuracy_min_swaps(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    # If the output does not match the 4-group structure (or the correct subset of words),
    # punish by assigning the max possible swaps.
    if not isinstance(pred_groups, list) or len(pred_groups) != 4 or len(gold_groups) != 4:
        return float(SWAP_MAX_VALUE)
    if any(not isinstance(g, list) or len(g) != 4 for g in pred_groups):
        return float(SWAP_MAX_VALUE)
    
    # Also check if it's the correct subset of words (hallucinations)
    all_words_pred = set(word.strip().upper() for group in pred_groups for word in group)
    all_words_gold = set(word.strip().upper() for group in gold_groups for word in group)
    if all_words_pred != all_words_gold:
        return float(SWAP_MAX_VALUE)
        
    pred_sets = [_norm(g) for g in pred_groups]
    gold_sets = [_norm(g) for g in gold_groups]
    best_misplaced = 16
    for perm in permutations(range(4)):
        misplaced = 0
        for i in range(4):
            j = perm[i]
            misplaced += 4 - len(pred_sets[i] & gold_sets[j])
        best_misplaced = min(best_misplaced, misplaced)
    return (best_misplaced + 1) // 2


def evaluate(
    split,
    metric_fns=None,
    solver_fn=None,
    max_samples=None,
    gold_from_row=None,
    verbose=False,
    verbose_every=10
):
    if metric_fns is None:
        metric_fns = {"accuracy_zero_one": accuracy_zero_one}
    if solver_fn is None:
        raise ValueError("solver_fn is required")
    if callable(metric_fns):
        metric_fns = {metric_fns.__name__: metric_fns}
    if isinstance(metric_fns, list):
        metric_fns = {fn.__name__: fn for fn in metric_fns}
    if gold_from_row is None:
        gold_from_row = gold_groups_from_row
    
    scores = {name: [] for name in metric_fns.keys()}
    n = len(split) if max_samples is None else min(max_samples, len(split))
    for i in range(n):
        row = split[i]
        words16 = row.get("words", [])
        if len(words16) != 16:
            continue
        gold = gold_from_row(row)
        if len(gold) != 4:
            continue
        if verbose:
            t0 = time.perf_counter()
            ts = time.strftime("%H:%M:%S", time.localtime())
        pred = solver_fn(words16)
        if verbose:
            elapsed = time.perf_counter() - t0
            ts_end = time.strftime("%H:%M:%S", time.localtime())
            gen_part = ""
            if hasattr(solver_fn, "__self__") and hasattr(solver_fn.__self__, "last_generate_seconds"):
                gen_part = f" generate={solver_fn.__self__.last_generate_seconds:.1f}s"
            print(f"[{ts} → {ts_end} | {elapsed:.1f}s{gen_part}] Puzzle {i+1} solved, min_swaps={accuracy_min_swaps(pred, gold)}")
        if not _is_valid_prediction(pred, gold):
            if verbose:
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] WARNING: Invalid or hallucinated output for puzzle {i+1}: {pred}")
                print(f"EXPECTED: {gold}")
                if hasattr(solver_fn, "__self__") and hasattr(solver_fn.__self__, "last_raw_response"):
                    print(f"--- RAW LLM OUTPUT ---\n{solver_fn.__self__.last_raw_response}\n----------------------")
        
        for name, fn in metric_fns.items():
            scores[name].append(fn(pred, gold))

        if verbose and verbose_every > 0 and (i + 1) % verbose_every == 0:
            print(f"[{time.strftime('%H:%M:%S', time.localtime())}] Processed {i + 1}/{n} samples")

    results = {
        name: (sum(vals) / len(vals) if vals else 0.0, len(vals))
        for name, vals in scores.items()
    }
    return results
