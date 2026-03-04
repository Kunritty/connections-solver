from itertools import permutations

from conn.data import gold_groups_from_row


def _norm(g: list) -> frozenset:
    return frozenset(w.strip() for w in g)


def accuracy_zero_one(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    if len(pred_groups) != 4 or len(gold_groups) != 4:
        return 0.0
    pred_sets = {_norm(g) for g in pred_groups}
    gold_sets = {_norm(g) for g in gold_groups}
    return 1.0 if pred_sets == gold_sets else 0.0


def accuracy_min_swaps(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    if len(pred_groups) != 4 or len(gold_groups) != 4:
        return float("inf")
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
    verbose=False
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
        pred = solver_fn(words16)
        
        for name, fn in metric_fns.items():
            scores[name].append(fn(pred, gold))

        if verbose and (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{n} samples")

    results = {
        name: (sum(vals) / len(vals) if vals else 0.0, len(vals))
        for name, vals in scores.items()
    }
    return results
