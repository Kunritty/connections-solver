from datasets import load_dataset


def load_connections_from_hf(split: str = "train"):
    """Load NYT Connections from Hugging Face (tm21cy/NYT-Connections) and return a split.

    The split has columns like: date, contest, words (16), answers (4 groups with description + words).
    """
    ds = load_dataset("tm21cy/NYT-Connections")
    if split not in ds:
        split = list(ds.keys())[0]
    return ds[split]


def gold_groups_from_row(row) -> list[list[str]]:
    """Extract the 4 answer groups (each 4 words) from a puzzle row."""
    return [list(g.get("words", [])) for g in row.get("answers", []) if len(g.get("words", [])) == 4]
