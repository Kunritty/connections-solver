from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExampleGroup:
    words: list[str]
    label: str = ""
    level: int | None = None


def gold_groups_from_row(row) -> list[list[str]]:
    """Extract the 4 answer groups (each 4 words) from a puzzle row."""
    return [list(g.get("words", [])) for g in row.get("answers", []) if len(g.get("words", [])) == 4]


def gold_example_groups_from_row(row) -> list[ExampleGroup]:
    """Extract the 4 answer groups with label and level metadata."""
    out: list[ExampleGroup] = []
    for idx, g in enumerate(row.get("answers", [])):
        words = list(g.get("words", []))
        if len(words) != 4:
            continue
        level = g.get("level")
        if level is None:
            level = idx
        out.append(ExampleGroup(
            words=words,
            label=g.get("answerDescription", ""),
            level=level,
        ))
    return out


def load_example_groups_from_csv(path: str | Path) -> list[ExampleGroup]:
    """Load example groups from a CSV with columns groupName, level, members."""
    path = Path(path)
    out: list[ExampleGroup] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            members = row.get("members", "")
            words = [w.strip() for w in members.split(",") if w.strip()]
            if len(words) != 4:
                continue
            level_str = row.get("level", "0")
            try:
                level = int(level_str)
            except ValueError:
                level = 0
            out.append(ExampleGroup(
                words=words,
                label=row.get("groupName", ""),
                level=level,
            ))
    return out
