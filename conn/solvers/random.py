from __future__ import annotations

import random

from conn.solvers.base import BaseSolver


class RandomSolver(BaseSolver):
    """Randomly form 4 groups of 4 (baseline)."""

    def __init__(self, seed: int | None = None):
        super().__init__(encoder=None)
        self.seed = seed

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        shuffled = words16[:]
        random.Random(self.seed).shuffle(shuffled)
        return [
            shuffled[0:4],
            shuffled[4:8],
            shuffled[8:12],
            shuffled[12:16],
        ]
