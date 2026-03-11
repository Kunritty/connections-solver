from __future__ import annotations

from conn.solvers.base import BaseSolver


class ZeroShotSolver(BaseSolver):
    """Uses DeBERTa contextualized embeddings over the full board (no prior examples)."""

    def __init__(self, encoder):
        super().__init__(encoder)

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        prompt = "\nNow group the following words based on their context:\n"
        vecs = self.encoder.embed_board(words16, prompt_prefix=prompt)
        return self._greedy_group(vecs, words16, self.encoder)
