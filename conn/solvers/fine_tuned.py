from __future__ import annotations

from conn.solvers.base import ExampleGroupsLike
from conn.solvers.few_shot import FewShotSolver


class FineTunedSolver(FewShotSolver):
    """Solver that uses a LoRA fine-tuned encoder via the few-shot pipeline."""

    def __init__(self, encoder, example_groups: ExampleGroupsLike | None = None):
        super().__init__(encoder, example_groups)
