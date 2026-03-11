from __future__ import annotations

from typing import TYPE_CHECKING

from conn.solvers.base import BaseSolver, ExampleGroupsLike, example_words

if TYPE_CHECKING:
    from data_loader import ExampleGroup


class FewShotSolver(BaseSolver):
    """Uses DeBERTa contextualized embeddings with few-shot examples injected into the prompt."""

    def __init__(self, encoder, example_groups: ExampleGroupsLike | None = None):
        super().__init__(encoder)
        self.example_groups = example_groups

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        prompt = self._build_prompt()
        vecs = self.encoder.embed_board(words16, prompt_prefix=prompt)
        return self._greedy_group(vecs, words16, self.encoder)

    def _build_prompt(self) -> str:
        lines: list[str] = [
            "You are solving NYT Connections. Each category has 4 related words.",
            "Below are examples written in a masked-LM style (predict the missing 4th word).",
            "",
        ]

        if self.example_groups:
            subset = self.example_groups[:10] if len(self.example_groups) > 10 else self.example_groups
            lines.extend(self._format_example(eg) for eg in subset)
            lines.append("")

        lines.extend([
            "Now, for the next board, form 4 groups of 4 words by semantic category.",
            "BOARD WORDS:",
        ])
        return "\n".join(lines) + "\n"

    def _format_example(self, eg: ExampleGroup | list[str]) -> str:
        words = example_words(eg)
        label = getattr(eg, "label", "") if hasattr(eg, "label") else ""
        label = str(label).strip() or "UNKNOWN"
        mask = self._mask_token()
        return f"CATEGORY: {label} | WORDS: {', '.join(words[:3])}, {mask}"

    def _mask_token(self) -> str:
        tok = getattr(self.encoder, "tokenizer", None)
        mask = getattr(tok, "mask_token", None) if tok is not None else None
        return mask or "[MASK]"
