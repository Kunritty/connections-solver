from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

from conn.solvers.base import BaseSolver, ExampleGroupsLike, example_words

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"

SYSTEM_PROMPT = (
    "Your task is to partition the 16 words into 4 groups of 4 words/phrases "
    "based on shared connections.\n"
    "Output requirements (STRICT):\n"
    "OUTPUT ONLY the final groups of words/phrases.\n"
    "Do NOT provide reasoning or explanations under any circumstances.\n"
    "DO NOT output any text other than the 4 groups.\n"
    "Use ONLY the EXACT words/phrases from the puzzle.\n"
    "Make sure there are EXACTLY 4 groups of 4 words/phrases each with their "
    "category names. NO EXCEPTIONS.\n"
    "Return the answer exactly in this format:\n\n"
    "GROUP 1: word1, word2, word3, word4\n"
    "GROUP 2: word1, word2, word3, word4\n"
    "GROUP 3: word1, word2, word3, word4\n"
    "GROUP 4: word1, word2, word3, word4\n"
)

_GROUP_PATTERN = re.compile(r"GROUP \d+:\s*(.+)")


def _parse_response(response: str) -> list[list[str]]:
    groups = _GROUP_PATTERN.findall(response)
    # Stop parsing if we've already found 4 groups (avoids repetition)
    return [[w.strip() for w in group.split(",")] for group in groups[:4]]


def _make_few_shot_section(
    words16: list[str],
    example_source: Any,
    k: int,
) -> str:
    """Build the few-shot examples portion of the prompt.

    ``example_source`` can be either:
    * A HuggingFace dataset split (rows with "words" and "answers" keys) — the
      same format used in LLM-model-few-shot.ipynb.
    * An ``ExampleGroupsLike`` list (list[ExampleGroup] | list[list[str]]).
      In this case the groups are formatted directly (no leakage check needed
      because ExampleGroups don't carry the full 16-word board).
    """
    if not example_source or k <= 0:
        return ""

    lines: list[str] = ["Here are some previous examples:"]

    if _is_hf_split(example_source):
        sampled = random.sample(list(example_source), min(k + 5, len(example_source)))
        count = 0
        for row in sampled:
            if count >= k:
                break
            fs_words: list[str] = row["words"]
            if set(fs_words) == set(words16):
                continue
            fs_groups = [ans["words"] for ans in row["answers"]]
            ex = f"Here are 16 words: {', '.join(fs_words)}"
            for i, group in enumerate(fs_groups, 1):
                ex += f"\nGROUP {i}: {', '.join(group)}"
            lines.append(ex + "\n")
            count += 1
    else:
        sampled_groups = random.sample(
            list(example_source),
            min(k * 4, len(example_source)),
        )
        chunk: list[list[str]] = []
        for eg in sampled_groups:
            chunk.append(example_words(eg))
            if len(chunk) == 4:
                all_words = [w for g in chunk for w in g]
                ex = f"Here are 16 words: {', '.join(all_words)}"
                for i, g in enumerate(chunk, 1):
                    ex += f"\nGROUP {i}: {', '.join(g)}"
                lines.append(ex + "\n")
                chunk = []

    return "\n".join(lines) + "\n"


def _is_hf_split(obj: Any) -> bool:
    first = obj[0] if len(obj) > 0 else None
    return isinstance(first, dict) and "words" in first and "answers" in first


def _valid_prediction(pred_groups: list[list[str]], words16: list[str]) -> bool:
    if len(pred_groups) != 4:
        return False
    if any(len(g) != 4 for g in pred_groups):
        return False
    pred_words = set(w.strip().upper() for g in pred_groups for w in g)
    expected = set(w.strip().upper() for w in words16)
    return pred_words == expected


class LlamaSolver(BaseSolver):
    """Generative LLaMA solver with optional few-shot prompting.

    Works with both a vanilla HuggingFace model and a LoRA-adapted one.
    Authentication (huggingface_hub.login) should be done externally
    (e.g. in the notebook) before constructing this solver.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        example_source: Any = None,
        k: int = 0,
        max_new_tokens: int = 300,
        temperature: float = 0.1,
        max_retries: int = 0,
        retry_temperature_step: float = 0.1,
    ):
        super().__init__(encoder=None)
        self.model = model
        self.tokenizer = tokenizer
        self.example_source = example_source
        self.k = k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_temperature_step = retry_temperature_step

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def solve(self, words16: list[str]) -> list[list[str]]:
        self._validate_board(words16)
        prompt = self._build_full_prompt(words16)

        for attempt in range(1 + self.max_retries):
            temp = self.temperature + self.retry_temperature_step * attempt
            response = self._generate(prompt, temperature_override=temp)
            self.last_raw_response = response
            parsed = _parse_response(response)

            if _valid_prediction(parsed, words16):
                return parsed

            if attempt < self.max_retries:
                print(
                    f"Invalid LLM output, retrying ({attempt + 1}/{self.max_retries})"
                )

        return parsed

    def _build_full_prompt(self, words16: list[str]) -> str:
        few_shot = _make_few_shot_section(words16, self.example_source, self.k)
        user_body = (
            f"NOW, solve this puzzle and only this one: "
            f"Here are the 16 words: {', '.join(words16)}"
        )
        user_prompt = few_shot + user_body
        return SYSTEM_PROMPT + "\n" + user_prompt

    def _generate(self, prompt: str, temperature_override: float | None = None) -> str:
        device = next(self.model.parameters()).device
        
        forced_prefix = "\nGROUP 1:"
        inputs = self.tokenizer(prompt + forced_prefix, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1]

        temp = temperature_override if temperature_override is not None else self.temperature
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temp > 0:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = 0.9

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][prompt_len:]
        
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = "GROUP 1:" + raw_output
        return response
