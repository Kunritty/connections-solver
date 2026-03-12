from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from conn.solvers.llama import SYSTEM_PROMPT

try:
    from peft import LoraConfig, TaskType, PeftModel, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment,misc]
    TaskType = None  # type: ignore[assignment,misc]
    PeftModel = None  # type: ignore[assignment,misc]
    get_peft_model = None  # type: ignore[assignment,misc]

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"


@dataclass
class LlamaFineTuneStats:
    epochs: int
    steps: int
    average_losses: list[float]


# ---------------------------------------------------------------------------
# SFT Dataset — prompt/target text pairs for causal-LM fine-tuning
# ---------------------------------------------------------------------------

def _format_target(groups: list[list[str]]) -> str:
    lines: list[str] = []
    for i, group in enumerate(groups, 1):
        lines.append(f"GROUP {i}: {' || '.join(group)}")
    return "\n".join(lines)


def _format_prompt(words16: list[str], few_shot_text: str = "") -> str:
    user_body = (
        f"NOW, solve this puzzle and only this one: "
        f"Here are the 16 words: {' || '.join(words16)}"
    )
    return SYSTEM_PROMPT + "\n" + few_shot_text + user_body


def _extract_groups(puzzle: Any) -> list[list[str]]:
    if isinstance(puzzle, dict):
        if "answers" in puzzle:
            return [
                list(a.get("words", []))
                for a in puzzle.get("answers", [])
                if len(a.get("words", [])) == 4
            ]
        if "groups" in puzzle:
            return [list(g) for g in puzzle.get("groups", []) if len(g) == 4]
    return []


def _extract_words(puzzle: Any) -> list[str]:
    if isinstance(puzzle, dict) and "words" in puzzle:
        return list(puzzle["words"])
    groups = _extract_groups(puzzle)
    return [w for g in groups for w in g]


class ConnectionsSFTDataset(Dataset):
    """Yields (prompt_text, target_text) pairs for causal-LM SFT."""

    def __init__(self, puzzles: list[Any], seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._examples = self._build(puzzles)
        if not self._examples:
            raise ValueError("No valid SFT examples built from the provided puzzles.")

    def _build(self, puzzles: list[Any]) -> list[tuple[str, str]]:
        examples: list[tuple[str, str]] = []
        for puzzle in puzzles:
            groups = _extract_groups(puzzle)
            if len(groups) != 4:
                continue
            words16 = _extract_words(puzzle)
            if len(words16) != 16:
                continue
            shuffled = list(words16)
            self._rng.shuffle(shuffled)
            prompt = _format_prompt(shuffled)
            target = _format_target(groups)
            examples.append((prompt, target))
        return examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self._examples[idx]


def _collate_sft(
    batch: list[tuple[str, str]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt+target, masking prompt tokens in the labels."""
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    for prompt, target in batch:
        full_text = prompt + "\n" + target + tokenizer.eos_token
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(labels) < len(full_ids):
            labels = labels + [-100] * (len(full_ids) - len(labels))
        labels = labels[: len(full_ids)]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ---------------------------------------------------------------------------
# LoRA model builder for causal LM
# ---------------------------------------------------------------------------

def _build_llama_lora_model(
    model_name: str,
    device: torch.device,
    r: int,
    alpha: int,
    dropout: float,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA fine-tuning. Install with `pip install peft`.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------

def finetune_llama_lora(
    puzzles: list[Any],
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    max_length: int = 1024,
    seed: int = 42,
    adapter_output_dir: str | Path | None = None,
    verbose: bool = False,
) -> tuple[Any, Any, LlamaFineTuneStats]:
    """Fine-tune LLaMA with LoRA on Connections puzzles using SFT.

    Returns (model, tokenizer, stats).
    """
    if not puzzles:
        raise ValueError("puzzles must not be empty.")

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    if verbose:
        print(f"Starting LLaMA LoRA SFT on {device} for {epochs} epochs...")

    model, tokenizer = _build_llama_lora_model(
        model_name=model_name,
        device=device,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    dataset = ConnectionsSFTDataset(puzzles=puzzles, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=lambda batch: _collate_sft(batch, tokenizer, max_length),
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    model.train()
    step_count = 0
    average_losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            step_count += 1

        epoch_avg_loss = running_loss / max(1, batch_count)
        average_losses.append(epoch_avg_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {epoch_avg_loss:.4f}")

    if adapter_output_dir is not None:
        out_path = Path(adapter_output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        if verbose:
            print(f"Adapter saved to {out_path}")

    model.eval()
    stats = LlamaFineTuneStats(epochs=epochs, steps=step_count, average_losses=average_losses)
    return model, tokenizer, stats


# ---------------------------------------------------------------------------
# Load a previously saved LoRA adapter
# ---------------------------------------------------------------------------

def load_llama_lora(
    adapter_dir: str | Path,
    base_model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[Any, Any]:
    """Load a fine-tuned LLaMA LoRA adapter. Returns (model, tokenizer)."""
    if PeftModel is None:
        raise ImportError("peft is required to load LoRA adapters. Install with `pip install peft`.")

    adapter_path = Path(adapter_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return model, tokenizer
