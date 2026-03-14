from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
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

try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes  # noqa: F401 – confirms the library is installed
    _BNB_AVAILABLE = True
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore[assignment,misc]
    _BNB_AVAILABLE = False

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"


@dataclass
class LlamaFineTuneStats:
    epochs: int
    steps: int
    average_losses: list[float]
    batch_losses: list[float]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LlamaFineTuneStats:
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# SFT Dataset — prompt/target text pairs for causal-LM fine-tuning
# ---------------------------------------------------------------------------

def _format_target(groups: list[list[str]]) -> str:
    lines: list[str] = []
    for i, group in enumerate(groups, 1):
        lines.append(f"GROUP {i}: {', '.join(group)}")
    return "\n".join(lines)


def _format_prompt(words16: list[str], few_shot_text: str = "") -> str:
    user_body = (
        f"NOW, solve this puzzle and only this one: "
        f"Here are the 16 words: {', '.join(words16)}"
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
    """Yields (prompt_text, target_text, words16, allowed_ids) for causal-LM SFT."""

    def __init__(self, puzzles: list[Any], tokenizer: Any, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._tokenizer = tokenizer
        self._examples = self._build(puzzles)
        if not self._examples:
            raise ValueError("No valid SFT examples built from the provided puzzles.")

    def _build(self, puzzles: list[Any]) -> list[tuple[str, str, list[str], list[int]]]:
        format_ids = _get_format_ids(self._tokenizer)
        examples: list[tuple[str, str, list[str], list[int]]] = []
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
            allowed_ids = _get_allowed_ids(words16, self._tokenizer, format_ids)
            examples.append((prompt, target, words16, allowed_ids))
        return examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[str, str, list[str], list[int]]:
        return self._examples[idx]


def _collate_sft(
    batch: list[tuple[str, str, list[str], list[int]]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Tokenize prompt+target, masking prompt tokens in the labels."""
    input_ids_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    words16_batch: list[list[str]] = []
    allowed_ids_batch: list[list[int]] = []

    for prompt, target, words16, allowed_ids in batch:
        full_text = prompt + "\n" + target + tokenizer.eos_token
        prompt_ids = tokenizer(prompt + "\n", add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(labels) < len(full_ids):
            labels = labels + [-100] * (len(full_ids) - len(labels))
        labels = labels[: len(full_ids)]

        input_ids_list.append(torch.tensor(full_ids, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))
        words16_batch.append(words16)
        allowed_ids_batch.append(allowed_ids)

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "_words16": words16_batch,
        "_allowed_ids": allowed_ids_batch,
    }


# ---------------------------------------------------------------------------
# Logit masking: CE only over allowed tokens (strict, efficient)
# ---------------------------------------------------------------------------

# NOTE: Performance worsened with hallucination penalties
_FORMAT_STRINGS = [
    "GROUP", " GROUP", "1", "2", "3", "4", " 1", " 2", " 3", " 4",
    ":", " :", ": ", ",", ", ", "\n", " ",
]


def _get_format_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for s in _FORMAT_STRINGS:
        ids.update(tokenizer(s, add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ids.add(tokenizer.pad_token_id)
    return ids


def _get_allowed_ids(words16: list[str], tokenizer: Any, format_ids: set[int] | None = None) -> list[int]:
    if format_ids is None:
        format_ids = _get_format_ids(tokenizer)
    allowed = set(format_ids)
    for word in words16:
        allowed.update(tokenizer(word, add_special_tokens=False)["input_ids"])
        allowed.update(tokenizer(" " + word, add_special_tokens=False)["input_ids"])
    return [tid for tid in allowed if tid >= 0]


def _causal_lm_loss_allowed_only(
    logits: torch.Tensor,
    labels: torch.Tensor,
    allowed_ids_per_batch: list[list[int]],
    device: torch.device,
    ignore_index: int = -100,
) -> torch.Tensor:
    """CE over allowed tokens only (disallowed get 0 prob). logits [B,T,V], labels [B,T]."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    max_A = max(len(a) for a in allowed_ids_per_batch)
    if max_A == 0:
        return torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=ignore_index
        )

    index_list: list[list[int]] = []
    valid_list: list[list[bool]] = []
    for allowed_ids in allowed_ids_per_batch:
        pad_len = max_A - len(allowed_ids)
        index_list.append(list(allowed_ids) + [0] * pad_len)
        valid_list.append([True] * len(allowed_ids) + [False] * pad_len)

    index_t = torch.tensor(index_list, dtype=torch.long, device=device)
    valid_t = torch.tensor(valid_list, dtype=torch.bool, device=device)
    id_to_idx_per_b: list[dict[int, int]] = [
        {tid: i for i, tid in enumerate(allowed_ids_per_batch[b])} for b in range(B)
    ]

    index_exp = index_t.unsqueeze(1).expand(B, T, max_A)
    gathered = torch.gather(shift_logits, 2, index_exp)
    gathered = gathered.masked_fill(~valid_t.unsqueeze(1).expand(B, T, max_A), float("-inf"))

    target_index = torch.full((B, T), ignore_index, dtype=torch.long, device=device)
    for b in range(B):
        id_to_idx = id_to_idx_per_b[b]
        for t in range(T):
            lbl = shift_labels[b, t].item()
            if lbl == ignore_index:
                continue
            if lbl in id_to_idx:
                target_index[b, t] = id_to_idx[lbl]

    return torch.nn.functional.cross_entropy(
        gathered.reshape(-1, max_A),
        target_index.reshape(-1),
        ignore_index=ignore_index,
    )


# ---------------------------------------------------------------------------
# LoRA model builder for causal LM
# ---------------------------------------------------------------------------

def _build_llama_lora_model(
    model_name: str,
    device: torch.device,
    r: int,
    alpha: int,
    dropout: float,
    use_4bit: bool = True,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA fine-tuning. Install with `pip install peft`.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit and device.type == "cuda":
        if not _BNB_AVAILABLE or BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes is required for 4-bit QLoRA. Install with `pip install bitsandbytes`."
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        model.to(device)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main fine-tuning entry point
# ---------------------------------------------------------------------------

def finetune_llama_lora(
    puzzles: list[Any],
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    max_length: int = 256,
    seed: int = 42,
    use_4bit: bool = True,
    adapter_output_dir: str | Path | None = None,
    use_logit_masking: bool = True,
    verbose: bool = False,
) -> tuple[Any, Any, LlamaFineTuneStats]:
    """Fine-tune LLaMA with LoRA (or QLoRA when use_4bit=True) on Connections puzzles.

    When use_logit_masking is True, disallowed tokens (not in the 16 puzzle words
    or format tokens) get logit -inf so the model cannot assign them probability.

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
    quant_label = "4-bit QLoRA" if (use_4bit and device.type == "cuda") else "LoRA fp16"
    if verbose:
        print(f"Starting LLaMA {quant_label} SFT on {device} for {epochs} epochs...")

    model, tokenizer = _build_llama_lora_model(
        model_name=model_name,
        device=device,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        use_4bit=use_4bit,
    )

    dataset = ConnectionsSFTDataset(puzzles=puzzles, tokenizer=tokenizer, seed=seed)
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

    # With device_map="auto" (4-bit), tensors must go to the model's first device
    batch_device = next(model.parameters()).device

    try:
        from tqdm.auto import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    model.train()
    step_count = 0
    average_losses: list[float] = []
    batch_losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        # Wrap loader in tqdm if verbose and tqdm is available
        loop_iterable = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}") if verbose and has_tqdm else loader

        for batch in loop_iterable:
            allowed_ids_batch = batch.pop("_allowed_ids")
            batch.pop("_words16")
            batch = {k: v.to(batch_device) for k, v in batch.items()}
            outputs = model(**batch)

            if use_logit_masking:
                loss = _causal_lm_loss_allowed_only(
                    outputs.logits,
                    batch["labels"],
                    allowed_ids_batch,
                    batch_device,
                )
            else:
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)
            batch_count += 1
            step_count += 1
            
            # Update progress bar with the current loss
            if verbose and has_tqdm:
                loop_iterable.set_postfix(loss=f"{batch_loss:.4f}")

        epoch_avg_loss = running_loss / max(1, batch_count)
        average_losses.append(epoch_avg_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {epoch_avg_loss:.4f}")

    model.eval()
    stats = LlamaFineTuneStats(epochs=epochs, steps=step_count, average_losses=average_losses, batch_losses=batch_losses)

    if adapter_output_dir is not None:
        out_path = Path(adapter_output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        stats_path = out_path / "training_stats.json"
        stats.save(stats_path)
        if verbose:
            print(f"Adapter saved to {out_path}")
            print(f"Training stats saved to {stats_path}")

    return model, tokenizer, stats


# ---------------------------------------------------------------------------
# Load a previously saved LoRA adapter
# ---------------------------------------------------------------------------

def load_llama_lora(
    adapter_dir: str | Path,
    base_model_name: str = DEFAULT_MODEL_NAME,
    use_4bit: bool = True,
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

    if use_4bit and device.type == "cuda":
        if not _BNB_AVAILABLE or BitsAndBytesConfig is None:
            raise ImportError(
                "bitsandbytes is required for 4-bit loading. Install with `pip install bitsandbytes`."
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer
