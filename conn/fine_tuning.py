from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from data_loader import ExampleGroup
from conn.encoder import DeBERTaEncoder

try:
    from peft import LoraConfig, TaskType, PeftModel, get_peft_model
except ImportError:  # pragma: no cover - handled by runtime check
    LoraConfig = None
    TaskType = None
    PeftModel = None
    get_peft_model = None


GroupedExamplesLike = list[ExampleGroup] | list[list[str]]


@dataclass
class FineTuneStats:
    epochs: int
    steps: int
    average_losses: list[float]


class ConnectionsPairDataset(Dataset):
    def __init__(
        self,
        puzzles: list[Any],
        negatives_per_positive: int = 2,
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        self._pairs = self._build_pairs(puzzles, negatives_per_positive)
        if not self._pairs:
            raise ValueError("No training pairs were built from the provided examples.")

    @staticmethod
    def _group_words(group: Any) -> list[str]:
        if isinstance(group, ExampleGroup):
            words = group.words
        elif isinstance(group, dict):
            words = group.get("words", [])
        else:
            words = group
        return [str(w).strip() for w in words if str(w).strip()]

    def _extract_groups(self, puzzle: Any) -> list[list[str]]:
        if isinstance(puzzle, dict):
            if "answers" in puzzle:
                groups = [a.get("words", []) for a in puzzle.get("answers", [])]
            elif "groups" in puzzle:
                groups = puzzle.get("groups", [])
            else:
                groups = []
        else:
            groups = puzzle
        out: list[list[str]] = []
        for g in groups:
            words = self._group_words(g)
            if len(words) == 4:
                out.append(words)
        return out

    def _build_pairs(
        self,
        puzzles: list[Any],
        negatives_per_positive: int,
    ) -> list[tuple[str, str, float]]:
        pairs: list[tuple[str, str, float]] = []
        for puzzle in puzzles:
            groups = self._extract_groups(puzzle)
            if len(groups) < 2:
                continue

            positive_pairs: list[tuple[str, str, float]] = []
            for group in groups:
                for i, j in combinations(range(len(group)), 2):
                    positive_pairs.append((group[i], group[j], 1.0))
            pairs.extend(positive_pairs)

            negative_candidates: list[tuple[str, str, float]] = []
            for g1_idx, g2_idx in combinations(range(len(groups)), 2):
                for w1 in groups[g1_idx]:
                    for w2 in groups[g2_idx]:
                        negative_candidates.append((w1, w2, -1.0))

            if not negative_candidates:
                continue

            max_neg = max(1, len(positive_pairs) * max(1, negatives_per_positive))
            if len(negative_candidates) > max_neg:
                negative_candidates = self._rng.sample(negative_candidates, k=max_neg)
            pairs.extend(negative_candidates)
        self._rng.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self._pairs[idx]


def _mean_pool_batch(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1e-6)
    return summed / counts


def _build_lora_model(
    model_name: str,
    device: torch.device,
    r: int,
    alpha: int,
    dropout: float,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA fine-tuning. Install it with `pip install peft`.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query_proj", "value_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, config)
    model.to(device)
    return model, tokenizer


def finetune_deberta_lora(
    puzzles: list[Any],
    model_name: str = "microsoft/deberta-v3-small",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    negatives_per_positive: int = 2,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    margin: float = 0.2,
    max_length: int = 16,
    seed: int = 42,
    adapter_output_dir: str | Path | None = None,
    verbose: bool = False,
) -> tuple[DeBERTaEncoder, FineTuneStats]:
    if not puzzles:
        raise ValueError("puzzles must not be empty.")

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if verbose:
        print(f"Starting LoRA fine-tuning on {device} for {epochs} epochs...")
        
    model, tokenizer = _build_lora_model(
        model_name=model_name,
        device=device,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    dataset = ConnectionsPairDataset(
        puzzles=puzzles,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=margin)

    model.train()
    step_count = 0
    average_losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        for a_words, b_words, labels in loader:
            enc_a = tokenizer(
                list(a_words),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc_b = tokenizer(
                list(b_words),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc_a = {k: v.to(device) for k, v in enc_a.items()}
            enc_b = {k: v.to(device) for k, v in enc_b.items()}
            labels_t = labels.to(device=device, dtype=torch.float32)

            out_a = model(**enc_a)
            out_b = model(**enc_b)
            emb_a = _mean_pool_batch(out_a.last_hidden_state, enc_a["attention_mask"])
            emb_b = _mean_pool_batch(out_b.last_hidden_state, enc_b["attention_mask"])
            emb_a = F.normalize(emb_a, p=2, dim=1)
            emb_b = F.normalize(emb_b, p=2, dim=1)

            loss = loss_fn(emb_a, emb_b, labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
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

    model.eval()
    encoder = DeBERTaEncoder(model=model, tokenizer=tokenizer, device=device)
    stats = FineTuneStats(epochs=epochs, steps=step_count, average_losses=average_losses)
    return encoder, stats


def load_lora_encoder(
    adapter_dir: str | Path,
    base_model_name: str = "microsoft/deberta-v3-small",
) -> DeBERTaEncoder:
    if PeftModel is None:
        raise ImportError("peft is required to load LoRA adapters. Install it with `pip install peft`.")

    adapter_path = Path(adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return DeBERTaEncoder(model=model, tokenizer=tokenizer, device=device)


def solve_fine_tuned(
    words16: list[str],
    encoder: DeBERTaEncoder,
    example_groups: GroupedExamplesLike | None = None,
) -> list[list[str]]:
    from conn.solvers import FewShotSolver
    return FewShotSolver(encoder, example_groups=example_groups).solve(words16)
