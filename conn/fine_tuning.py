from __future__ import annotations

import random
from dataclasses import dataclass
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
except ImportError:  # pragma: no cover
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


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (SupCon)
# ---------------------------------------------------------------------------

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Loss over L2-normalized embeddings.

    Pulls embeddings that share the same label together and pushes
    apart embeddings with different labels.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] L2-normalized embeddings.
            labels:   [N] integer group labels (globally unique across boards in the batch).
        """
        device = features.device
        n = features.size(0)
        if labels.dim() != 1 or labels.size(0) != n:
            raise RuntimeError(
                f"SupConLoss: features.size(0)={n} must match labels.size(0)={labels.size(0)} (labels.dim()={labels.dim()}). "
                "Ensure flat_vecs and flat_labels have the same length in the training loop."
            )

        sim_matrix = features @ features.T / self.temperature

        # Mask: same label = positive pair (excluding self)
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = ~torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = label_eq & self_mask

        # For numerical stability, subtract row-wise max before exp
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * self_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        # Mean of log-prob over positive pairs per anchor
        pos_counts = pos_mask.float().sum(dim=1).clamp_min(1.0)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_counts

        loss = -mean_log_prob_pos.mean()
        return loss


# ---------------------------------------------------------------------------
# Board Dataset — yields (words16, group_labels) for full-board training
# ---------------------------------------------------------------------------

class ConnectionsBoardDataset(Dataset):
    """Yields shuffled 16-word boards with integer group labels (0-3)."""

    def __init__(self, puzzles: list[Any], seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._boards = self._build_boards(puzzles)
        if not self._boards:
            raise ValueError("No valid boards built from the provided puzzles.")

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

    def _build_boards(self, puzzles: list[Any]) -> list[tuple[list[str], list[int]]]:
        boards: list[tuple[list[str], list[int]]] = []
        for puzzle in puzzles:
            groups = self._extract_groups(puzzle)
            if len(groups) != 4:
                continue
            words: list[str] = []
            labels: list[int] = []
            for group_idx, group in enumerate(groups):
                words.extend(group)
                labels.extend([group_idx] * len(group))
            if len(words) != 16:
                continue
            boards.append((words, labels))
        return boards

    def __len__(self) -> int:
        return len(self._boards)

    def __getitem__(self, idx: int) -> tuple[list[str], list[int]]:
        words, labels = self._boards[idx]
        combined = list(zip(words, labels))
        self._rng.shuffle(combined)
        shuffled_words, shuffled_labels = zip(*combined)
        return list(shuffled_words), list(shuffled_labels)


def _collate_boards(batch: list[tuple[list[str], list[int]]]) -> tuple[list[list[str]], list[list[int]]]:
    """Collate so batch_words[i] = i-th board's 16 words; avoid default_collate transposing."""
    words_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    return words_list, labels_list


# ---------------------------------------------------------------------------
# Batched word-embedding extraction (mirrors DeBERTaEncoder.embed_board)
# ---------------------------------------------------------------------------

def _extract_word_embeddings(
    model,
    tokenizer,
    batch_words: list[list[str]],
    device: torch.device,
    max_length: int = 512,
    prompt_prefix: str = "",
) -> torch.Tensor:
    """Extract contextualized, L2-normalized word embeddings for a batch of boards.

    Returns: Tensor of shape [batch_size, 16, hidden_dim].
    """
    full_texts = [prompt_prefix + ", ".join(words) for words in batch_words]

    inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model(**inputs)
    last_hidden = out.last_hidden_state
    hidden_dim = last_hidden.size(-1)

    batch_vecs: list[torch.Tensor] = []
    prefix_len = len(prompt_prefix)

    for b_idx, words in enumerate(batch_words):
        word_vecs: list[torch.Tensor] = []
        search_start = prefix_len
        b_offsets = offsets[b_idx].tolist()

        for w in words:
            w_start = full_texts[b_idx].find(w, search_start)
            if w_start == -1:
                w_start = full_texts[b_idx].lower().find(w.lower(), search_start)

            if w_start != -1:
                w_end = w_start + len(w)
                search_start = w_end
                token_indices = [
                    i for i, (ts, te) in enumerate(b_offsets)
                    if ts < w_end and te > w_start
                ]
                if token_indices:
                    word_emb = last_hidden[b_idx, token_indices].mean(dim=0)
                else:
                    word_emb = torch.zeros(hidden_dim, device=device)
            else:
                word_emb = torch.zeros(hidden_dim, device=device)

            word_vecs.append(word_emb)

        batch_vecs.append(torch.stack(word_vecs, dim=0))

    vecs = torch.stack(batch_vecs, dim=0)
    vecs = F.normalize(vecs, p=2, dim=-1)
    return vecs


# ---------------------------------------------------------------------------
# LoRA model builder
# ---------------------------------------------------------------------------

def _build_lora_model(
    model_name: str,
    device: torch.device,
    r: int,
    alpha: int,
    dropout: float,
):
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise ImportError("peft is required for LoRA fine-tuning. Install with `pip install peft`.")

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


# ---------------------------------------------------------------------------
# Main fine-tuning entry point (contextual board-level SupCon)
# ---------------------------------------------------------------------------

def finetune_deberta_lora(
    puzzles: list[Any],
    model_name: str = "microsoft/deberta-v3-small",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    temperature: float = 0.07,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    max_length: int = 512,
    seed: int = 42,
    adapter_output_dir: str | Path | None = None,
    verbose: bool = False,
) -> tuple[DeBERTaEncoder, FineTuneStats]:
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
        print(f"Starting contextual LoRA fine-tuning on {device} for {epochs} epochs...")

    model, tokenizer = _build_lora_model(
        model_name=model_name,
        device=device,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )

    dataset = ConnectionsBoardDataset(puzzles=puzzles, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_boards,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )
    loss_fn = SupConLoss(temperature=temperature)

    model.train()
    step_count = 0
    average_losses: list[float] = []

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        for batch_words, batch_labels in loader:
            # batch_words: list of 16-word lists; batch_labels: list of 16-int lists or tensor (bs, 16)
            bs = len(batch_words)
            vecs = _extract_word_embeddings(
                model, tokenizer, batch_words, device, max_length=max_length,
            )
            if vecs.size(0) != bs or vecs.size(1) != 16:
                raise RuntimeError(
                    f"Expected embeddings shape ({bs}, 16, -) for {bs} boards, got {vecs.shape}. "
                    "Tokenizer/model batch size must match len(batch_words)."
                )
            flat_vecs = vecs.view(bs * 16, -1)

            # Offset labels so each board's groups are globally unique
            global_labels: list[torch.Tensor] = []
            for b_idx in range(bs):
                bl = batch_labels[b_idx]
                board_labels = bl if isinstance(bl, torch.Tensor) else torch.tensor(bl, dtype=torch.long, device=device)
                if board_labels.dim() == 0:
                    board_labels = board_labels.unsqueeze(0)
                global_labels.append(board_labels.to(device).long() + b_idx * 4)
            flat_labels = torch.cat(global_labels, dim=0)
            if flat_labels.size(0) != flat_vecs.size(0):
                raise RuntimeError(
                    f"flat_vecs.size(0)={flat_vecs.size(0)} != flat_labels.size(0)={flat_labels.size(0)}. "
                    f"bs={bs}, len(batch_labels)={len(batch_labels) if hasattr(batch_labels, '__len__') else 'N/A'}."
                )

            loss = loss_fn(flat_vecs, flat_labels)
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


# ---------------------------------------------------------------------------
# Load a previously saved LoRA adapter
# ---------------------------------------------------------------------------

def load_lora_encoder(
    adapter_dir: str | Path,
    base_model_name: str = "microsoft/deberta-v3-small",
) -> DeBERTaEncoder:
    if PeftModel is None:
        raise ImportError("peft is required to load LoRA adapters. Install with `pip install peft`.")

    adapter_path = Path(adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    base_model = AutoModel.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()
    return DeBERTaEncoder(model=model, tokenizer=tokenizer, device=device)


# ---------------------------------------------------------------------------
# Convenience: solve a board with a fine-tuned encoder + FewShotSolver
# ---------------------------------------------------------------------------

def solve_fine_tuned(
    words16: list[str],
    encoder: DeBERTaEncoder,
    example_groups: GroupedExamplesLike | None = None,
) -> list[list[str]]:
    from conn.solvers import FewShotSolver
    return FewShotSolver(encoder, example_groups=example_groups).solve(words16)
