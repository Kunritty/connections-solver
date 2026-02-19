import torch
from torch.nn.functional import normalize


def _mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask=None,
) -> torch.Tensor:
    mask = attention_mask.bool()
    if special_tokens_mask is not None:
        mask = mask & (~special_tokens_mask.bool().to(mask.device))
    if mask.sum() == 0:
        mask = attention_mask.bool()
    x = last_hidden_state[0][mask[0]]
    return x.mean(dim=0)


class DeBERTaEncoder:
    def __init__(self, model, tokenizer, device, max_length: int = 32, cache_size: int = 10000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self._embed_cache: dict[str, torch.Tensor] = {}
        self._cache_max = cache_size

    def embed_phrase(self, phrase: str) -> torch.Tensor:
        phrase = phrase.strip().lower()
        if phrase in self._embed_cache:
            return self._embed_cache[phrase]
        inputs = self.tokenizer(
            phrase,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = inputs.pop("special_tokens_mask", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.to(self.device)
        vec = _mean_pool(
            out.last_hidden_state, inputs["attention_mask"], special_tokens_mask
        ).float().cpu()
        vec = vec / (vec.norm(p=2) + 1e-12)
        if len(self._embed_cache) < self._cache_max:
            self._embed_cache[phrase] = vec
        return vec

    def group_similarity(self, embeddings: torch.Tensor) -> float:
        """Average pairwise cosine similarity (embeddings: [n, hidden])."""
        X = normalize(embeddings, dim=1)
        sims = X @ X.T
        n = sims.size(0)
        if n < 2:
            return 1.0
        mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
        sims = sims[mask]
        return float(sims.mean())
