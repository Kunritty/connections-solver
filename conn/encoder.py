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

    def embed_board(self, words: list[str], prompt_prefix: str = "") -> torch.Tensor:
        """
        Embeds all words simultaneously to utilize DeBERTa's cross-attention.
        Extracts the contextualized embeddings for each word.
        
        Args:
            words: List of 16 words to embed.
            prompt_prefix: Optional text to prepend (e.g. for few-shot examples).
            
        Returns:
            A tensor of shape [len(words), hidden_dim] containing L2-normalized embeddings.
        """
        # Join words with a neutral separator (comma works well for lists)
        board_text = ", ".join(words)
        full_text = prompt_prefix + board_text
        
        # Tokenize with offset mapping to locate the exact tokens for each word
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024, # Need a larger max length to fit all words + prompt
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        
        # Extract the character offsets for each token
        offsets = inputs.pop("offset_mapping")[0].cpu().tolist()
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.model(**inputs)
            
        last_hidden = out.last_hidden_state[0]
        word_vecs = []
        
        # We only search for words in the board portion, not in the prompt prefix
        search_start = len(prompt_prefix)
        
        for w in words:
            # Find the exact character span for this word
            w_start = full_text.find(w, search_start)
            if w_start == -1:
                # Fallback to case-insensitive search
                w_start = full_text.lower().find(w.lower(), search_start)
                
            if w_start != -1:
                w_end = w_start + len(w)
                search_start = w_end # update search start for next word
                
                # Find all tokens that overlap with this character span
                token_indices = []
                for idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start < w_end and tok_end > w_start:
                        token_indices.append(idx)
                        
                if token_indices:
                    # Mean pool the tokens that make up this word
                    word_emb = last_hidden[token_indices].mean(dim=0)
                else:
                    word_emb = torch.zeros(last_hidden.size(-1), device=last_hidden.device)
            else:
                # Word somehow not found (shouldn't happen)
                word_emb = torch.zeros(last_hidden.size(-1), device=last_hidden.device)
                
            word_vecs.append(word_emb.cpu())
            
        # Stack into [num_words, hidden_dim]
        vecs = torch.stack(word_vecs, dim=0)
        # L2-normalize
        vecs = vecs / (vecs.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return vecs

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
