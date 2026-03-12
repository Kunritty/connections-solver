from conn.solvers.base import BaseSolver



class LlamaSolver(BaseSolver):
    def __init__(self, encoder):
        super().__init__(encoder)

    def solve(self, words16: list[str]) -> list[list[str]]:
        return []

# https://huggingface.co/meta-llama/Llama-3.1-8B?library=transformers