from conn.solvers.base import BaseSolver
from huggingface_hub import login
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from data_loader import ExampleGroup

# Using same Llama 3.1 model as other notebooks
# https://huggingface.co/meta-llama/Llama-3.1-8B?library=transformers

MODEL_NAME = "meta-llama/Llama-3.1-8B"

SYSTEM_PROMPT = """
    Your task is to partition the 16 words into 4 groups of 4 words/phrases based on shared connections. 
    Output requirements (STRICT): 
    OUTPUT ONLY the final groups of words/phrases.
    Do NOT provide reasoning or explanations under any circumstances.
    DO NOT output any text other than the 4 groups.
    Use ONLY the EXACT words/phrases from the puzzle.
    Make sure there are EXACTLY 4 groups of 4 words/phrases each with their category names. NO EXCEPTIONS.
    Return the answer exactly in this format:

    GROUP 1: word1 || word2 || word3 || word4
    GROUP 2: word1 || word2 || word3 || word4
    GROUP 3: word1 || word2 || word3 || word4
    GROUP 4: word1 || word2 || word3 || word4
"""

# TODO: Add login to the notebook
login(token=os.environ.get("HF_API_KEY"))


class LlamaSolver(BaseSolver):
    def __init__(self, encoder, example_groups: list[ExampleGroup] = []):
        super().__init__(encoder)
        self.system_prompt = SYSTEM_PROMPT
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.example_groups = example_groups

    def solve(self, words16: list[str]) -> list[list[str]]:
        user_prompt = self._build_user_prompt(words16)
        prompt = self.system_prompt + "\n\n" + user_prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_groups = self._parse_response(response)
        return []

    def _build_user_prompt(self, words16: list[str]) -> str:
        return f"Here are the 16 words: {words16}"
