import re

SYSTEM_PROMPT_STRING = """
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



def call_llm(system_prompt, user_prompt, client,
                        model = "llama3.1-8b",
                        temperature = 0.1,
                        max_tokens = 600
                    ):
    """
    Sends a chat request to the Cerebras API and returns the response content.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        msg = response.choices[0].message
        return (msg.content or "").strip()
    
    except Exception as e:
        return f"ERROR: {e}"

def convert_puzzle_to_prompt(words16):
    word_list_str = " || ".join(words16)
    return f"Here are the 16 words: {word_list_str}"

def parse_response_to_pred_groups(response):
    pattern = r"GROUP \d+: (.+)"
    
    groups = re.findall(pattern, response)
    parsed_groups = [ [word.strip() for word in group.split("||")] for group in groups ]
    
    return parsed_groups

import random

#Number of times LLM can retry if it hallucinates before marking it as an error
MAX_RETRIES = 4

def valid_prediction(pred_groups, words16):
    if pred_groups is None:
        return False
    if len(pred_groups) != 4:
        return False
    if any(len(group) != 4 for group in pred_groups):
        return False
    
    pred_words = [word for group in pred_groups for word in group]
    if set(pred_words) == set(words16):
        return True
    else:
        return False

def make_few_shot_prompt(words16, k, split_for_few_shot):
    few_shot_prompt = "Here are some previous examples:\n"
    count = 0
    for fs_row in split_for_few_shot:
        fs_words = fs_row["words"]
        # Skip because of leakage
        if set(fs_words) == set(words16):
            continue
        fs_groups = [ans["words"] for ans in fs_row["answers"]]
        fs_text = f"Here are 16 words: {' || '.join(fs_words)}\n"
        for i, group in enumerate(fs_groups, 1):
            fs_text += f"GROUP {i}: {' || '.join(group)}\n"
        few_shot_prompt += fs_text + "\n"
        count += 1
        if count >= k:
            break
    return few_shot_prompt

def solve_puzzle(words16, k=0, split_for_few_shot=None, model="llama3.1-8b", temperature=0.1, max_tokens=600, max_retries=MAX_RETRIES, client=None):
    few_shot_prompt = ""
    if k > 0 and split_for_few_shot is not None:
        few_shot_examples = random.sample(list(split_for_few_shot), k)
        few_shot_prompt = make_few_shot_prompt(words16, k, few_shot_examples)
    user_prompt = few_shot_prompt + "NOW, solve this puzzle and only this one: " + convert_puzzle_to_prompt(words16)

    attempt = 0
    while attempt <= max_retries:
        response = call_llm(
            SYSTEM_PROMPT_STRING,
            user_prompt,
            client,
            model=model,
            temperature=(temperature + 0.1*attempt),  
            max_tokens=max_tokens
        )

        pred_groups = parse_response_to_pred_groups(response)

        if valid_prediction(pred_groups, words16):
            return pred_groups

        print(f"Invalid LLM output: {response}\nCorrect Answer: {words16}\nRetrying ({attempt+1}/{max_retries})")
        attempt += 1

    print("ERROR: LLM failed after retries")
    return []