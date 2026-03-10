import time
import random
import csv
from itertools import permutations

TARGET_FILE_1 = "./data/train_split_data.csv"
TARGET_FILE_2 = "./data/test_split_data.csv"

# _norm and accuracy_min_swaps from conn/metrics.py

def _norm(g: list) -> frozenset:
    return frozenset(w.strip() for w in g)

def accuracy_min_swaps(pred_groups: list[list[str]], gold_groups: list[list[str]]) -> float:
    if len(pred_groups) != 4 or len(gold_groups) != 4:
        return float("inf")
    pred_sets = [_norm(g) for g in pred_groups]
    gold_sets = [_norm(g) for g in gold_groups]
    best_misplaced = 16
    for perm in permutations(range(4)):
        misplaced = 0
        for i in range(4):
            j = perm[i]
            misplaced += 4 - len(pred_sets[i] & gold_sets[j])
        best_misplaced = min(best_misplaced, misplaced)
    return (best_misplaced + 1) // 2

def select_words(target_file: str, diff: str) -> list[list[str]]:
  '''
  Input: path to a csv file containing puzzle data, difficulty from 0-3 or "random"
  Returns a puzzle solution in the form list[list, list, list, list]
  '''
  possible_categories = []

  with open(target_file, mode='r', newline='', encoding='utf-8') as file:
      fieldnames = ['groupName', 'difficulty', 'words']
      reader = csv.DictReader(file, fieldnames=fieldnames)

      for row in reader:
          if diff == "random" or row['difficulty'].strip() == str(diff):
              # convert to a set
              word_list = [w.strip() for w in row['words'].split(',')]
              possible_categories.append(word_list)

  # check least 4 categories to make a valid puzzle
  if len(possible_categories) < 4:
      raise ValueError(f"Not enough categories found for difficulty: {diff}")

  # Return 4 random sets from filtered list
  return random.sample(possible_categories, 4)

def get_predictions(words: list[list[str]], time_list: list) -> list[list[str]]:
    '''Input: a list of a list of 4 words which the user is prompted to partition into connections.
    The list is randomized, then printed out to the user in the form:
    [word][word][word][word]
    [word][word][word][word]
    [word][word][word][word]
    [word][word][word][word]
    Enter the first connection group in the form "word_1, word_2, word_3, word_4": {user input}
    Enter the second connection group in the form "word_1, word_2, word_3, word_4": {user input}
    Enter the third connection group in the form "word_1, word_2, word_3, word_4": {user input}
    Enter the fourth connection group in the form "word_1, word_2, word_3, word_4": {user input}
    Output: User partitioned words and the time_list should be set to [start_time, finish_time].'''
    
    # randomize list of input words
    flat_words = [word for group in words for word in group]
    random.shuffle(flat_words)
    
    # print to user
    print("\n--- Niche Connections Puzzle ---")
    for i in range(0, 16, 4):
        # pad the words
        row = flat_words[i:i+4]
        print(f"[{row[0]:<12}] [{row[1]:<12}] [{row[2]:<12}] [{row[3]:<12}]")
    print("-" * 30)
    
    # record start time
    start = time.time()
    time_list.append(start)
    
    # prompt user for groups
    user_results = []
    prompts = ["first", "second", "third", "fourth"]
    
    for p_name in prompts:
        user_input = input(f"Enter the {p_name} connection group 'word1, word2, word3, word4': ")
        # convert the comma-separated string into a list of normal strings
        group = [w.strip().upper() for w in user_input.split(',')]
        user_results.append(group)
    
    # record end time
    end = time.time()
    time_list.append(end)
    
    # return user groups
    return user_results

def main():
    print("This program is used to test a human in the same way our model is tested.")
    
    words = select_words(TARGET_FILE_2, "random")
        
    t_list = []
    
    pred_words = get_predictions(words, t_list)
    
    swaps: float = accuracy_min_swaps(pred_words, words)
    
    total_time = t_list[1]-t_list[0]
    
    print(f"Puzzle finished in {total_time:.3f} seconds with an accuracy of {swaps:.3f} min swaps.")
    print("Answers were: " + str(words))

if __name__ == "__main__":
  main()