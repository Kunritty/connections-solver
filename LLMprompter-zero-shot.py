from openai import OpenAI
from google import genai
import csv
import random

EXAMPLE_WORDS = ['COWARD', 'LASER', 'SCOTTIE', 'BONES', 'JIM', 'CORAL', 'BALANCE', 'CHEKHOV', 'THISTLE', 'SHAW', 'SHELLS', 'TRACTOR', 'TARTAN', 'BAGPIPES', 'MILLER', 'TEETH']

TARGET_FILE = "./data/puzzle_data1.csv"
TARGET_FILE_2 = "./data/puzzle_data2.csv"
TARGET_FILE_3 = "./data/puzzle_data3.csv"

def get_params_multiple() -> list:
  '''Returns a list in the form [C or G for ChatGPT or Gemini, path to words csv, api key, difficulty from 0-3 or "rand", number of puzzles (as an int)]'''
  return_list = []
  while(True):
    print("Using (C)hatGPT OR (G)emini API OR (Q)uit: ")
    temp = input()
    if(temp.lower() == "c" or temp.lower() == "g"):
      return_list.append(temp)
      break
    elif(temp.lower() == "q"):
      return []
    else:
      print("Enter 'C', 'G', or 'Q'")
      
  while(True):
    print("Enter the path to a csv file containing words OR 'default' for hardcoded path OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    elif(temp.lower() == "default"):
      return_list.append(TARGET_FILE)
      break
    else:
      return_list.append(temp)
      break
  
  while(True):
    print("Enter the API key for the model or 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    else:
      return_list.append(temp)
      break
    
  while(True):
    print("Enter the difficulty for the puzzle '0' OR '1' OR '2' OR '3' OR 'rand' for random difficulty OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    elif(temp.lower() in ["0", "1", "2", "3", "rand"]):
      return_list.append(temp)
      break
    else:
      print("Invalid input. Please try again.")
      
  while(True):
    print("Enter the number of puzzles to run OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    else:
      try:
        return_list.append(int(temp))
      except:
        print("Please enter an integer.")
      else:
        break
  
  return return_list

def get_params_single() -> list:
  '''Returns a list in the form [C or G for ChatGPT or Gemini, path to words csv, api key, difficulty from 0-3 or "rand"] or [] if exit program'''
  return_list = []
  while(True):
    print("Using (C)hatGPT OR (G)emini API OR (Q)uit: ")
    temp = input()
    if(temp.lower() == "c" or temp.lower() == "g"):
      return_list.append(temp)
      break
    elif(temp.lower() == "q"):
      return []
    else:
      print("Enter 'C', 'G', or 'Q'")
      
  while(True):
    print("Enter the path to a csv file containing words OR 'default' for hardcoded path OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    elif(temp.lower() == "default"):
      return_list.append(TARGET_FILE)
      break
    else:
      return_list.append(temp)
      break
  
  while(True):
    print("Enter the API key for the model OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    else:
      return_list.append(temp)
      break
    
  while(True):
    print("Enter the difficulty for the puzzle '0' OR '1' OR '2' OR '3' OR 'rand' for random difficulty OR 'Q' to quit: ")
    temp = input()
    if(temp.lower() == "q"):
      return []
    elif(temp.lower() in ["0", "1", "2", "3", "rand"]):
      return_list.append(temp)
      break
    else:
      print("Invalid input. Please try again.")
  
  return return_list

def buildprompt(words: list) -> str:
  '''builds the return string with the words randomized'''
  
  prompt = '''You are solving a NYT Connections puzzle. 
          You are given exactly these 16 words in this format: 
          {''' 
          
  random.shuffle(words)
  for word in words:
    if(word != words[-1]):
      prompt += word
      prompt += ", "
    else:
      prompt += word
  
          
  prompt += '''}
          Your task is to partition the 16 words into 4 groups of 4 words based on shared connections. 
          Output requirements (STRICT): 
          Output EXACTLY four groups. 
          Each group must contain EXACTLY four words. 
          Use ONLY the words provided. Do NOT add explanations. 
          Do NOT add category names. 
          Do NOT add commentary. 
          Do NOT include extra whitespace or text. 
          Output must match EXACTLY this format: [{word1, word2, word3, word4}, {word5, word6, word7, word8}, {word9, word10, word11, word12}, {word13, word14, word15, word16}] 
          Use curly braces {} around each group. 
          Use square brackets [] around the entire answer. 
          Separate groups with comma + space. 
          Separate words within a group with comma + space. 
          Do not reorder letters inside words. 
          Every input word must appear exactly once. 
          If multiple valid solutions exist, return any one valid solution. 
          Only output the final grouped answer in the exact required format.'''
  return prompt

def execute_gpt_prompt(prompt: str, given_api_key: str) -> str:
  '''executes a given prompt using OpenAI's api and returns the result'''
  try:
    client = OpenAI(
      api_key = given_api_key
    )
    
    response = client.responses.create(
      model="gpt-5-nano",
      input=prompt,
      store=True,
    )

    return response.output_text
  except Exception as e:
    print("An exception occurred with description: " + str(e))
    return "Error"
  
def execute_gemini_prompt(prompt: str, given_api_key: str) -> str:
  '''executes a given prompt using Google's api and returns the result'''
  try:
    client = genai.Client(api_key=given_api_key)
    
    response = client.models.generate_content(
      model="gemini-3.1-pro-preview",
      contents=prompt
    )

    return response.text
  
  except Exception as e:
    print("An exception occurred with description: " + str(e))
    return "Error"

def select_words(target_file: str, diff: str) -> list:
  '''
  Input: path to a csv file containing puzzle data, difficulty from 0-3 or "random"
  Returns a puzzle solution in the form list[set, set, set, set]
  '''
  possible_categories = []

  with open(target_file, mode='r', newline='', encoding='utf-8') as file:
      fieldnames = ['groupName', 'difficulty', 'words']
      reader = csv.DictReader(file, fieldnames=fieldnames)

      for row in reader:
          if diff == "random" or row['difficulty'].strip() == str(diff):
              # convert to a set
              word_list = [w.strip() for w in row['words'].split(',')]
              possible_categories.append(set(word_list))

  # check least 4 categories to make a valid puzzle
  if len(possible_categories) < 4:
      raise ValueError(f"Not enough categories found for difficulty: {diff}")

  # Return 4 random sets from filtered list
  return random.sample(possible_categories, 4)


def main():
  print("Welcome to the LLM Zero-shot prompter. Please enter (S)ingle or (M)ultiple for the amount of puzzles to solve: ")
  amt = input()
  num_puzzles = 1
  paramlist = []
  if(amt.lower() == "s"):
    paramlist = get_params_single()
  elif(amt.lower() == "m"):
    paramlist = get_params_multiple()
    if paramlist: num_puzzles = paramlist[4]
  else:
    print("Invalid input. Please use 'S' OR 'M'.")
    
  if not paramlist:
    print("Exiting.")
    return
  
  mode, file_path, api_key, diff = paramlist[0], paramlist[1], paramlist[2], paramlist[3]
  
  for i in range(num_puzzles):
    print(f"\n--- Running Puzzle {i+1} ---")
        
    # Get the 4 sets
    solution_sets = select_words(file_path, diff)
    
    # Flatten sets into a list of 16 words for the prompt
    all_words = []
    for s in solution_sets:
        all_words.extend(list(s))
    
    # Build the prompt
    prompt = buildprompt(all_words)
    
    # Execute based on 'C' or 'G'
    if mode.lower() == 'c':
        result = execute_gpt_prompt(prompt, api_key)
    else:
        result = execute_gemini_prompt(prompt, api_key)
        
    print("LLM Response:")
    print(result)
    print("Correct Solution Sets:")
    print(solution_sets)

  
  
if __name__ == "__main__":
  main()
