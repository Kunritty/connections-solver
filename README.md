# Connections Solver

### Authors: Kyle Yin Xu, Kunritty He, Micah Cheng 

Exploration of a BERT-based model and tuned LLM to solve real NYT Connections puzzle and artificially generated puzzles. Artificial puzzes will be generated with an LLM pipeline.



## Important References:

[Making New Connections: LLMs as Puzzle Generators for The New York Times' Connections Word Game](https://arxiv.org/abs/2407.11240)  

- Using LLMs as a method for generating synthetic connections puzzle data
- Our baseline for attempts to generate synthetic training datasets

[LLMs as Method Actors: A Model for Prompt Engineering and Architecture](https://arxiv.org/abs/2411.05778v2)

- Proof of concept for current LLMs' ability to comprehend and solve complex semantic connections puzzles
- Reference for experiments involving LLM evaluation and comparison to our fine-tuned models

## Repository Walkthrough

To grasp a general summary of our project findings, we have a walkthrough notebook `project.ipynb`

## External Libraries

The full list of external libraries can be found in `requirements.txt` and is also listed below:

```pip-requirements
pandas
torch
transformers
numpy
datasets
matplotlib
protobuf
sentencepiece
peft
bitsandbytes
cerebras.cloud.sdk
huggingface_hub
dotenv
hf_xet
```

All other code in the codebase is directly written and managed by our team members. No external code was directly used within the files apart from the imported libraries mentioned above.

## Setup

Install Conda ([link](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)) and create a new Conda environment for this project.

Ensure conda is installed and added to PATH:

```
$ conda --version
```

Create and activate new environment. Make sure you're in this environment whenever installing and running code:

```
$ conda create --name connections python=3.10
$ conda activate connections
(connections) $
(connections) $ conda deactivate
```

Once you're in the env install all the required libraries:

```
$ conda activate connections
(connections) $ pip install -r requirements.txt
```

Your environment should be set up, now ensure the Python kernel for the notebook files is set to this new Conda environment, and you should be able to run them with no issues.

### LLM Notebooks Setup

To make your own calls to the Llama LLM, go to [Cerebras](https://www.cerebras.ai/) and make a free account. Then with your free api key, save it to your conda environment using the following:

```
conda env config vars set CEREBRAS_API_KEY="your_api_key_here"
```

After restarting your environment, you should be all set to run the LLM notebooks.

### LLM .py file Setup

Make an account at either [https://platform.openai.com/](https://platform.openai.com/) or [https://aistudio.google.com/](https://aistudio.google.com/) and obtain an API key to use. Dependencies: install google-genai and openai using pip install.

### Instructions for use of the human_test.py

Run the file by doing:
python human_test.py
in a terminal or by running the py file. Run from root directory so files can be read.
Results will be logged to data/human_measurement.csv

## File structure

```
connections-solver/
├── conn/                           # Shared solver library (used by DeBERTa/LLM notebooks)
│   ├── __init__.py                 # Re-exports public API
│   ├── encoder.py                  # DeBERTa/encoder helpers
│   ├── search.py                   # Greedy / heuristic group search
│   ├── metrics.py                  # Evaluation metrics (accuracy, min swaps, etc.)
│   ├── fine_tuning.py              # DeBERTa LoRA fine-tuning helpers
│   ├── llama_fine_tuning.py        # LLaMA LoRA fine-tuning helpers
│   └── solvers/                    # Concrete solver implementations
│       ├── __init__.py
│       ├── base.py                 # Base solver interfaces
│       ├── contextual.py           # DeBERTa contextual solver
│       ├── isolated.py             # Isolated-phrase DeBERTa solver
│       ├── llama.py                # LLaMA-based solver
│       └── random.py               # Random grouping baseline
├── data_loader/                    # Data loading and splitting utilities
│   ├── __init__.py
│   ├── loader.py                   # Core CSV/JSON loading
│   ├── dataset_split.py            # Train/test split helpers
│   └── cross_validation.py         # Cross-validation utilities
├── data/                           # Datasets, splits, and human-study artifacts
│   ├── connections_words.csv       # Word list / puzzle vocabulary
│   ├── examples.txt                # Example categories (JSONL) for few-shot generation
│   ├── word_bank.txt               # Word bank for generation pipeline
│   ├── train_split_data.csv        # Train split (precomputed)
│   ├── test_split_data.csv         # Test split (precomputed)
│   ├── LLM_few_shot_results.csv    # Raw LLM few-shot results
│   ├── human_measurement.csv       # Human eval log
│   ├── human_performance_overview.png
│   ├── human_performance_timeline.png
│   └── human_rolling_performance.png
├── reports/                        # Saved metrics, outputs, and figures
│   ├── deberta_zero_shot/          # DeBERTa zero-shot test metrics/outputs
│   ├── deberta_few_shot/           # DeBERTa few-shot test metrics/outputs
│   ├── deberta_lora/               # DeBERTa LoRA test metrics/outputs
│   ├── figures/                    # PNG figures for paper/report
│   └── LLM_few_shot_results.csv    # Aggregated LLM results
├── DeBERTA-model-zero-shot.ipynb   # DeBERTa embedding-based solver (no examples)
├── DeBERTa-few-shot.ipynb          # DeBERTa solver with example groups
├── DeBERTa-lora-final.ipynb        # Final DeBERTa LoRA evaluation
├── DeBERTa-lora-experiment.ipynb   # LoRA ablations / experiments
├── bert-model-zero-shot.ipynb      # BERT-based puzzle solving
├── generation-BERT-few-shot.ipynb  # BERT MLM pipeline for generating distractors
├── LLM-model-zero-shot.ipynb       # LLM-based puzzle solving (notebook)
├── LLM-model-few-shot.ipynb        # LLM few-shot evaluation (notebook)
├── LLMprompter-zero-shot.py        # Scripted LLM evaluation via API key
├── random_grouping_baseline.ipynb  # Random grouping baseline
├── data_exploration.ipynb          # Data exploration / sanity checks
├── data_evaluation.ipynb           # DeBERTa evaluation and plots
├── Visualizations.ipynb            # Additional visualizations and statistics
├── human_test.py                   # CLI script for human baseline study
├── human_test_eval.ipynb           # Analysis of human_test results
├── project.ipynb                   # High-level project walkthrough
├── requirements.txt
├── connections_env.yml             # Conda environment definition
├── connections_env_verbose.yaml    # Verbose Conda environment definition
├── .gitignore
└── .env.example                    # Example environment variables for API keys
```

Run notebooks from the repo root so `import conn` and `data_loader` work.