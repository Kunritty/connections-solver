# Connections Solver

Exploration of a BERT-based model and tuned LLM to solve real NYT Connections puzzle and artificially generated puzzles. Artificial puzzes will be generated with an LLM pipeline.


## Important References:


[Making New Connections: LLMs as Puzzle Generators for The New York Times' Connections Word Game](https://arxiv.org/abs/2407.11240)  
- Using LLMs as a method for generating synthetic connections puzzle data
- Our baseline for attempts to generate synthetic training datasets

[LLMs as Method Actors: A Model for Prompt Engineering and Architecture](https://arxiv.org/abs/2411.05778v2)
- Proof of concept for current LLMs' ability to comprehend and solve complex semantic connections puzzles
- Reference for experiments involving LLM evaluation and comparison to our fine-tuned models

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

## File structure

```
connections-solver/
├── conn/                         # Shared library for DeBERTa notebooks
│   ├── __init__.py               # Re-exports public API
│   ├── data.py                   # Data loading and preprocessing utilities
│   ├── encoder.py                # Wrapper for DeBERTa models and phrase embedding logic
│   ├── search.py                 # Greedy search algorithms for group finding
│   ├── solver.py                 # Core solver logic (baseline, zero-shot, few-shot strategies)
│   ├── fine_tuning.py            # LoRA fine-tuning scripts and adapter loading
│   └── metrics.py                # Evaluation metrics (accuracy, min swaps)
├── adapters/                     # LoRA fine-tuned DeBERTa model adapters (train outputs)
├── DeBERTA-model-zero-shot.ipynb # DeBERTa embedding-based solver (no examples)
├── DeBERTa-few-shot.ipynb        # DeBERTa solver with example groups from train
├── DeBERTa-lora-fine-tuning.ipynb# DeBERTa solver with LoRA fine-tuning
├── bert-model-zero-shot.ipynb    # BERT-based puzzle solving
├── generation-BERT-few-shot.ipynb# BERT MLM pipeline for generating false categories
├── LLM-model-zero-shot.ipynb     # LLM-based puzzle solving
├── random_grouping_baseline.ipynb# Random grouping baseline
├── Visualizations.ipynb          # Visualizations and statistics
├── requirements.txt
├── connections_env.yml           # Conda environment definition
├── connections_env_verbose.yaml  # Verbose Conda environment definition
├── .gitignore
├── data/                         # Datasets and example data
│   ├── connections_words.csv     # Word list / puzzle vocabulary
│   ├── examples.txt              # Example categories (JSON) for few-shot generation
│   ├── puzzle_data1.csv
│   ├── puzzle_data2.csv
│   ├── puzzle_data3.csv
│   └── word_bank.txt             # Word bank for generation pipeline
```

Run notebooks from the repo root so `import conn` works for the DeBERTa notebooks.