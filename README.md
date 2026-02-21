# connections-solver
Exploration of a BERT-based model and tuned LLM to solve real NYT Connections puzzle and artificially generated puzzles. Artificial puzzes will be generated with an LLM pipeline.
<br>
### Important References:
<a href="https://arxiv.org/abs/2407.11240">Making New Connections: LLMs as Puzzle Generators for The New York Times' Connections Word Game
</a>
<br>
<a href="https://arxiv.org/abs/2411.05778v2">LLMs as Method Actors: A Model for Prompt Engineering and Architecture
</a>

## Setup
Install Conda (<a href="https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation">link</a>) and create a new Conda environment for this project.

Ensure conda is installed and added to PATH:

```
$ conda --version
```

Create and activate new environment. Make sure you're in this environment whenever installing and running code:

```
$ conda create --name cs175 python=3.10
$ conda activate cs175
(cs175) $
(cs175) $ conda deactivate
```

Once you're in the env install all the required libraries:

```
pip install -r requirements.txt
```

Your environment should be set up, now ensure the Python kernel for the notebook files is set to this new Conda environment, and you should be able to run them with no issues.

## File structure

```
connections-solver/
├── conn/                         # Shared library for DeBERTa notebooks
│   ├── __init__.py               # Re-exports public API
│   ├── data.py                   # load_connections_from_hf(), gold_groups_from_row()
│   ├── encoder.py                # DeBERTaEncoder (embed_phrase, group_similarity)
│   ├── solver.py                 # solve_puzzle_zero_shot(), solve_puzzle_few_shot()
│   └── metrics.py                # accuracy_zero_one, accuracy_min_swaps, evaluate()
├── DeBERTA-model-zero-shot.ipynb # DeBERTa embedding-based solver (no examples)
├── DeBERTa-few-shot.ipynb        # DeBERTa solver with example groups from train
├── bert-model-zero-shot.ipynb    # BERT-based puzzle solving
├── generation-BERT-few-shot.ipynb # BERT MLM pipeline for generating false categories
├── requirements.txt
├── .gitignore
├── data/                         # Datasets and example data
│   ├── connections_words.csv    # Word list / puzzle vocabulary
│   ├── examples.txt             # Example categories (JSON) for few-shot generation
│   ├── puzzle_data1.csv
│   ├── puzzle_data2.csv
│   ├── puzzle_data3.csv
│   └── word_bank.txt            # Word bank for generation pipeline
```

Run notebooks from the repo root so `import conn` works for the DeBERTa notebooks.
