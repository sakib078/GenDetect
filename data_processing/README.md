# Data Processing & Exploration (SHARED)

**Owner:** Member A · **Used by:** Both members (Member B reuses the split, cleaning, and processed data in Week 2).

This folder is the single source of truth for the dataset, EDA, preprocessing, and the train/val/test split. Both the classical model and the Transformer must consume outputs from here so results are comparable.

## Contents

| File / folder | Purpose |
|---|---|
| `data/raw/` | Original Kaggle "AI vs Human Text" dataset (not committed — see `.gitignore`) |
| `data/processed/` | Cleaned data + saved train/val/test splits (shared by both models) |
| `eda.ipynb` | Exploratory analysis: text length, vocabulary, class distribution |
| `preprocessing.py` | Reusable cleaning + tokenization pipeline |
| `data_split.py` | Creates the ONE canonical train/val/test split — reused by everyone |

## Rules

- **One split, reused everywhere.** Generate the split once here; the classical model and the Transformer both load it. Never re-split.
- Keep `preprocessing.py` importable so Member B can call the same functions.
- Don't commit large raw data — download via script/instructions instead.