# Data Processing & Exploration (SHARED)

**Owner:** Member A · **Used by:** Both members (Member B reuses the split, cleaning, and processed data in Week 2).

This folder is the single source of truth for the dataset, EDA, preprocessing, and the train/val/test split. Both the classical model and the Transformer must consume outputs from here so results are comparable.

## Contents

| File / folder | Purpose |
|---|---|
| `data/AI_Human.csv` | Original Kaggle "AI vs Human Text" dataset — 487,235 rows (not committed — see `.gitignore`) |
| `data/processed/` | Cleaned data + saved `train/val/test` splits (shared by both models) |
| `eda.ipynb` | Exploratory analysis: class balance, text length, sentences, sub-word length, vocabulary |
| `preprocessing.py` | Reusable cleaning + tokenization pipeline (word / sentence / sub-word) |
| `data_split.py` | Creates the ONE canonical train/val/test split — reused by everyone |

## Pipeline design — one clean, two model-ready outputs

The classical model and the Transformer need **different** amounts of cleaning, so `preprocessing.py`
produces both from a shared base:

| Function | Output | For | Why |
|---|---|---|---|
| `basic_clean()` | light normalization (unicode, URLs/HTML, whitespace) | shared base | safe for both models |
| `classical_preprocess()` | lowercase + de-punct + de-stopword + lemmatize/stem string | **TF-IDF** | shrinks vocabulary so a linear model finds signal |
| `transformer_text()` | `basic_clean` only (keeps case & punctuation) | **DistilBERT** | the model's own sub-word tokenizer needs that context; over-cleaning hurts it |

> Pipeline order follows the lecture's *Basic NLP Pipeline*: Encoding → Tokenization → Normalization → Stop-word removal → Stemming/Lemmatization.

## How to run (Part 1)

```bash
pip install -r ../requirements.txt

# 1. Explore
jupyter notebook eda.ipynb

# 2. Build the canonical split (writes train/val/test to data/processed/)
python data_split.py                 # full dataset, 70/15/15, seed=42
python data_split.py --sample 50000  # fast subset while iterating
python data_split.py --no-classical  # skip the slow heavy-clean column
```

Quick pipeline sanity check: `python preprocessing.py` prints one sentence through every stage.

## Rules

- **One split, reused everywhere.** Generate the split once here; the classical model and the Transformer both load `data/processed/{train,val,test}`. Never re-split.
- Keep `preprocessing.py` importable so Member B can call the same functions.
- Don't commit large raw/processed data — download the dataset via Kaggle instead.