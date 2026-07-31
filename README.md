# GenDetect

Detecting AI-Generated Text (Human vs. AI text classification)

LLMs (e.g., GPT) have made it hard to tell human-written from machine-generated text — with implications for academic integrity, content moderation, and misinformation detection.

Goal: Build, evaluate, and compare different NLP models that classify a given text as "human-written" (0) or "AI-generated" (1).

The project requires: text preprocessing, feature representation, classic ML, modern deep learning (Transformers), rigorous evaluation, model behavior analysis, and an ethical analysis.

## Repository Structure

```
AITrace/
├── PROJECT_PLAN.md         # Requirements, approach, rubric
├── WORK_DIVISION.md        # Per-member task assignments
├── TODO.md                 # Member A's Week 1 checklist
│
├── data_processing/        # SHARED — dataset, EDA, preprocessing, canonical split
│   ├── data/               # raw/ + processed/ (git-ignored)
│   ├── eda.ipynb
│   ├── preprocessing.py    # reusable cleaning + tokenization pipeline
│   └── data_split.py       # ONE canonical train/val/test split (reused by both models)
│
├── classical_model/        # Baseline: TF-IDF + classic classifier
│   ├── tfidf_features.py
│   ├── train_baseline.ipynb
│   ├── metrics.py          # shared metrics (accuracy/precision/recall/F1)
│   └── results/
│
├── fine_tuning/            # Transformer fine-tuning (DistilBERT)
│   ├── train_transformer.py
│   ├── hyperparameter_log.md
│   └── results/
│
└── report/                 # Final report + slides (PDF)
```

`data_processing/` is the shared source of truth: both the classical model and the Transformer consume its cleaned data and the **same** train/val/test split so results are comparable.

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data** — download the Kaggle "AI vs Human Text" dataset into `data_processing/data/raw/` (not committed).
3. **Data processing** — run `data_processing/eda.ipynb`, then generate the canonical split with `data_processing/data_split.py`.
4. **Baseline** — run `classical_model/train_baseline.ipynb` to build TF-IDF features and train the classic model.
5. **Transformer** — run `fine_tuning/train_transformer.py` to fine-tune DistilBERT and compare against the baseline.

## Docs

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** — full requirements, timeline, and grading rubric.
- **[WORK_DIVISION.md](WORK_DIVISION.md)** — who does what.
