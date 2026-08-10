# GenDetect — Detecting AI-Generated Text

Human vs. AI text classification (**0 = human-written**, **1 = AI-generated**).

LLMs (e.g., GPT) have made it hard to tell human-written from machine-generated text — with
implications for academic integrity, content moderation, and misinformation detection.

**Goal:** build, evaluate, and *compare* two very different NLP models — a classical **TF-IDF +
linear classifier** baseline and a fine-tuned **DistilBERT** Transformer — and critically assess
whether a high accuracy score actually means real detection ability.

**Headline finding:** both models hit **~99.97% / 99.95% macro-F1** on the benchmark, but that score
is a *dataset shortcut* — out of distribution they **fail in opposite directions** (see
[`RESULTS_REPORT.md`](RESULTS_REPORT.md) §8). High on the benchmark, unproven in the wild.

## Repository Structure

```
final_project/
├── README.md                  # This file — overview, structure, how to run
├── PROJECT_PLAN.md            # Requirements, approach, rubric
├── RESULTS_REPORT.md          # Full results + interpretation + cross-model comparison (§8)
├── requirements.txt           # Python dependencies
├── sources.txt                # Dataset / reference sources
├── .gitignore
│
├── data_processing/           # SHARED — dataset, EDA, preprocessing, canonical split
│   ├── README.md
│   ├── data/                  # raw/ + processed/ (git-ignored; download via Kaggle)
│   ├── eda.ipynb              # EDA: text length, vocabulary, class distribution
│   ├── preprocessing.py       # reusable cleaning + tokenization pipeline
│   └── data_split.py          # ONE canonical train/val/test split (reused by both models)
│
├── classical_model/           # Baseline: TF-IDF + linear classifier
│   ├── README.md
│   ├── tfidf_features.py       # TF-IDF feature builder (no-leakage fit)
│   ├── metrics.py             # SHARED metrics (accuracy / precision / recall / F1)
│   ├── train_baseline.ipynb    # train + evaluate + diagnostics
│   └── classical_results/      # scores, plots, saved models (*.joblib)
│
├── transformer_model/         # Fine-tuned DistilBERT (Hugging Face Trainer)
│   ├── README.md
│   ├── train_transformer.py
│   ├── evaluate_test.py
│   ├── error_analysis.py
│   ├── EXPERIMENT_LOG.md       # hyperparameter runs + results
│   └── results/               # test metrics, confusion matrix, misclassified.csv
│                              # (model weights are git-ignored — too large)
│
├── generalization_test/       # Out-of-distribution cross-model test
│   ├── predict_demo.ipynb      # runs BOTH models on unseen text
│   ├── ood_pdf_comparison.csv
│   ├── ood_vs_indist.png       # in-dist vs OOD chart (report)
│   └── ood_slide_chart.png     # cleaner 2-model chart (slides)
│
├── test_data/                 # 6 paired human/AI PDFs (3 topics) for the OOD probe
│
└── docs/                      # Final report + slides (PDF/PPTX)
```

`data_processing/` is the shared source of truth: the classical model and the Transformer consume
its cleaned data and the **same** train/val/test split, and both are scored by the shared
`classical_model/metrics.py`, so results are directly comparable.

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data** — download the Kaggle *AI vs Human Text* dataset into
   `data_processing/data/raw/` (not committed).
   - Or skip processing and grab the **pre-processed data + canonical split** here:
     [Google Drive](https://drive.google.com/drive/folders/14NRNFwWsYVeNV_FyXGSq3zgyLwnDnKkB?usp=sharing) →
     place the files in `data_processing/data/processed/`.
3. **Data processing** — run `data_processing/eda.ipynb`, then generate the canonical split with
   `data_processing/data_split.py`.
4. **Classical baseline** — run `classical_model/train_baseline.ipynb` to build TF-IDF features,
   train the classifiers, and produce diagnostics.
5. **Transformer** — run `transformer_model/train_transformer.py` to fine-tune DistilBERT, then
   `evaluate_test.py` for test metrics.
6. **Generalization test** — run `generalization_test/predict_demo.ipynb` to score **both** models
   on the out-of-distribution PDFs and short-text probes.

## Documentation

- **[RESULTS_REPORT.md](RESULTS_REPORT.md)** — full results, interpretation, error analysis, and the
  cross-model (Classical vs Transformer) comparison in §8.
- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** — requirements, timeline, and grading rubric.
- **[transformer_model/EXPERIMENT_LOG.md](transformer_model/EXPERIMENT_LOG.md)** — DistilBERT runs.
- **`docs/`** — final report and presentation slides.

## Team

| Member | Role | Ownership |
|---|---|---|
| **Sakib Mansuri** (Member A) | Classical ML & Data | dataset, EDA, preprocessing, TF-IDF baseline, error analysis, generalization test |
| **Kunal** (Member B) | Transformer & Deployment | Hugging Face pipeline, DistilBERT fine-tuning + hyperparameter tuning, reproducibility |

Analysis, report, and presentation are shared, with per-section ownership.