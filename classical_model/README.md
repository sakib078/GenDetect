# Classical Model (Baseline)

**Owner:** Sakib S · **Week 1**

The "classic" baseline the Transformer must beat. Consumes the cleaned data and the canonical split from `../data_processing/`.

## Contents

| File / folder | Purpose |
|---|---|
| `tfidf_features.py` | Build TF-IDF features from preprocessed text |
| `train_baseline.ipynb` | Train + evaluate classic classifier(s): Logistic Regression / Naive Bayes / SVM |
| `metrics.py` | Shared evaluation module — accuracy, precision, recall, F1 (also used by Member B) |
| `results/` | Saved baseline scores, tables, plots |

## Goal

- Establish a valid **baseline score** on the shared test set.
- For top marks: train **2+ classic models** or add feature engineering beyond basic TF-IDF, then compare.
- Report accuracy, precision, recall, F1 — not just accuracy.