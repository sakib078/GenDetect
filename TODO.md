# TODO — Member A (Week 1)

> My personal checklist for Week 1. Scope: **Data Processing & Exploration** first, then the **Classical Model**.
> Work order matters — the whole team depends on Part 1 outputs (shared split + preprocessing).
> See [WORK_DIVISION.md](WORK_DIVISION.md) and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the big picture.

---

## Part 1 — Data Processing & Exploration  → `data_processing/`  *(DO THIS FIRST)*

### Setup & load
- [ ] Download the Kaggle "AI vs Human Text" dataset into `data_processing/data/raw/`
- [ ] Write a small loader (script or top of `eda.ipynb`) to read the dataset into a DataFrame
- [ ] Sanity check: row count, columns, label values are `0` (human) / `1` (AI)

### Exploratory Data Analysis → `data_processing/eda.ipynb`
- [ ] **Class distribution** — how balanced is human vs. AI? (bar chart)
- [ ] **Text length** — distribution of word/char counts per class (histograms)
- [ ] **Vocabulary** — vocab size, most common words, differences between classes
- [ ] Check for nulls, duplicates, and any weird/garbage rows
- [ ] Write 3–5 bullet findings (feeds the report's "Dataset & EDA" section)

### Preprocessing pipeline → `data_processing/preprocessing.py`
- [ ] Decide + implement cleaning (lowercasing, punctuation, whitespace, etc.)
- [ ] Decide tokenization approach and how to handle very long / very short texts
- [ ] Make it an **importable function** so Member B can reuse it in Week 2
- [ ] Save cleaned data to `data_processing/data/processed/`

### Canonical split → `data_processing/data_split.py`  *(SHARED — critical)*
- [ ] Create ONE train / validation / test split (set a fixed `random_state`)
- [ ] Save the splits to `data_processing/data/processed/` so both models load the exact same test set
- [ ] Tell Member B the split is ready

---

## Part 2 — Classical (Baseline) Model  → `classical_model/`  *(AFTER Part 1)*

### Features → `classical_model/tfidf_features.py`
- [ ] Build TF-IDF features from the preprocessed text (scikit-learn `TfidfVectorizer`)
- [ ] Fit on train only; transform val/test

### Train baseline → `classical_model/train_baseline.ipynb`
- [ ] Train a classic classifier (Logistic Regression to start)
- [ ] **Stretch (for "Excellent"):** also train Naive Bayes and/or SVM and compare
- [ ] Tune a couple of basic settings (e.g., TF-IDF `ngram_range`, `max_features`, regularization `C`)

### Evaluate → `classical_model/metrics.py` (+ `results/`)
- [ ] Build a shared `metrics.py` returning **accuracy, precision, recall, F1** (Member B reuses this)
- [ ] Evaluate on the shared **test set**
- [ ] Record the **baseline score** — this is the number the Transformer must beat
- [ ] Save metrics table + any plots to `classical_model/results/`

---

## Part 3 — Week 1 wrap-up (shared with Member B)
- [ ] Confirm partner + repo/Colab setup is done
- [ ] Contribute to the **1-page project plan** deliverable (due end of Week 1)
- [ ] Commit + push my work to GitHub

---

## Definition of Done (my Week 1 scope)
- [ ] Dataset loaded; EDA complete with written findings
- [ ] Reusable `preprocessing.py` + one canonical saved split
- [ ] Working baseline model with a recorded score
- [ ] Metrics reported as accuracy / precision / recall / F1 (not just accuracy)
- [ ] Everything committed under `data_processing/` and `classical_model/`