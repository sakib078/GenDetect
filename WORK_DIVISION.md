# SEA 820 NLP Final Project — Work Division (2 Members)

> Per-member task assignments. The full requirements, approach, rubric, and repo structure live in **[PROJECT_PLAN.md](PROJECT_PLAN.md)**. Replace "Member A / Member B" with real names once the partner is confirmed.

---

## Roles Overview

The workload is split so **both members touch modeling, analysis, and writing** (the rubric rewards both members contributing).

- **Member A → "Classic ML & Data" lead** — owns the dataset, EDA, preprocessing pipeline, and the baseline classic model + its evaluation.
- **Member B → "Transformer & Deployment" lead** — owns the Hugging Face pipeline, Transformer fine-tuning + hyperparameter tuning, and reproducibility/repo hygiene.

Analysis, report, and presentation are **shared** with clear ownership per section (below).

---

## Week 1 — Foundations & Classic Model

| Task | Owner | Notes |
|---|---|---|
| Confirm partner; create shared **GitHub** repo + **Google Colab** environment | **Both** (B sets up repo, A sets up Colab) | Repo structure, branch strategy, `requirements.txt` |
| Download & load the Kaggle "AI vs Human Text" dataset | **Member A** | Store loader script/notebook |
| **EDA** — text length, vocabulary, class distribution | **Member A** | Plots + short written findings |
| Build **preprocessing pipeline** — tokenization, cleaning, handling text lengths | **Member A** | Reusable module shared with Member B |
| Create **TF-IDF features** | **Member A** | scikit-learn `TfidfVectorizer` |
| Train **classic classifier(s)** — Logistic Regression / Naive Bayes / SVM | **Member A** (B reviews) | For "Excellent": train **2+** and compare |
| Establish **baseline score** (the number the Transformer must beat) | **Member A** | Record accuracy/P/R/F1 |
| Set up train/validation/test **split** (shared, reused by both models) | **Member B** | Critical: both models must use the **same test set** |
| Write **1-page project plan** (approach + task assignments) | **Both** | B drafts, A reviews — **due end of Week 1** |

---

## Week 2 — Fine-Tuning a Transformer Model

| Task | Owner | Notes |
|---|---|---|
| Load data with **Hugging Face `datasets`** library | **Member B** | Reuse the shared split from Week 1 |
| Set up **tokenizer** for pre-trained model (DistilBERT-base-uncased) | **Member B** | Correct tokenization/formatting for the model |
| Choose model — **DistilBERT recommended** (RoBERTa optional for stretch) | **Member B** | Justify choice in report |
| Fine-tune using **Hugging Face `Trainer` API** | **Member B** | |
| **Hyperparameter experiments** — learning rate, batch size, epochs | **Member B** | **Keep an experiment log** (table of runs + results) |
| Evaluate fine-tuned model on **test set** | **Member B** | Same metrics as baseline |
| **Initial comparison** vs. Week-1 baseline — is it better, by how much? | **Both** | A supplies baseline numbers, B supplies Transformer numbers |
| Build shared **evaluation/metrics module** (accuracy, precision, recall, F1) | **Member A** | So both models report metrics identically |

---

## Week 3 — Analysis, Reporting & Presentation

| Task | Owner | Notes |
|---|---|---|
| **Error Analysis** — inspect misclassified examples (FP/FN), hypothesize why | **Member A** (best model) | Specific examples; patterns (short texts, topics) + improvement ideas |
| **Metric comparison** — final tables/plots for both models | **Member B** | Discuss trade-offs between metrics |
| **Ethical Considerations** — who benefits/is harmed, dataset/model bias, non-native flagging | **Member A** drafts, **Both** refine | Connect to actual project findings |
| **Final Report (PDF)** | **Both** (see section ownership below) | Professional, well-structured, proofread |
| **Code cleanup** — comments, organization | **Both** (each cleans own code) | |
| **`README.md`** — how to run, reproduce trivially | **Member B** | Make reproduction trivial for top marks |
| **Presentation slides (PDF)** — 5–7 min | **Both** | Both must present |
| **Rehearse presentation** (time it: 5–7 min) | **Both** | |

---

## Final Report — Section Ownership

Required sections: **Introduction, Methods, Results, Analysis, Conclusion** (+ Ethics).

| Report Section | Primary Owner | Reviewer |
|---|---|---|
| Introduction / Problem & Goal | Member B | Member A |
| Dataset & EDA | Member A | Member B |
| Methods — Preprocessing & Classic Baseline | Member A | Member B |
| Methods — Transformer Fine-Tuning & Hyperparameters | Member B | Member A |
| Results — Metrics & Model Comparison (tables/plots) | Member B | Member A |
| Error Analysis & Insights | Member A | Member B |
| Ethical Considerations | Member A | Member B |
| Conclusion & Future Work | Member B | Member A |

> Each member proofreads the other's sections for grammar/clarity (rubric penalizes grammar errors).

---

## Shared Responsibilities & Coordination Rules

- **Same test set for both models** — agree on the split in Week 1; never re-split.
- **Same metrics module** — both models report accuracy, precision, recall, F1 identically.
- **Commit regularly** to GitHub; use branches + pull requests so both members' contributions are visible (rubric values reproducibility & "all team members contribute").
- **Keep the experiment log up to date** during Week 2 — needed for the "systematic hyperparameter tuning" marks.
- **Weekly sync** at the end of each week to hand off the deliverable.