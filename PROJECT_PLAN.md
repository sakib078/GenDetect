# SEA 820 NLP Final Project — Project Plan

**Project:** Detecting AI-Generated Text (Human vs. AI text classification)
**Team Size:** 2 students
**Duration:** 3 Weeks
**Grade Weight:** 20% of course (scored out of 100)

> This document captures the full project requirements and our approach. The per-member task assignments live in **[WORK_DIVISION.md](WORK_DIVISION.md)**.

---

## 1. Project Overview & Goal

LLMs (e.g., GPT) have made it hard to tell human-written from machine-generated text — with implications for academic integrity, content moderation, and misinformation detection.

**Goal:** Build, evaluate, and compare different NLP models that classify a given text as **"human-written" (0)** or **"AI-generated" (1)**.

The project requires: text preprocessing, feature representation, classic ML, modern deep learning (Transformers), rigorous evaluation, model behavior analysis, and an ethical analysis.

---

## 2. Core Learning Objectives (must all be demonstrated)

1. **Apply Transfer Learning** — fine-tune a pre-trained Transformer (e.g., DistilBERT or RoBERTa).
2. **Implement and Compare Models** — at least two types: a *classic* model (e.g., Logistic Regression + TF-IDF) **and** a Transformer.
3. **Conduct Rigorous Evaluation** — beyond accuracy: precision, recall, F1-score + detailed error analysis.
4. **Analyze Model Behavior** — investigate what features/patterns distinguish human vs. AI text.
5. **Consider Ethical Implications** — societal impact, potential biases, ethical challenges of deploying a detector.

---

## 3. Dataset

- **Dataset:** "AI vs. Human Text" dataset from **Kaggle** (curated for this classification challenge).
- **Content:** Student essays and other text forms — relevant and challenging.
- **Labels:** `0` = human, `1` = AI-generated. Balanced examples from human authors and various AI models.

---

## 4. Approach & Timeline (3 Weeks)

### Week 1 — Foundations & Classic Model
**Deliverable:** A working baseline model + a 1-page project plan.
- Confirm partner; set up shared **GitHub** repo + **Google Colab** environment.
- Download & load the Kaggle dataset.
- **EDA** — analyze text length, vocabulary, class distribution.
- Build a robust **preprocessing pipeline** — tokenization, cleaning, handling text lengths.
- Create **TF-IDF features** and train a classic classifier (Logistic Regression / Naive Bayes / SVM) with scikit-learn.
- Establish a **baseline score** (the number the Transformer must beat).
- Agree on a single train/validation/test **split** to be reused by both models.
- Write the **1-page project plan** (approach + task assignments).

### Week 2 — Fine-Tuning a Transformer Model
**Deliverable:** A fine-tuned Transformer + initial comparison results.
- Load data with the **Hugging Face `datasets`** library.
- Set up the **tokenizer** for a pre-trained model (DistilBERT-base-uncased).
- Choose a model — **DistilBERT recommended** (RoBERTa optional for stretch).
- Fine-tune using the **Hugging Face `Trainer` API**.
- **Hyperparameter experiments** — learning rate, batch size, epochs; keep an **experiment log**.
- Evaluate on the **test set**; compare against the Week-1 baseline (better? by how much?).

### Week 3 — Analysis, Reporting & Presentation
**Deliverable:** Final submission — code + report (PDF) + slides (PDF).
- **Error analysis** — inspect misclassified examples (FP/FN); hypothesize why; suggest improvements.
- **Ethical considerations** — who benefits/is harmed, dataset/model bias, non-native-speaker flagging.
- **Final report** — Intro, Methods, Results, Analysis, Ethics, Conclusion.
- **Code cleanup** + `README.md` that makes reproduction trivial.
- **Presentation slides** (5–7 min) + rehearsal.

---

## 5. Submission Requirements (all three required)

1. **Source Code** — link to a well-organized **GitHub** repo with all code (notebooks/scripts) and a `README.md`.
2. **Final Report (PDF)** — professional report covering the project start to finish.
3. **Presentation Slides (PDF)** — slides for a **5–7 minute** final presentation.

---

## 6. Grading Rubric (how we are scored — target every line)

| Category | Weight | Criterion | Points |
|---|---|---|---|
| **1. Code & Implementation** | 40% | A. Baseline "Classic" Model | /10 |
| | | B. Transformer Fine-Tuning | /20 |
| | | C. Code Quality & Reproducibility | /10 |
| **2. Evaluation & Analysis** | 30% | A. Performance Metrics & Comparison | /15 |
| | | B. Error Analysis & Insights | /15 |
| **3. Final Report & Ethics** | 20% | A. Report Clarity & Structure | /10 |
| | | B. Ethical Considerations | /10 |
| **4. Presentation** | 10% | A. Clarity & Delivery | /10 |
| **Total** | | | **/100** |

**To hit the "Excellent" band, aim for the stretch goals:**
- Baseline: compare **2+ classic models** or do feature engineering beyond basic TF-IDF.
- Transformer: show **systematic hyperparameter tuning**; optionally try a more advanced model/technique (**RoBERTa, PEFT/LoRA**).
- Metrics: full comparison table/plots + discuss **metric trade-offs** in context.
- Error analysis: inspect **specific** false positives/negatives with hypotheses + improvement suggestions.
- Ethics: connect discussion back to **your specific findings** (e.g., non-native English speakers flagged incorrectly).
- Presentation: **both members contribute**, stay within time.

---

## 7. Suggested Repository Structure

```
ai-text-detector/
├── README.md                 # How to run + reproduce
├── requirements.txt
├── data/                     # (or download script if data too large for git)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_transformer_finetune.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── preprocessing.py      # shared pipeline
│   ├── data_split.py         # shared train/val/test split
│   ├── metrics.py            # shared evaluation module
│   └── train_transformer.py
├── experiments/
│   └── hyperparameter_log.md # log of runs + results
├── results/                  # saved plots, tables, metrics
└── report/
    ├── final_report.pdf
    └── slides.pdf
```

---

## 8. Definition of Done (checklist against the rubric)

- [ ] Baseline classic model implemented + valid baseline score recorded (bonus: 2+ models / feature engineering)
- [ ] Transformer fine-tuned via HF Trainer with a documented hyperparameter log (bonus: RoBERTa/PEFT)
- [ ] Clean, commented, organized code + clear `README.md` that makes reproduction trivial
- [ ] Accuracy, precision, recall, F1 reported for **both** models, in tables/plots, with trade-off discussion
- [ ] Error analysis on **specific** misclassified examples with hypotheses + improvement ideas
- [ ] Professional report with all required sections, proofread, no major grammar errors
- [ ] Ethical section tied to specific findings (bias, non-native speakers, misuse)
- [ ] 5–7 min presentation, both members contribute, rehearsed and timed
- [ ] GitHub repo link + report PDF + slides PDF submitted