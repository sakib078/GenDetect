# TODO — Detecting AI-Generated Text

## Remaining
- [ ] (optional) Final report polish / commit + push

## Done
Ran `generalization_test/predict_demo.ipynb` → OOD result: classical 83%, **DistilBERT 50% (worse)**;
`MODEL_COMPARISON.md` §B filled with numbers, verdict, and `ood_vs_indist.png`.

Part 1 — data processing, EDA, canonical split (`data_processing/`). Part 2 — classical baseline:
TF-IDF + LogReg/LinearSVC, tuning, diagnostics, shortcut analysis, `RESULTS_REPORT.md`
(`classical_model/`). Cross-model — validated Member B's DistilBERT (fair split + leakage fix),
wrote `MODEL_COMPARISON.md`, and moved/​wired `predict_demo.ipynb` into `generalization_test/` for
the OOD comparison.
