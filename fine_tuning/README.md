# Transformer Fine-Tuning

**Owner:** Member B · **Week 2**

Fine-tunes a pre-trained Transformer (DistilBERT recommended; RoBERTa/PEFT optional) using the Hugging Face `Trainer` API.

> **Reuse `../data_processing/`**: load the same cleaned data and the same canonical train/val/test split so results are directly comparable to the baseline. Report metrics with the shared `../classical_model/metrics.py`.

## Planned contents (Week 2)

| File / folder | Purpose |
|---|---|
| `train_transformer.py` / `.ipynb` | Tokenize + fine-tune the Transformer |
| `hyperparameter_log.md` | Log of runs (learning rate, batch size, epochs) + results |
| `results/` | Saved model metrics, comparison vs. baseline |