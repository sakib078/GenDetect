# DistilBERT Hyperparameter Experiment Log

| Run | Data | LR | Batch Size | Epochs | Val Macro-F1 | Test Macro-F1 | Notes |
|-----|------|-----|-----------|--------|----------------|----------------|-------|
| 1   | 20K sample | 2e-5 | 16 | 3 | 99.14% | 99.32% | initial pipeline validation |
| 2   | Full 487K  | 2e-5 | 16 | 3 | 99.94% | 99.95% | official final model |
| 3   | 20K sample | 5e-5 | 16 | 3 | 99.25% | 99.32% | hyperparameter comparison — higher LR |
