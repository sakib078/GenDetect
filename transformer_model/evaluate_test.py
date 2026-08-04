"""
evaluate_test.py — Final TEST-set evaluation of the fine-tuned DistilBERT model.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classical_model"))
from metrics import print_metrics, plot_confusion  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "results", "best_model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_processing", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    device = get_device()
    model.to(device)
    print(f"[device] using: {device}")

    preds = []
    batch_size = 32
    texts = test_df["text_clean"].tolist()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  [predict] {i + len(batch):,}/{len(texts):,}")

    y_true = test_df["label"].values
    y_pred = np.array(preds)

    m = print_metrics("DistilBERT (TEST)", y_true, y_pred)
    plot_confusion(y_true, y_pred, title="DistilBERT — Test Confusion Matrix",
                   savepath=os.path.join(RESULTS_DIR, "confusion_test.png"))

    with open(os.path.join(RESULTS_DIR, "transformer_score.txt"), "w") as f:
        f.write("DistilBERT fine-tuned (TEST)\n")
        for k, v in m.items():
            f.write(f"{k}: {v}\n")

    test_df = test_df.copy()
    test_df["pred"] = y_pred
    errors = test_df[test_df["label"] != test_df["pred"]]
    errors.to_csv(os.path.join(RESULTS_DIR, "misclassified.csv"), index=False)
    print(f"\nSaved {len(errors)} misclassified examples to results/misclassified.csv")


if __name__ == "__main__":
    main()
