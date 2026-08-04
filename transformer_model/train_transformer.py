"""
train_transformer.py — Fine-tune DistilBERT on the canonical split (SEA 820 · GenDetect)
Member B — Transformer & Deployment lead.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "classical_model"))
from metrics import compute_metrics  # noqa: E402

MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_processing", "data", "processed")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TEXT_COL = "text_clean"
LABEL_COL = "label"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.parquet")
    df = pd.read_parquet(path)
    return df[[TEXT_COL, LABEL_COL]].rename(columns={TEXT_COL: "text", LABEL_COL: "labels"})


def tokenize_batch(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return compute_metrics(labels, preds)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"[device] using: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_df = load_split("train")
    val_df = load_split("val")
    print(f"[data] train={len(train_df):,}  val={len(val_df):,}")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    val_ds = val_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "run"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=hf_compute_metrics,
    )

    print("[train] starting fine-tuning...")
    start = time.time()
    trainer.train()
    print(f"[train] done in {(time.time() - start) / 60:.1f} min")

    val_results = trainer.evaluate()
    print("\n=== Validation results ===")
    for k, v in val_results.items():
        print(f"  {k}: {v}")

    save_dir = os.path.join(RESULTS_DIR, "best_model")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[save] model saved to {save_dir}")


if __name__ == "__main__":
    main()
