"""
data_split.py — Build the ONE canonical train/val/test split (SEA 820 · AITrace)

WHY THIS FILE EXISTS
--------------------
Both the classical model and the Transformer must be trained and evaluated on the
EXACT same examples, or the comparison is meaningless. This script produces that
single shared split once and saves it to `data/processed/`. Everyone loads from here.

WHAT IT GUARANTEES
  * Stratified split -> train/val/test keep the same human:AI ratio as the full data
    (VALUE: the dataset is imbalanced ~63% human / 37% AI, so a random split could
    skew a small test set; stratifying keeps evaluation fair and stable).
  * Fixed random seed -> the split is reproducible run-to-run and machine-to-machine.
  * Deduplicated + null-free -> no leakage of identical texts across splits.

USAGE
    python data_split.py                     # full dataset, default 70/15/15 split
    python data_split.py --sample 50000      # quick iteration on a subset
    python data_split.py --no-classical      # skip the slow heavy-clean column
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing import transformer_text, classical_preprocess


HERE = os.path.dirname(os.path.abspath(__file__))

# Default paths auto-adapt to where you run:
#   * On Colab, if the shared Drive folder exists, read/write there so the split
#     persists across sessions and is shared by both members.
#   * Otherwise fall back to the local repo layout.
# Override either with --input / --outdir at any time.
DRIVE_DIR = "/content/drive/MyDrive/NLP_project"
if os.path.isdir(DRIVE_DIR):
    DEFAULT_INPUT = os.path.join(DRIVE_DIR, "AI_Human.csv")
    DEFAULT_OUTDIR = os.path.join(DRIVE_DIR, "processed")
else:
    DEFAULT_INPUT = os.path.join(HERE, "data", "AI_Human.csv")
    DEFAULT_OUTDIR = os.path.join(HERE, "data", "processed")


def load_and_clean(input_path: str, sample: int | None, seed: int) -> pd.DataFrame:
    """Load the CSV and remove leakage risks BEFORE splitting.

    WHAT: standardize the label to int, drop empty texts and exact duplicates.
    VALUE: duplicates landing in both train and test would inflate scores; cleaning
    here means every split file downstream is already trustworthy.
    """
    print(f"[load] reading {input_path} ...")
    df = pd.read_csv(input_path, usecols=["text", "generated"])
    print(f"[load] raw rows: {len(df):,}")

    # Label: 0.0/1.0 float -> clean int {0=human, 1=AI}.
    df["label"] = df["generated"].astype(float).round().astype(int)
    df = df.drop(columns=["generated"])

    # Drop nulls / blanks / duplicates (leakage + noise removal).
    before = len(df)
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    print(f"[clean] dropped {before - len(df):,} null/blank/duplicate rows "
          f"-> {len(df):,} unique rows")

    if sample is not None and sample < len(df):
        # Stratified downsample keeps the class ratio while iterating quickly.
        df, _ = train_test_split(
            df, train_size=sample, stratify=df["label"], random_state=seed
        )
        print(f"[sample] using stratified subset of {len(df):,} rows")

    dist = df["label"].value_counts(normalize=True).sort_index()
    print(f"[clean] class balance -> human(0): {dist.get(0, 0):.1%} | "
          f"AI(1): {dist.get(1, 0):.1%}")
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Stratified 3-way split: train / validation / test.

    val_size is expressed as a fraction of the WHOLE dataset (not of the remainder),
    so `--test-size 0.15 --val-size 0.15` gives a clean 70/15/15.
    VALUE: a separate validation set lets us tune hyperparameters WITHOUT touching
    the test set, so the final test score stays an honest estimate of generalization.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    # Re-scale val fraction relative to the remaining train_val pool.
    val_rel = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["label"], random_state=seed
    )
    return train, val, test


def _map_with_progress(series: pd.Series, fn, label: str) -> list:
    """Apply `fn` over a column with light progress printing.
    VALUE: the heavy clean over hundreds of thousands of rows is slow; progress
    output reassures the user it's working rather than hung.
    """
    out, n, t0 = [], len(series), time.time()
    step = max(1, n // 10)
    for i, v in enumerate(series):
        out.append(fn(v))
        if (i + 1) % step == 0:
            print(f"   [{label}] {i + 1:,}/{n:,} ({(i + 1) / n:.0%}) "
                  f"in {time.time() - t0:.0f}s")
    return out


def add_columns(df: pd.DataFrame, do_classical: bool) -> pd.DataFrame:
    """Attach model-ready text columns to a split.

    text_clean     -> light clean, Transformer input (cheap).
    text_classical -> heavy clean string, TF-IDF input (slow; toggle with --no-classical).
    """
    df = df.copy()
    df["text_clean"] = _map_with_progress(df["text"], transformer_text, "clean")
    if do_classical:
        df["text_classical"] = _map_with_progress(
            df["text"], lambda t: classical_preprocess(t, method="lemmatize"), "classical"
        )
    return df


def save_split(df: pd.DataFrame, outdir: str, name: str):
    """Save one split as Parquet (fast/compact) with a CSV fallback.
    VALUE: Parquet preserves dtypes and loads far faster than CSV for repeated
    experiments; the CSV fallback keeps it working without pyarrow installed.
    """
    os.makedirs(outdir, exist_ok=True)
    try:
        path = os.path.join(outdir, f"{name}.parquet")
        df.to_parquet(path, index=False)
    except Exception:  # pyarrow/fastparquet not available
        path = os.path.join(outdir, f"{name}.csv")
        df.to_csv(path, index=False)
    print(f"[save] {name}: {len(df):,} rows -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Build the canonical train/val/test split.")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--sample", type=int, default=None, help="use a stratified subset")
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    ap.add_argument("--no-classical", action="store_true",
                    help="skip the slow heavy-clean TF-IDF column")
    args = ap.parse_args()

    df = load_and_clean(args.input, args.sample, args.seed)
    train, val, test = make_splits(df, args.test_size, args.val_size, args.seed)
    print(f"[split] train={len(train):,}  val={len(val):,}  test={len(test):,}")

    do_classical = not args.no_classical
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"[process] building columns for '{name}' ...")
        part = add_columns(part, do_classical)
        save_split(part, args.outdir, name)

    print("[done] canonical split ready. Both models must load from:", args.outdir)


if __name__ == "__main__":
    main()