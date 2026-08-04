"""
error_analysis.py — inspect DistilBERT's misclassifications.
"""

import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_processing", "data", "processed")


def main():
    errors = pd.read_csv(os.path.join(RESULTS_DIR, "misclassified.csv"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    fp = errors[(errors["pred"] == 1) & (errors["label"] == 0)]
    fn = errors[(errors["pred"] == 0) & (errors["label"] == 1)]

    print(f"False Positives (human flagged as AI): {len(fp)}")
    print(f"False Negatives (AI missed as human): {len(fn)}")

    print("\n=== 3 False Positives ===")
    for _, row in fp.head(3).iterrows():
        print(f'  "{row["text_clean"][:150]}..."\n')

    print("=== 3 False Negatives ===")
    for _, row in fn.head(3).iterrows():
        print(f'  "{row["text_clean"][:150]}..."\n')

    errors["word_count"] = errors["text_clean"].str.split().str.len()
    test_df["word_count"] = test_df["text_clean"].str.split().str.len()

    print(f"Avg word count — misclassified: {errors['word_count'].mean():.1f}")
    print(f"Avg word count — full test set: {test_df['word_count'].mean():.1f}")


if __name__ == "__main__":
    main()
