"""
metrics.py — Shared evaluation for AI-vs-Human classifiers (SEA 820 · GenDetect)

WHY THIS IS SHARED
------------------
Both the classical baseline (Member A) and the DistilBERT model (Member B) must be
scored the EXACT same way, or the headline question — "did the Transformer beat the
baseline?" — is unanswerable. Member B imports this module too, so every number is
directly comparable.

WHY THESE METRICS (not just accuracy)
-------------------------------------
The dataset is imbalanced (~63% human / 37% AI). Plain accuracy is misleading: a model
that always predicts "human" already scores ~63% while catching ZERO AI text. So the
headline metric here is **macro-F1** (averages the two classes equally) alongside
**per-class precision/recall**, which reveal whether we actually catch the minority AI
class. This directly implements the EDA's "handling class imbalance" decision.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Canonical label convention for the whole project: 0 = human, 1 = AI-generated.
LABELS = [0, 1]
TARGET_NAMES = ["human", "AI"]


def compute_metrics(y_true, y_pred) -> dict:
    """Return a flat dict of the numbers we report everywhere.

    VALUE: one function = one definition of "the metrics", so the baseline and the
    Transformer can never accidentally be scored differently. Includes macro-F1 (the
    imbalance-aware headline) plus per-class precision/recall/F1 and class supports.
    """
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    for i, name in enumerate(TARGET_NAMES):
        out[f"precision_{name}"] = p[i]
        out[f"recall_{name}"] = r[i]
        out[f"f1_{name}"] = f[i]
        out[f"support_{name}"] = int(s[i])
    return out


def report(y_true, y_pred) -> str:
    """Full sklearn text report (per-class + macro/weighted averages).
    VALUE: the human-readable table to paste straight into the write-up.
    """
    return classification_report(
        y_true, y_pred, labels=LABELS, target_names=TARGET_NAMES,
        digits=4, zero_division=0,
    )


def print_metrics(name: str, y_true, y_pred) -> dict:
    """Pretty-print the headline numbers for one model and return the metrics dict."""
    m = compute_metrics(y_true, y_pred)
    print(f"=== {name} ===")
    print(f"  accuracy : {m['accuracy']:.4f}")
    print(f"  macro-F1 : {m['macro_f1']:.4f}   <- headline metric (imbalance-aware)")
    print(f"  AI    P/R/F1: {m['precision_AI']:.4f} / {m['recall_AI']:.4f} / {m['f1_AI']:.4f}")
    print(f"  human P/R/F1: {m['precision_human']:.4f} / {m['recall_human']:.4f} / {m['f1_human']:.4f}")
    return m


def results_table(results: dict):
    """Turn {model_name: metrics_dict} into a tidy comparison DataFrame.
    VALUE: the side-by-side leaderboard for the report, sorted by the headline metric.
    """
    import pandas as pd

    df = pd.DataFrame(results).T
    cols = ["accuracy", "macro_f1", "f1_AI", "recall_AI", "precision_AI", "f1_human"]
    df = df[[c for c in cols if c in df.columns]]
    return df.sort_values("macro_f1", ascending=False)


def plot_confusion(y_true, y_pred, title="Confusion Matrix",
                   normalize=None, savepath=None):
    """Plot (and optionally save) the confusion matrix.
    VALUE: shows WHERE errors fall — e.g. how many humans get flagged as AI, which is
    the ethically sensitive false-positive direction the EDA warned about.
    normalize='true' shows per-class rates instead of raw counts.
    """
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues",
              values_format=".2f" if normalize else "d")
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=120, bbox_inches="tight")
        print(f"[saved] {savepath}")
    return fig