"""
preprocessing.py — Shared text preprocessing pipeline (SEA 820 · GenDetect)

WHY THIS FILE EXISTS
--------------------
One preprocessing module feeds BOTH models so their inputs are consistent and
the comparison is fair. But the two models want DIFFERENT amounts of cleaning:

  * Classical model (TF-IDF + LogReg/NB/SVM)
      -> wants HEAVY cleaning: lowercase, strip punctuation/stopwords, lemmatize.
         Value: shrinks the vocabulary so a sparse linear model can find signal.

  * Transformer (DistilBERT fine-tuning)
      -> wants MINIMAL cleaning. It has its OWN sub-word tokenizer (WordPiece) and
         relies on casing, punctuation and stopwords for context.
         Value: over-cleaning here DESTROYS signal and hurts accuracy.

So this module exposes a shared light clean plus two model-specific outputs:

  basic_clean(text)          -> light normalization safe for BOTH models
  classical_preprocess(text) -> heavy pipeline, returns a space-joined string for TF-IDF
  transformer_text(text)     -> basic_clean only (keeps case/punctuation) for the HF tokenizer

PIPELINE ORDER (from the lecture's "Basic NLP Pipeline", order matters):
  i. Text Encoding -> ii. Tokenization -> iii. Normalization
  -> iv. Stop-word Removal -> v. Stemming / Lemmatization
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ---------------------------------------------------------------------------
# One-time NLTK resource download.
# WHAT: fetches the tokenizer models, stopword list, and lemmatizer database.
# ---------------------------------------------------------------------------
def _ensure_nltk():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()

# Build these ONCE at import time, not per-call.
_STOPWORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()
_LEMMATIZER = WordNetLemmatizer()

# Precompiled regexes for the light clean (compiled once = faster on large data).
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")  # used only in the HEAVY (classical) path



# STEP i + iii (light): TEXT ENCODING + BASIC NORMALIZATION  — SHARED BY BOTH

def basic_clean(text: str) -> str:
    """Light, meaning-preserving clean that is SAFE for both models.

    WHAT it does: normalizes unicode/encoding, strips URLs and HTML, and
    collapses runaway whitespace/newlines.
    WHY / VALUE: the raw essays contain HTML fragments, links, and irregular
    spacing that are noise for every model. Removing them helps TF-IDF AND the
    Transformer, WITHOUT throwing away the casing/punctuation the Transformer needs.
    """
    if not isinstance(text, str):
        text = str(text)

    # i. Text Encoding: normalize to canonical unicode, drop non-encodable bytes.
    #    VALUE: "café"/"café" and stray control chars become consistent.
    text = unicodedata.normalize("NFKC", text)
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")

    text = URL_PATTERN.sub(" ", text)         # links carry no human-vs-AI signal
    text = HTML_TAG_PATTERN.sub(" ", text)    # strip leftover markup
    text = WHITESPACE_PATTERN.sub(" ", text)  # collapse newlines/tabs/multi-space
    return text.strip()



# STEP ii: TOKENIZATION  (word / sentence / sub-word)

def tokenize_words(text: str) -> list[str]:
    """Word tokenization (NLTK).
    VALUE: splits text into word units — the atoms every downstream step and the
    TF-IDF vocabulary are built from.
    """
    return word_tokenize(text)


def tokenize_sentences(text: str) -> list[str]:
    """Sentence tokenization (NLTK).
    VALUE: gives sentence counts / avg sentence length — useful EDA features,
    since AI text often has more uniform sentence structure than human text.
    """
    return sent_tokenize(text)


@lru_cache(maxsize=4)
def _get_hf_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Lazily load a Hugging Face sub-word tokenizer (cached).
    VALUE: lets us measure length in the SAME sub-word units DistilBERT will use,
    so EDA can tell us how many essays exceed the 512-token limit (truncation risk).
    Imported lazily so the classical-model workflow doesn't require `transformers`.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def tokenize_subword(text: str, model_name: str = "distilbert-base-uncased") -> list[str]:
    """Sub-word (WordPiece) tokenization — how the Transformer actually sees text.
    VALUE: e.g. "unhappiness" -> ["un", "##happi", "##ness"]; counting these tells
    us the true input length for DistilBERT and informs the truncation strategy.
    """
    return _get_hf_tokenizer(model_name).tokenize(text)



# STEP iii–v (heavy): NORMALIZE + STOPWORD REMOVAL + STEM/LEMMATIZE  — CLASSICAL

def classical_preprocess(
    text: str,
    *,
    method: str = "lemmatize",   # "lemmatize" | "stem" | None
    remove_stopwords: bool = True,
    min_token_len: int = 2,
    return_tokens: bool = False,
):
    """Heavy cleaning pipeline for the TF-IDF / classical model.

    Steps (order matters): basic_clean -> lowercase -> keep letters only
    -> word tokenize -> drop stopwords & 1-char tokens -> stem OR lemmatize.

    method:
        "lemmatize" -> WordNet lemmas ("studies"->"study"). VALUE: real words,
                       readable features, good default for interpretability.
        "stem"      -> Porter stems ("studies"->"studi"). VALUE: smaller vocab;
                       lets us compare stem-vs-lemma as feature engineering.
    remove_stopwords: drop "the/is/and". VALUE: removes high-frequency, low-signal
                      words so TF-IDF weight lands on content words.

    Returns a space-joined string (ready for TfidfVectorizer) or a token list.
    """
    text = basic_clean(text).lower()          # iii. Normalization: lowercase
    text = NON_ALPHA_PATTERN.sub(" ", text)    # iii. drop punctuation + digits
    tokens = word_tokenize(text)               # ii. Tokenization

    # iv. Stop-word removal + drop ultra-short tokens (noise like "a", "i").
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) >= min_token_len]
    else:
        tokens = [t for t in tokens if len(t) >= min_token_len]

    # v. Stemming / Lemmatization: collapse morphological variants to one feature.
    if method == "lemmatize":
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]
    elif method == "stem":
        tokens = [_STEMMER.stem(t) for t in tokens]

    return tokens if return_tokens else " ".join(tokens)



# TRANSFORMER PATH

def transformer_text(text: str) -> str:
    """Text prepared for DistilBERT fine-tuning.
    VALUE: intentionally just `basic_clean` — we KEEP casing and punctuation and
    let the model's own WordPiece tokenizer do the rest. Cleaning more would strip
    the very context the Transformer learns from.
    """
    return basic_clean(text)


# DATAFRAME HELPER — apply the whole pipeline to a dataset in one call

def add_processed_columns(
    df,
    text_col: str = "text",
    *,
    classical_method: str = "lemmatize",
):
    """Add all model-ready text columns to a DataFrame.

    Produces (from ONE shared base clean):
        text_clean          -> basic_clean            (Transformer input)
        text_classical      -> heavy pipeline string  (TF-IDF input)

    VALUE: both models load the SAME dataframe/split and just pick their column,
    guaranteeing they train and are evaluated on identical underlying examples.
    """
    df = df.copy()
    df["text_clean"] = df[text_col].map(transformer_text)
    df["text_classical"] = df[text_col].map(
        lambda t: classical_preprocess(t, method=classical_method)
    )
    return df


# ---------------------------------------------------------------------------
# Quick self-test: `python preprocessing.py`
# lets a teammate eyeball each stage's output on a sample sentence.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = (
        "Cars have been around since the 1900s!! Visit http://example.com. "
        "The studies show that automobiles ARE responsible for emissions."
    )
    print("RAW           :", sample)
    print("basic_clean   :", basic_clean(sample))
    print("word tokens   :", tokenize_words(basic_clean(sample))[:12])
    print("sentences     :", tokenize_sentences(sample))
    print("classical(lem):", classical_preprocess(sample, method="lemmatize"))
    print("classical(stm):", classical_preprocess(sample, method="stem"))
    print("transformer   :", transformer_text(sample))
