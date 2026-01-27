# ============================================================
# FILE    : generate_aspect_sentiment_v1.py
# PURPOSE : Phase-4 Gold Label Generator — Aspect Sentiment
# INPUT   : text, sentiment, primary_aspect
# OUTPUT  : text, sentiment, primary_aspect, aspect_sentiment
# LABELS  : 0 = Negative, 1 = Neutral, 2 = Positive
# STRATEGY: Accuracy Optimized (enterprise safe)
# ============================================================

import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/cx_phase3/val.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase4/val_with_aspect_sentiment.csv"

# ------------------------------------------------------------
# KEYWORD BANKS
# ------------------------------------------------------------

NEGATIVE_KW = [
    "late", "delay", "delayed", "not delivered", "not received", "missing",
    "damaged", "broken", "lost", "wrong", "bad", "poor", "worst",
    "rude", "unprofessional", "angry", "harassed", "no response",
    "not helpful", "very slow", "waiting long", "no update",
    "refund not received", "refund delayed", "complaint", "issue", "problem",
    "fake", "incorrect", "misdelivered", "stuck", "pending"
]

POSITIVE_KW = [
    "good", "great", "excellent", "nice", "perfect", "smooth",
    "fast", "quick", "on time", "timely", "prompt", "helpful",
    "polite", "courteous", "professional", "satisfied", "happy",
    "thanks", "thank you", "appreciate", "well done", "resolved",
    "support helped", "issue resolved", "very good"
]

NEUTRAL_KW = [
    "ok", "fine", "average", "normal", "no issue", "as expected",
    "information", "query", "status", "update", "check",
    "confirm", "details", "process", "procedure"
]

# ------------------------------------------------------------
# ASPECT SENSITIVE ADJUSTMENTS
# ------------------------------------------------------------

NEGATIVE_ASPECT_BIAS = {
    0: True,  # Delay → usually negative
    1: True,  # Wrong
    2: True,  # Damage/Lost
    7: True   # Refund
}

POSITIVE_ASPECT_ALLOW = {
    4: True,  # Support
    3: True   # Behaviour
}

# ------------------------------------------------------------
# ASSIGN ASPECT SENTIMENT
# ------------------------------------------------------------

def assign_aspect_sentiment(text, global_sentiment, aspect):
    t = text.lower()

    # ---------------- HARD NEGATIVE ----------------
    if any(k in t for k in NEGATIVE_KW):
        return 0  # Negative

    # ---------------- HARD POSITIVE ----------------
    if any(k in t for k in POSITIVE_KW):
        # Positive allowed only for some aspects
        if POSITIVE_ASPECT_ALLOW.get(aspect, False):
            return 2
        # For operational aspects, positive becomes neutral-safe
        else:
            return 1

    # ---------------- GLOBAL SENTIMENT FALLBACK ----------------
    # sentiment column: 0=Neg, 1=Neu, 2=Pos

    # Negative global & operational aspect
    if global_sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
        return 0

    # Positive global & support / behaviour
    if global_sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
        return 2

    # ---------------- DEFAULT SAFE ----------------
    return 1  # Neutral


# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_cols = ["text", "sentiment", "primary_aspect"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

labels = []

for _, row in df.iterrows():
    asp_sent = assign_aspect_sentiment(
        row["text"],
        int(row["sentiment"]),
        int(row["primary_aspect"])
    )
    labels.append(asp_sent)

df["aspect_sentiment"] = labels

# SAVE

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Aspect-Sentiment labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect-Sentiment Distribution ---")
print(df["aspect_sentiment"].value_counts())

print("\n--- By Aspect (sample) ---")
print(pd.crosstab(df["primary_aspect"], df["aspect_sentiment"]))
