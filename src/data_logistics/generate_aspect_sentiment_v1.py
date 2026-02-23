# ============================================================
# FILE    : generate_aspect_sentiment_v1.py
# PURPOSE : Phase-4 Gold Label Generator — Aspect Sentiment
# INDUSTRY: LOGISTICS (HINGLISH + EN)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_aspect_3step.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_aspect_3step.csv"

# ------------------------------------------------------------
# KEYWORD BANKS (ENGLISH + HINGLISH)
# ------------------------------------------------------------

NEGATIVE_KW = [
    # English
    "late", "delay", "delayed", "not delivered", "not received",
    "missing", "damaged", "broken", "lost", "wrong",
    "bad", "poor", "worst", "rude", "unprofessional",
    "angry", "harassed", "no response", "not helpful",
    "very slow", "waiting long", "no update",
    "refund not received", "refund delayed",
    "complaint", "issue", "problem",
    "fake", "incorrect", "misdelivered", "stuck", "pending",

    # Hinglish / Indian usage
    "bahut late", "zyada late", "late ho gaya",
    "deliver nahi hua", "receive nahi hua",
    "parcel nahi mila", "package nahi mila",
    "damage ho gaya", "toota hua",
    "galat", "bekar", "bahut bura",
    "worst service", "bad service",
    "call nahi aaya", "response nahi mila",
    "refund nahi mila", "paise wapas nahi aaye",
    "issue hai", "problem hai", "galat delivery"
]

POSITIVE_KW = [
    # English
    "good", "great", "excellent", "nice", "perfect",
    "smooth", "fast", "quick", "on time", "timely",
    "prompt", "helpful", "polite", "courteous",
    "professional", "satisfied", "happy",
    "thanks", "thank you", "appreciate",
    "well done", "resolved", "issue resolved",
    "support helped", "very good",

    # Hinglish
    "achha", "bahut achha", "badiya",
    "shandaar", "mast", "accha laga",
    "service achhi thi", "time pe mila",
    "on time mila", "problem solve ho gaya",
    "helpful tha", "support ne help ki"
]

NEUTRAL_KW = [
    # English
    "ok", "fine", "average", "normal",
    "no issue", "as expected",
    "information", "query", "status",
    "update", "check", "confirm",
    "details", "process", "procedure",

    # Hinglish
    "theek hai", "normal hai",
    "bas info chahiye",
    "sirf update chahiye",
    "status batao",
    "details chahiye",
    "process samajhna hai"
]

# ------------------------------------------------------------
# ASPECT SENSITIVE ADJUSTMENTS
# ------------------------------------------------------------

NEGATIVE_ASPECT_BIAS = {
    0: True,  # Delay
    1: True,  # Wrong Delivery
    2: True,  # Damage / Lost
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
    t = str(text).lower()

    # ---------------- HARD NEGATIVE ----------------
    if any(k in t for k in NEGATIVE_KW):
        return 0  # Negative

    # ---------------- HARD POSITIVE ----------------
    if any(k in t for k in POSITIVE_KW):
        if POSITIVE_ASPECT_ALLOW.get(aspect, False):
            return 2  # Positive
        else:
            return 1  # Neutral-safe

    # ---------------- GLOBAL SENTIMENT FALLBACK ----------------
    if global_sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
        return 0

    if global_sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
        return 2

    # ---------------- DEFAULT SAFE ----------------
    return 1  # Neutral


# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

required_cols = ["text", "sentiment", "primary_aspect"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["aspect_sentiment"] = df.apply(
    lambda r: assign_aspect_sentiment(
        r["text"],
        int(r["sentiment"]),
        int(r["primary_aspect"])
    ),
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Aspect Sentiment (HINGLISH enhanced) generated successfully")
print("\n--- Aspect Sentiment Distribution ---")
print(df["aspect_sentiment"].value_counts())

print("\n--- By Aspect (sample) ---")
print(pd.crosstab(df["primary_aspect"], df["aspect_sentiment"]))