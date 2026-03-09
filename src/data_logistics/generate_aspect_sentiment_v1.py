# # ============================================================
# # FILE    : generate_aspect_sentiment_v1.py
# # PURPOSE : Phase-4 Gold Label Generator — Aspect Sentiment
# # INDUSTRY: LOGISTICS (HINGLISH + EN)
# # ============================================================

# import pandas as pd

# # ------------------------------------------------------------
# # PATHS
# # ------------------------------------------------------------

# INPUT_PATH  = r"D:/bert_data/logistics data/train+intent+aspects.csv"
# OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment.csv"

# # ------------------------------------------------------------
# # KEYWORD BANKS (ENGLISH + HINGLISH)
# # ------------------------------------------------------------

# NEGATIVE_KW = [
#     # English
#     "late", "delay", "delayed", "not delivered", "not received",
#     "missing", "damaged", "broken", "lost", "wrong",
#     "bad", "poor", "worst", "rude", "unprofessional",
#     "angry", "harassed", "no response", "not helpful",
#     "very slow", "waiting long", "no update",
#     "refund not received", "refund delayed",
#     "complaint", "issue", "problem",
#     "fake", "incorrect", "misdelivered", "stuck", "pending",

#     # Hinglish / Indian usage
#     "bahut late", "zyada late", "late ho gaya",
#     "deliver nahi hua", "receive nahi hua",
#     "parcel nahi mila", "package nahi mila",
#     "damage ho gaya", "toota hua",
#     "galat", "bekar", "bahut bura",
#     "worst service", "bad service",
#     "call nahi aaya", "response nahi mila",
#     "refund nahi mila", "paise wapas nahi aaye",
#     "issue hai", "problem hai", "galat delivery"
# ]

# POSITIVE_KW = [
#     # English
#     "good", "great", "excellent", "nice", "perfect",
#     "smooth", "fast", "quick", "on time", "timely",
#     "prompt", "helpful", "polite", "courteous",
#     "professional", "satisfied", "happy",
#     "thanks", "thank you", "appreciate",
#     "well done", "resolved", "issue resolved",
#     "support helped", "very good",

#     # Hinglish
#     "achha", "bahut achha", "badiya",
#     "shandaar", "mast", "accha laga",
#     "service achhi thi", "time pe mila",
#     "on time mila", "problem solve ho gaya",
#     "helpful tha", "support ne help ki"
# ]

# NEUTRAL_KW = [
#     # English
#     "ok", "fine", "average", "normal",
#     "no issue", "as expected",
#     "information", "query", "status",
#     "update", "check", "confirm",
#     "details", "process", "procedure",

#     # Hinglish
#     "theek hai", "normal hai",
#     "bas info chahiye",
#     "sirf update chahiye",
#     "status batao",
#     "details chahiye",
#     "process samajhna hai"
# ]

# # ------------------------------------------------------------
# # ASPECT SENSITIVE ADJUSTMENTS
# # ------------------------------------------------------------

# NEGATIVE_ASPECT_BIAS = {
#     0: True,  # Delay
#     1: True,  # Wrong Delivery
#     2: True,  # Damage / Lost
#     7: True   # Refund
# }

# POSITIVE_ASPECT_ALLOW = {
#     4: True,  # Support
#     3: True   # Behaviour
# }

# # ------------------------------------------------------------
# # ASSIGN ASPECT SENTIMENT
# # ------------------------------------------------------------

# def assign_aspect_sentiment(text, global_sentiment, aspect):
#     t = str(text).lower()

#     # ---------------- HARD NEGATIVE ----------------
#     if any(k in t for k in NEGATIVE_KW):
#         return 0  # Negative

#     # ---------------- HARD POSITIVE ----------------
#     if any(k in t for k in POSITIVE_KW):
#         if POSITIVE_ASPECT_ALLOW.get(aspect, False):
#             return 2  # Positive
#         else:
#             return 1  # Neutral-safe

#     # ---------------- GLOBAL SENTIMENT FALLBACK ----------------
#     if global_sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
#         return 0

#     if global_sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
#         return 2

#     # ---------------- DEFAULT SAFE ----------------
#     return 1  # Neutral


# # ------------------------------------------------------------
# # APPLY TO DATASET
# # ------------------------------------------------------------

# df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

# required_cols = ["text", "sentiment", "primary_aspect"]
# for c in required_cols:
#     if c not in df.columns:
#         raise ValueError(f"Missing column: {c}")

# df["aspect_sentiment"] = df.apply(
#     lambda r: assign_aspect_sentiment(
#         r["text"],
#         int(r["sentiment"]),
#         int(r["primary_aspect"])
#     ),
#     axis=1
# )

# df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

# print("✅ Aspect Sentiment (HINGLISH enhanced) generated successfully")
# print("\n--- Aspect Sentiment Distribution ---")
# print(df["aspect_sentiment"].value_counts())

# print("\n--- By Aspect (sample) ---")
# print(pd.crosstab(df["primary_aspect"], df["aspect_sentiment"]))

# ============================================================
# FILE    : generate_aspect_sentiment_v2.py
# PURPOSE : Enterprise Gold Label Generator — Aspect Sentiment
# INDUSTRY: LOGISTICS (Aspect v4 aligned)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/logistics data/train+intent+aspect_v4.csv"
OUTPUT_PATH = r"D:/bert_data/logistics data/train+intent+aspects+aspects_sentiment_v4.csv"

# ------------------------------------------------------------
# SENTIMENT IDS
# ------------------------------------------------------------
"""
0 = Negative
1 = Neutral
2 = Positive
"""

# ------------------------------------------------------------
# KEYWORD BANKS (HIGH PRECISION ONLY)
# ------------------------------------------------------------

NEGATIVE_KW = [
    "late", "delay", "delayed",
    "not delivered", "not received",
    "missing", "lost", "damaged", "broken",
    "wrong delivery", "misdelivered",
    "rude", "unprofessional",
    "no response", "refund not received",
    "refund pending", "fake status",

    # Hinglish
    "bahut late", "deliver nahi hua",
    "parcel nahi mila", "toota hua",
    "galat delivery", "refund nahi mila",
    "response nahi mila", "badtameez"
]

POSITIVE_KW = [
    "good service", "great service",
    "excellent", "very helpful",
    "polite", "professional",
    "resolved", "issue resolved",
    "satisfied", "happy",
    "thanks", "thank you", "appreciate",

    # Hinglish
    "achhi service", "bahut achha",
    "support ne help ki",
    "problem solve ho gaya",
    "shukriya", "dhanyavaad"
]

# ------------------------------------------------------------
# ASPECT POLICIES (LOCKED)
# ------------------------------------------------------------

NEGATIVE_ONLY_ASPECTS = {1, 2, 3}
NEGATIVE_NEUTRAL_ASPECTS = {0, 6, 7, 8, 9, 10, 11, 12}
POSITIVE_ALLOWED_ASPECTS = {4, 5}
APPRECIATION_ASPECT = 13

# ------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------

def assign_aspect_sentiment(text, aspect, global_sentiment):
    t = str(text).lower()

    # 🔒 HARD RULE 1: Appreciation is ALWAYS Positive
    if aspect == APPRECIATION_ASPECT:
        return 2

    # 🔒 HARD RULE 2: Explicit negative language
    if any(k in t for k in NEGATIVE_KW):
        return 0

    # 🔒 HARD RULE 3: Explicit positive language
    if any(k in t for k in POSITIVE_KW):
        if aspect in POSITIVE_ALLOWED_ASPECTS:
            return 2
        else:
            return 1  # neutral-safe for non-positive aspects

    # 🔒 HARD RULE 4: Aspect semantic constraints
    if aspect in NEGATIVE_ONLY_ASPECTS:
        return 0

    if aspect in NEGATIVE_NEUTRAL_ASPECTS:
        return 0 if global_sentiment == 0 else 1

    if aspect in POSITIVE_ALLOWED_ASPECTS:
        if global_sentiment == 2:
            return 2
        elif global_sentiment == 0:
            return 0
        else:
            return 1

    # 🔒 SAFE DEFAULT
    return 1  # Neutral

# ------------------------------------------------------------
# APPLY
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

required_cols = ["text", "sentiment", "primary_aspect"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df["aspect_sentiment"] = df.apply(
    lambda r: assign_aspect_sentiment(
        r["text"],
        int(r["primary_aspect"]),
        int(r["sentiment"])
    ),
    axis=1
)

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Aspect Sentiment v2 generated successfully")
print("\n--- Distribution ---")
print(df["aspect_sentiment"].value_counts())

print("\n--- By Aspect ---")
print(pd.crosstab(df["primary_aspect"], df["aspect_sentiment"]))