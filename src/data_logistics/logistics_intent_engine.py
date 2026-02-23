# ============================================================
# FILE    : logistics_intent_engine.py
# PURPOSE : Logistics Domain — Customer Intent Generator (7-class)
# STRATEGY: Banking-aligned intent taxonomy
# MODE    : OVERWRITE existing CSV
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS (OVERWRITE SAME FILE)
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_intent_1step.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_intent_1step.csv"

# ------------------------------------------------------------
# KEYWORD BANK (LOGISTICS – EN + HINGLISH)
# ------------------------------------------------------------

DELAY_KW = [
    "delay", "late", "delayed", "not delivered", "delivery pending",
    "still waiting", "no delivery", "delivery slow",
    "bahut late", "abhi tak nahi aaya", "deliver nahi hua",
    "late delivery", "delivery delay"
]

COMPLAINT_KW = [
    "complaint", "issue", "problem", "bad service", "worst service",
    "rude", "misbehave", "fraud", "scam", "cheat",
    "missing", "lost", "damaged", "broken", "refund",
    "galat", "bekar", "problem hai", "issue hai",
    "wrong delivery", "package missing"
]

PRAISE_KW = [
    "good", "nice", "excellent", "great", "perfect",
    "happy", "satisfied", "thanks", "thank you",
    "awesome", "smooth", "fast delivery",
    "achha", "bahut achha", "badiya", "shandaar",
    "on time", "timely delivery"
]

INQUIRY_KW = [
    "where", "when", "status", "track", "tracking",
    "update", "why", "how",
    "kab aayega", "kaha hai", "status kya hai",
    "tracking number", "awb", "consignment"
]

NEGATIVE_OTHER_KW = [
    "disappointed", "upset", "unhappy",
    "not satisfied", "poor experience",
    "expectation nahi mila", "dukhi", "naraz"
]

POSITIVE_OTHER_KW = [
    "keep it up", "good job", "well done",
    "impressed", "loved it",
    "service achhi lagi", "experience acha raha"
]

# ------------------------------------------------------------
# INTENT ASSIGNMENT
# ------------------------------------------------------------

def assign_logistics_intent(text, sentiment):
    if not isinstance(text, str) or not text.strip():
        return 5  # Neutral_Other

    t = text.lower()

    # 0 = Delay
    if any(k in t for k in DELAY_KW):
        return 0

    # 1 = Complaint
    if sentiment == 0 or any(k in t for k in COMPLAINT_KW):
        return 1

    # 3 = Inquiry
    if any(k in t for k in INQUIRY_KW):
        return 3

    # 2 = Praise
    if sentiment == 2 or any(k in t for k in PRAISE_KW):
        return 2

    # 4 = Negative_Other
    if any(k in t for k in NEGATIVE_OTHER_KW):
        return 4

    # 6 = Positive_Other
    if any(k in t for k in POSITIVE_OTHER_KW):
        return 6

    # 5 = Neutral_Other (SAFE FALLBACK)
    return 5

# ------------------------------------------------------------
# APPLY (OVERWRITE MODE)
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    encoding="latin1",
    sep=",",
    engine="python",
    on_bad_lines="skip"
)

if "text" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("Required columns missing: text / sentiment")

df["customer_intent"] = df.apply(
    lambda x: assign_logistics_intent(x["text"], int(x["sentiment"])),
    axis=1
)

# OVERWRITE SAME FILE (NO NEW DATASET CREATED)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Logistics intent (7-class) generated successfully")
print("Output:", OUTPUT_PATH)
print("\n--- Intent Distribution ---")
print(df["customer_intent"].value_counts())    