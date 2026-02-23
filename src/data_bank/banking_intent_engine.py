# ============================================================
# FILE    : banking_intent_engine_v2.py
# PURPOSE : Banking Customer Intent (English + Hinglish)
# LEVEL   : Production / Golden Data
# ============================================================

import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/Banking_data/bank_train_with_aspect_2.csv"
OUTPUT_PATH = r"D:/bert_data/Banking_data/banking_with_intent.csv"

# ------------------------------------------------------------
# KEYWORD BANK — BANKING (EN + HINGLISH)
# ------------------------------------------------------------

DELAY_KW = [
    # English
    "delay", "delayed", "pending", "waiting", "still not",
    "yet to", "no update", "slow process", "taking time",
    "approval pending", "verification pending",
    # Hinglish
    "abhi tak nahi", "kab tak", "pending hai", "late ho gaya",
    "time lag raha", "process slow", "abhi process nahi hua"
]

COMPLAINT_KW = [
    # English
    "problem", "issue", "complaint", "not working", "failed",
    "error", "wrong", "incorrect", "bad", "worst", "pathetic",
    "poor service", "rude", "misbehave", "harassment",
    "charged", "overcharged", "extra charge", "amount deducted",
    "refund", "reversal", "not received", "fraud", "scam",
    "unauthorized", "blocked", "declined",
    # Hinglish
    "problem hai", "issue hai", "kaam nahi kar raha",
    "galat", "bekar", "bahut kharab", "faltu service",
    "rude behaviour", "misbehave kiya",
    "paise kat gaye", "extra paisa", "galat charge",
    "refund nahi mila", "fraud hua", "scam hua",
    "account block", "card block"
]

PRAISE_KW = [
    # English
    "good", "nice", "excellent", "great", "awesome", "amazing",
    "smooth", "happy", "satisfied", "thank you", "thanks",
    "appreciate", "well done", "fantastic", "helpful", "polite",
    # Hinglish
    "bahut acha", "acha service", "badiya", "shandaar",
    "smooth hai", "kaafi acha", "thanku", "dhanyavaad",
    "helpful staff", "polite staff"
]

INQUIRY_KW = [
    # English
    "how", "what", "when", "where", "why",
    "status", "update", "check", "track",
    "procedure", "process", "details", "information",
    "apply", "application", "steps", "charges",
    "interest", "rate", "limit", "eligibility",
    # Hinglish
    "kaise", "kya", "kab", "kyu", "kaun",
    "status batao", "update chahiye",
    "apply kaise kare", "process kya hai",
    "charges kya hai", "interest kitna hai",
    "limit kitni hai"
]

NEGATIVE_TONE_KW = [
    # English
    "angry", "frustrated", "disappointed", "upset",
    "irritated", "fed up",
    # Hinglish
    "gussa", "pareshan", "dukhi", "nirash",
    "irritate", "tang aa gaya"
]

POSITIVE_TONE_KW = [
    # English
    "love", "liked", "enjoy", "impressed",
    "best", "recommended", "trust",
    # Hinglish
    "pasand aaya", "bharosa", "recommend karunga",
    "best bank", "acha laga"
]

# ------------------------------------------------------------
# INTENT ASSIGNMENT (STRICT PRIORITY)
# ------------------------------------------------------------

def assign_banking_intent(text: str, sentiment: int):
    """
    sentiment:
      0 = Negative
      1 = Neutral
      2 = Positive
    """

    if not isinstance(text, str) or not text.strip():
        return 5, "neutral_other"

    t = text.lower()

    # 0 = Delay (explicit waiting / pending)
    if any(k in t for k in DELAY_KW):
        return 0, "delay"

    # 1 = Complaint (strong negative)
    if sentiment == 0 or any(k in t for k in COMPLAINT_KW):
        return 1, "complaint"

    # 2 = Praise (explicit appreciation)
    if sentiment == 2 and any(k in t for k in PRAISE_KW):
        return 2, "praise"

    # 3 = Inquiry (questions / how-to)
    if any(re.search(rf"\b{k}\b", t) for k in INQUIRY_KW):
        return 3, "inquiry"

    # 4 = Negative Other (emotion but no clear ask)
    if sentiment == 0 or any(k in t for k in NEGATIVE_TONE_KW):
        return 4, "negative_other"

    # 6 = Positive Other (positive tone, no praise)
    if sentiment == 2 or any(k in t for k in POSITIVE_TONE_KW):
        return 6, "positive_other"

    # 5 = Neutral Other (fallback)
    return 5, "neutral_other"

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="utf-8")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace("\ufeff", "", regex=False)
)

required_cols = ["text", "sentiment"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

intents, intent_flags = [], []

for _, row in df.iterrows():
    intent, flag = assign_banking_intent(
        row["text"],
        int(row["sentiment"])
    )
    intents.append(intent)
    intent_flags.append(flag)

df["customer_intent"] = intents
df["intent_flag"] = intent_flags

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Banking intent labels (EN + Hinglish) generated")
print("Output:", OUTPUT_PATH)
print("\nIntent Distribution:")
print(df["customer_intent"].value_counts())

# ------------------------------------------------------------
# QUICK SANITY TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("paise kat gaye refund nahi mila", 0),
        ("how to apply credit card", 1),
        ("bahut acha service mila branch me", 2),
        ("loan approval abhi tak pending hai", 1),
        ("mobile app smooth hai", 2),
        ("charges kya hai zero balance account ke", 1)
    ]
    for s, sent in samples:
        print(s, "->", assign_banking_intent(s, sent))