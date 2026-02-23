# ============================================================
# FILE    : banking_aspect_sentiment_engine_v1.py
# PURPOSE : Banking Aspect Sentiment Gold Label Generator
#
# INPUT   : text, sentiment, primary_aspect
# OUTPUT  : + aspect_sentiment
#
# LABELS  :
#   0 = Negative
#   1 = Neutral
#   2 = Positive
#
# STRATEGY:
# - Keyword driven (EN + Hinglish)
# - Aspect-aware bias
# - Sentiment-safe fallback
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/Banking_data/bank_train_with_aspect_2.csv"
OUTPUT_PATH = r"D:/bert_data/Banking_data/bank_train_with_aspect+aspect sentiment.csv"

# ------------------------------------------------------------
# KEYWORD BANKS (BANKING – EN + HINGLISH)
# ------------------------------------------------------------

NEGATIVE_KW = [
    # English
    "failed", "error", "issue", "problem", "complaint",
    "bad", "poor", "worst", "pathetic",
    "not working", "not received", "pending",
    "delay", "late", "stuck", "declined",
    "charged", "overcharged", "wrong charge",
    "refund not received", "reversal pending",
    "fraud", "scam", "unauthorized",
    "rude", "misbehave", "harassment",
    # Hinglish
    "problem hai", "issue hai", "kaam nahi kar raha",
    "galat", "bekar", "bahut kharab",
    "paise kat gaye", "extra paisa kata",
    "refund nahi mila", "pending hai",
    "fraud hua", "scam ho gaya",
    "rude staff", "staff badtameez",
    "delay ho gaya"
]

POSITIVE_KW = [
    # English
    "good", "great", "excellent", "nice",
    "smooth", "fast", "quick",
    "helpful", "polite", "professional",
    "satisfied", "happy",
    "thank you", "thanks", "appreciate",
    "resolved", "issue resolved",
    # Hinglish
    "acha", "bahut acha", "badiya",
    "smooth hai", "fast service",
    "helpful staff", "polite staff",
    "satisfied hoon", "khush hoon",
    "thank you bank"
]

NEUTRAL_KW = [
    # English
    "ok", "fine", "average",
    "information", "details",
    "process", "procedure",
    "query", "status", "update",
    # Hinglish
    "jaankari", "process kya hai",
    "details chahiye", "status batao",
    "check karna hai"
]

# ------------------------------------------------------------
# ASPECT GROUPING (BANKING)
# ------------------------------------------------------------

# Operational / Risk-heavy aspects → mostly negative by nature
NEGATIVE_ASPECT_BIAS = {
    0: True,   # Transaction_Issue
    1: True,   # Charges
    2: True,   # Loan_Credit
    7: True,   # ATM
    8: True,   # Minimum_Balance
    10: True,  # Interest_Rate
    12: True   # Security
}

# Aspects where positive sentiment is valid
POSITIVE_ASPECT_ALLOW = {
    5: True,   # Appreciation
    6: True,   # Customer_Service
    9: True,   # Branch
    3: True    # Mobile_App
}

# ------------------------------------------------------------
# ASSIGN ASPECT SENTIMENT
# ------------------------------------------------------------

def assign_banking_aspect_sentiment(text, global_sentiment, aspect):
    """
    global_sentiment:
        0 = Negative
        1 = Neutral
        2 = Positive
    """

    if not isinstance(text, str) or not text.strip():
        return 1  # Neutral safe

    t = text.lower()

    # ---------------- HARD NEGATIVE ----------------
    if any(k in t for k in NEGATIVE_KW):
        return 0

    # ---------------- HARD POSITIVE ----------------
    if any(k in t for k in POSITIVE_KW):
        if POSITIVE_ASPECT_ALLOW.get(aspect, False):
            return 2
        else:
            # Operational aspect cannot be truly positive
            return 1

    # ---------------- NEUTRAL SIGNAL ----------------
    if any(k in t for k in NEUTRAL_KW):
        return 1

    # ---------------- GLOBAL SENTIMENT FALLBACK ----------------

    # Negative sentiment + negative-biased aspect
    if global_sentiment == 0 and NEGATIVE_ASPECT_BIAS.get(aspect, False):
        return 0

    # Positive sentiment + allowed aspect
    if global_sentiment == 2 and POSITIVE_ASPECT_ALLOW.get(aspect, False):
        return 2

    # ---------------- DEFAULT SAFE ----------------
    return 1


# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    encoding="latin1",
    sep=",",
    engine="python",
    on_bad_lines="warn"
)

df.columns = df.columns.str.strip().str.lower()

required_cols = ["text", "sentiment", "primary_aspect"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

# Clean sentiment column
df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
df = df.dropna(subset=["sentiment"])
df["sentiment"] = df["sentiment"].astype(int)

labels = []

for _, row in df.iterrows():
    asp_sent = assign_banking_aspect_sentiment(
        row["text"],
        int(row["sentiment"]),
        int(row["primary_aspect"])
    )
    labels.append(asp_sent)

df["aspect_sentiment"] = labels

# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Banking aspect-sentiment labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect Sentiment Distribution ---")
print(df["aspect_sentiment"].value_counts())

print("\n--- By Aspect (cross-tab) ---")
print(pd.crosstab(df["primary_aspect"], df["aspect_sentiment"]))

# ------------------------------------------------------------
# QUICK TEST
# ------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        ("paise kat gaye refund nahi mila", 0, 0),
        ("branch staff bahut helpful tha", 2, 9),
        ("mobile app smooth hai", 2, 3),
        ("interest rate bahut zyada hai", 0, 10),
        ("customer care ne issue resolve kar diya", 2, 6),
        ("atm paisa nahi diya", 0, 7)
    ]

    for s, sent, asp in samples:
        print(s, "->", assign_banking_aspect_sentiment(s, sent, asp))