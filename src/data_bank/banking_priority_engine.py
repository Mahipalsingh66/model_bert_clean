# ============================================================
# FILE    : banking_priority_engine.py
# PURPOSE : Banking Priority Gold Label Generator
#
# INPUT   : text, sentiment, primary_aspect, aspect_sentiment, emotion
# OUTPUT  : + priority
#
# PRIORITY LABELS:
#   0 = LOW
#   1 = MEDIUM
#   2 = HIGH
#   3 = CRITICAL
#
# EMOTION LABELS (LOCKED):
#   0 = Angry
#   1 = Frustrated
#   2 = Fear
#   3 = Sad
#   4 = Neutral
#   5 = Happy
#   6 = Satisfied
#
# INDUSTRY : BANKING
# STRATEGY : Enterprise Escalation Logic (Risk-first)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS (UPDATE IF REQUIRED)
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/Banking_data/bank_train_with_aspect+aspect sentiment+emotions.csv"
OUTPUT_PATH = r"D:/bert_data/Banking_data/bank_train_with_aspect+aspect sentiment+emotions+priority.csv"

# ------------------------------------------------------------
# BANKING ASPECT GROUPS
# ------------------------------------------------------------

# High financial / legal risk aspects
CRITICAL_ASPECTS = {
    0,  # Transaction_Issue
    1,  # Charges
    2,  # Loan_Credit
    7,  # ATM
    8,  # Minimum_Balance
    10, # Interest_Rate
    12  # Security / Fraud
}

# Operational pain but not legal
OPERATIONAL_ASPECTS = {
    3,  # Mobile_App
    6,  # Customer_Service
    9   # Branch
}

# Experience-only
EXPERIENCE_ASPECTS = {
    4,  # Staff_Negative
    5,  # Appreciation
    11, # Offers
    13  # General
}

# ------------------------------------------------------------
# PRIORITY ASSIGNMENT LOGIC
# ------------------------------------------------------------

def assign_priority(sentiment, aspect, aspect_sentiment, emotion):
    """
    sentiment:
        0 = Negative, 1 = Neutral, 2 = Positive
    aspect_sentiment:
        0 = Negative, 1 = Neutral, 2 = Positive
    emotion:
        0 = Angry
        1 = Frustrated
        2 = Fear
        3 = Sad
        4 = Neutral
        5 = Happy
        6 = Satisfied
    """

    # --------------------------------------------------------
    # 🔴 CRITICAL (Immediate Escalation)
    # --------------------------------------------------------

    # Fear (fraud / security) is ALWAYS critical
    if emotion == 2:
        return 3

    # Angry + financial / legal aspect
    if emotion == 0 and aspect in CRITICAL_ASPECTS:
        return 3

    # Strong negative on security or transaction
    if sentiment == 0 and aspect in CRITICAL_ASPECTS and aspect_sentiment == 0:
        return 3

    # --------------------------------------------------------
    # 🟠 HIGH (Supervisor Attention)
    # --------------------------------------------------------

    # Angry but non-financial
    if emotion == 0:
        return 2

    # Frustrated + operational or financial
    if emotion == 1 and aspect in (CRITICAL_ASPECTS | OPERATIONAL_ASPECTS):
        return 2

    # Negative staff behaviour
    if sentiment == 0 and aspect == 4:
        return 2

    # --------------------------------------------------------
    # 🟡 MEDIUM (Follow-up Required)
    # --------------------------------------------------------

    # Frustrated normal cases
    if emotion == 1:
        return 1

    # Sad customers
    if emotion == 3:
        return 1

    # Neutral sentiment but operational issues
    if sentiment == 1 and aspect in OPERATIONAL_ASPECTS:
        return 1

    # --------------------------------------------------------
    # 🟢 LOW (No Escalation)
    # --------------------------------------------------------

    # Happy / satisfied customers
    if emotion in {5, 6}:
        return 0

    # Positive sentiment
    if sentiment == 2:
        return 0

    # Default safe
    return 0

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(
    INPUT_PATH,
    encoding="latin1",
    engine="python",
    on_bad_lines="warn"
)

df.columns = df.columns.str.strip().str.lower()

required_cols = ["sentiment", "primary_aspect", "aspect_sentiment", "emotion"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols)
df[required_cols] = df[required_cols].astype(int)

df["priority"] = df.apply(
    lambda x: assign_priority(
        x["sentiment"],
        x["primary_aspect"],
        x["aspect_sentiment"],
        x["emotion"]
    ),
    axis=1
)

# ------------------------------------------------------------
# SAVE OUTPUT
# ------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Banking priority labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Priority Distribution ---")
print(df["priority"].value_counts())

print("\n--- Priority by Emotion ---")
print(pd.crosstab(df["emotion"], df["priority"]))

print("\n--- Priority by Aspect ---")
print(pd.crosstab(df["primary_aspect"], df["priority"]))