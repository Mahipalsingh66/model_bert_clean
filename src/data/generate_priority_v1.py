# ============================================================
# FILE    : generate_priority_v1.py
# PURPOSE : Phase-5.5 GOLD GENERATOR — PRIORITY ENGINE
# INPUT   : text, sentiment, primary_aspect, aspect_sentiment, emotion
# OUTPUT  : + priority
# LABELS  :
#   0 = LOW
#   1 = MEDIUM
#   2 = HIGH
#   3 = CRITICAL
# STRATEGY: Enterprise Escalation Logic (Business Safe)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS (CHANGE IF NEEDED)
# ------------------------------------------------------------
INPUT_PATH  = r"D:/model_bert_copy/data/gold/cx_phase5/train.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase6/train_with_priority.csv"

# ------------------------------------------------------------
# CONSTANTS (YOUR TAXONOMY)
# ------------------------------------------------------------

# sentiment: 0=Negative, 1=Neutral, 2=Positive
# emotion  : 0=Calm, 1=Satisfied, 2=Frustrated, 3=Angry, 4=Very Angry

# primary_aspect mapping (as used in your system)
# 0 = Delay
# 1 = Wrong Delivery
# 2 = Damage / Lost
# 3 = Behaviour
# 4 = Support
# 5 = Tracking
# 6 = Pricing
# 7 = Refund

# priority labels
# 0 = LOW
# 1 = MEDIUM
# 2 = HIGH
# 3 = CRITICAL

# ------------------------------------------------------------
# PRIORITY ASSIGNMENT LOGIC (ENTERPRISE RULE ENGINE)
# ------------------------------------------------------------

def assign_priority(sentiment, aspect, aspect_sentiment, emotion):

    # --------------------------------------------------------
    # 🔥 CRITICAL PRIORITY (IMMEDIATE ESCALATION)
    # --------------------------------------------------------

    # Very Angry always critical
    if emotion == 4:
        return 3

    # Angry + financial / loss risk
    if emotion == 3 and aspect in [2, 7]:   # damage/lost, refund
        return 3

    # Strong negative on damage / refund
    if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
        return 3

    # --------------------------------------------------------
    # 🔥 HIGH PRIORITY (ESCALATE TO SUPERVISOR)
    # --------------------------------------------------------

    # Angry complaints
    if emotion == 3:
        return 2

    # Frustrated but operational risk
    if emotion == 2 and aspect in [0, 1, 5]:  # delay, wrong, tracking
        return 2

    # Negative behaviour issues
    if sentiment == 0 and aspect == 3:
        return 2

    # --------------------------------------------------------
    # 🟡 MEDIUM PRIORITY (MONITOR / FOLLOW-UP)
    # --------------------------------------------------------

    # Frustrated normal cases
    if emotion == 2:
        return 1

    # Neutral but operational aspects
    if sentiment == 1 and aspect in [0, 1, 5]:
        return 1

    # Mild negative support
    if sentiment == 0 and aspect == 4:
        return 1

    # --------------------------------------------------------
    # 🟢 LOW PRIORITY (NO ESCALATION)
    # --------------------------------------------------------

    # Satisfied or calm cases
    if emotion in [0, 1]:
        return 0

    # Positive sentiment
    if sentiment == 2:
        return 0

    # Default safe
    return 0

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required = ["sentiment", "primary_aspect", "aspect_sentiment", "emotion"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

priorities = []

for _, row in df.iterrows():
    p = assign_priority(
        int(row["sentiment"]),
        int(row["primary_aspect"]),
        int(row["aspect_sentiment"]),
        int(row["emotion"])
    )
    priorities.append(p)

df["priority"] = priorities

# SAVE
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Priority labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Priority Distribution ---")
print(df["priority"].value_counts())

print("\n--- Priority by Emotion ---")
print(pd.crosstab(df["emotion"], df["priority"]))

print("\n--- Priority by Aspect ---")
print(pd.crosstab(df["primary_aspect"], df["priority"]))
