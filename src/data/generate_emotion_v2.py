# ============================================================
# FILE    : generate_emotion_v2.py
# PURPOSE : Phase-5 GOLD GENERATOR — 5-Class Emotion (MAX COVERAGE)
# LABELS  :
#   0 = Calm
#   1 = Satisfied
#   2 = Frustrated
#   3 = Angry
#   4 = Very Angry
# STRATEGY: Aggressive recall for Angry / Very Angry + High precision
# NOTE    : Enterprise keyword bank (global CX vocabulary)
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
INPUT_PATH  = r"D:/model_bert_copy/data/gold/cx_phase4/val.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase5/val_with_emotion.csv"

# ------------------------------------------------------------
# GLOBAL KEYWORD BANKS (MAXIMUM COVERAGE)
# Sources: CX datasets, call-center logs, banking, logistics, SaaS, telecom
# ------------------------------------------------------------

VERY_ANGRY_KW = [
    # Legal / Threat / Extreme
    "fraud", "cheated", "scam", "fake", "forged", "illegal",
    "legal action", "court", "consumer court", "police complaint",
    "file case", "lawsuit", "sue", "lawyer", "legal notice",

    # Extreme anger
    "worst", "pathetic", "disgusting", "horrible", "terrible",
    "unacceptable", "shameful", "shocking", "ridiculous",
    "never again", "boycott", "blacklist",

    # Threat language
    "i will complain", "i will report", "i will escalate",
    "i will post", "i will go public", "i will cancel",
    "i am done", "fed up", "enough is enough",

    # Abuse / harassment
    "harassment", "abuse", "threatened", "mentally tortured",
    "traumatized", "insulted", "humiliated",

    # Refund + extreme
    "money stolen", "refund fraud", "refund cheated", "amount not returned",
    "payment stuck", "charged twice", "double charged",
]

ANGRY_KW = [
    # Strong complaint
    "angry", "furious", "very upset", "irritated", "annoyed",
    "frustrating", "not acceptable", "complaint", "complaining",
    "bad service", "poor service", "worst service",

    # Delivery / ops anger
    "still not delivered", "not delivered yet", "delay again",
    "wrong delivery", "misdelivered", "lost package", "missing item",
    "damaged product", "broken item",

    # Support anger
    "no response", "no reply", "no callback", "nobody helped",
    "support useless", "agent rude", "executive rude",

    # Refund anger
    "refund pending", "refund delayed", "not refunded",
    "money not received", "amount pending",

    # Escalation language
    "escalate", "supervisor", "manager", "senior team",
]

FRUSTRATED_KW = [
    # Mild negative
    "late", "delay", "delayed", "waiting", "still waiting",
    "pending", "in progress", "not updated", "no update",
    "no information", "status unknown",

    # Confusion / concern
    "confused", "not clear", "unclear", "concern", "worried",
    "need help", "please check", "please update",
    "follow up", "remind", "again asking",

    # Service issues
    "problem", "issue", "error", "mistake", "incorrect",
    "wrong status", "wrong update",
]

SATISFIED_KW = [
    # Praise
    "good", "great", "nice", "excellent", "perfect", "amazing",
    "awesome", "fantastic", "brilliant", "wonderful",

    # Gratitude
    "thanks", "thank you", "thx", "much appreciated",
    "grateful", "thankful",

    # Resolution
    "resolved", "fixed", "sorted", "completed", "done",
    "issue solved", "problem solved",

    # Service praise
    "quick service", "fast service", "on time", "timely",
    "support helped", "agent helpful", "very helpful",

    # Satisfaction
    "happy", "satisfied", "very satisfied", "pleased",
]

CALM_KW = [
    # Neutral / metadata
    "ok", "fine", "normal", "average", "as expected",
    "just information", "query", "enquiry", "status",
    "checking", "confirming", "details", "process",
    "procedure", "policy", "terms",
]

# ------------------------------------------------------------
# EMOTION ASSIGNMENT LOGIC (ENTERPRISE ORDERED)
# ------------------------------------------------------------

def assign_emotion(text, sentiment, aspect, aspect_sentiment):
    t = text.lower()

    # ---------------- VERY ANGRY (CRITICAL FIRST) ----------------
    if any(k in t for k in VERY_ANGRY_KW):
        return 4

    # Extreme operational + negative
    if sentiment == 0 and aspect in [2, 7] and aspect_sentiment == 0:
        return 4

    # ---------------- ANGRY ----------------
    if any(k in t for k in ANGRY_KW):
        return 3

    if sentiment == 0 and aspect_sentiment == 0:
        return 3

    # ---------------- FRUSTRATED ----------------
    if any(k in t for k in FRUSTRATED_KW):
        return 2

    if sentiment in [0, 1] and aspect in [0, 1, 5]:  # delay, wrong, tracking
        return 2

    # ---------------- SATISFIED ----------------
    if any(k in t for k in SATISFIED_KW):
        return 1

    if sentiment == 2 and aspect_sentiment == 2:
        return 1

    # ---------------- CALM ----------------
    if any(k in t for k in CALM_KW):
        return 0

    # Default fallback by sentiment
    if sentiment == 2:
        return 1
    if sentiment == 0:
        return 2

    return 0

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required = ["text", "sentiment", "primary_aspect", "aspect_sentiment"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

emotions = []

for _, row in df.iterrows():
    emo = assign_emotion(
        row["text"],
        int(row["sentiment"]),
        int(row["primary_aspect"]),
        int(row["aspect_sentiment"])
    )
    emotions.append(emo)

df["emotion"] = emotions

# SAVE

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Emotion labels generated successfully (v2 — MAX COVERAGE)")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Emotion Distribution ---")
print(df["emotion"].value_counts())

print("\n--- Emotion by Sentiment ---")
print(pd.crosstab(df["sentiment"], df["emotion"]))

print("\n--- Emotion by Aspect ---")
print(pd.crosstab(df["primary_aspect"], df["emotion"]))
