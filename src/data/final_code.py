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
# ============================================================
# FILE    : generate_aspect_v3.py
# PURPOSE : FINAL Enterprise Aspect Rule Engine (Production)
# INPUT   : text,sentiment,customer_intent
# OUTPUT  : text,sentiment,customer_intent,primary_aspect,aspect_flag
#
# TAXONOMY:
#   0 = Delay
#   1 = Wrong Delivery
#   2 = Damage / Lost
#   3 = Behaviour
#   4 = Support
#   5 = Tracking
#   6 = Pricing
#   7 = Refund
# ============================================================

import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/v2.3_multitask/val_multi_v1.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase3/val_with_aspect_v3.csv"

# ------------------------------------------------------------
# KEYWORD BANK (MAX COVERAGE, ENTERPRISE GRADE)
# ------------------------------------------------------------

DELAY_KW = [
    "late", "delay", "delayed", "one day late", "two days late", "three days late",
    "not on time", "not delivered on time", "delivery pending", "pending delivery",
    "pending for days", "still waiting", "waiting for delivery", "no delivery yet",
    "after due date", "delivery not attempted", "attempted late",
    "shipment delayed", "out for delivery but not delivered",
    "postponed", "rescheduled delivery", "delivery attempt failed",
    "delivery failed", "delivery postponed", "delivery rescheduled",
    "not delivered yet", "delivery slow", "delivery very late",
    "took long time", "taking too long"
]

WRONG_KW = [
    "wrong address", "wrong location", "wrong place", "wrong area",
    "delivered to wrong", "delivered somewhere else", "delivered other place",
    "someone else received", "unknown person received",
    "maid received", "neighbor received", "security received",
    "not my order", "not my package", "not my parcel",
    "wrong pin code", "wrong building", "wrong flat",
    "misdelivered", "incorrect delivery", "wrong house", "wrong society",
    "wrong recipient", "delivered to another person"
]

DAMAGE_KW = [
    "damaged", "damage", "broken", "cracked", "leak", "leaked",
    "torn", "open packet", "opened parcel", "empty box",
    "missing item", "item missing", "parcel missing", "content missing",
    "lost", "not received", "did not receive", "never received",
    "package lost", "shipment lost", "product missing",
    "stolen", "tampered", "seal broken", "box damaged",
    "package empty", "nothing inside", "product damaged"
]

BEHAVIOUR_KW = [
    "rude", "misbehave", "misbehavior", "behaviour", "behavior",
    "bad attitude", "unprofessional", "shouted", "argued",
    "angry delivery", "delivery boy rude", "staff rude",
    "threatened", "abused", "not polite", "very arrogant",
    "impolite", "bad manners", "harassment", "unacceptable behaviour",
    "did not contact", "not contacted", "no call", "did not call",
    "no intimation", "did not inform", "no message",
    "delivery boy not cooperative", "delivery agent rude",
    "not responsive delivery boy"
]

TRACKING_KW = [
    "track", "tracking", "tracking not updated", "status not updated",
    "no update", "no tracking", "awb", "waybill", "consignment",
    "where is my order", "where is my parcel", "where is shipment",
    "no status", "status wrong", "location not updated",
    "showing delivered but not received", "fake delivery status",
    "system shows delivered", "tracking wrong", "tracking error",
    "not reflecting", "tracking problem", "tracking issue",
    "no movement", "stuck in transit"
]

PRICING_KW = [
    "charged", "extra charge", "overcharged", "wrong amount",
    "high charge", "pricing issue", "price mismatch",
    "fee", "service charge", "cod charge", "cash on delivery charge",
    "billing issue", "payment issue", "amount deducted",
    "double charged", "refund amount wrong",
    "cost issue", "pricing problem", "rate issue"
]

REFUND_KW = [
    "refund", "refunded", "money back", "amount not credited",
    "return", "returned", "return process", "return completed",
    "refund pending", "refund delayed", "not refunded",
    "refund not received", "replacement", "exchange",
    "cancelled order refund", "reverse pickup",
    "pickup completed but refund not received",
    "waiting for refund", "refund issue", "refund problem"
]

SUPPORT_KW = [
    "call", "called", "contact", "support", "customer care",
    "helpline", "complaint", "registered complaint",
    "no response", "not responding", "no reply", "waiting for response",
    "ticket", "case id", "grievance", "escalation", "follow up",
    "no callback", "service issue", "helpdesk", "query",
    "customer service", "service center", "support team"
]

# ------------------------------------------------------------
# ADVANCED SOFT PATTERNS (RECOVER FALLBACK DATA)
# ------------------------------------------------------------

SOFT_DELAY_PATTERNS = [
    "delivery very slow", "still not received", "long delay", "taking long",
    "delivery taking time", "delay in delivery"
]

SOFT_TRACKING_PATTERNS = [
    "no update from courier", "no update yet", "waiting for update",
    "status not clear", "no information", "no tracking update"
]

SOFT_BEHAVIOUR_PATTERNS = [
    "not contacted", "no call received", "did not inform",
    "no intimation given", "no message received"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT (STRICT PRIORITY + SOFT RECOVERY)
# ------------------------------------------------------------

def assign_aspect(text):
    t = text.lower()

    # 2 = DAMAGE / LOST (HIGHEST SEVERITY)
    if any(k in t for k in DAMAGE_KW):
        return 2, "damage_lost"

    # 1 = WRONG DELIVERY
    if any(k in t for k in WRONG_KW):
        return 1, "wrong_delivery"

    # 0 = DELAY (HARD)
    if any(k in t for k in DELAY_KW):
        return 0, "delay"

    # 7 = REFUND
    if any(k in t for k in REFUND_KW):
        return 7, "refund"

    # 6 = PRICING
    if any(k in t for k in PRICING_KW):
        return 6, "pricing"

    # 5 = TRACKING (HARD)
    if any(k in t for k in TRACKING_KW):
        return 5, "tracking"

    # 3 = BEHAVIOUR (HARD)
    if any(k in t for k in BEHAVIOUR_KW):
        return 3, "behaviour"

    # ---------------- SOFT RECOVERY LAYER ----------------

    # Soft Delay
    if any(k in t for k in SOFT_DELAY_PATTERNS):
        return 0, "soft_delay"

    # Soft Tracking
    if any(k in t for k in SOFT_TRACKING_PATTERNS):
        return 5, "soft_tracking"

    # Soft Behaviour
    if any(k in t for k in SOFT_BEHAVIOUR_PATTERNS):
        return 3, "soft_behaviour"

    # ---------------- SUPPORT (ONLY IF EXPLICIT) ----------------

    if any(k in t for k in SUPPORT_KW):
        return 4, "support"

    # ---------------- FINAL FALLBACK ----------------

    return 4, "fallback_support"


# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_cols = ["text", "sentiment", "customer_intent"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

aspects = []
flags = []

for text in df["text"]:
    asp, flag = assign_aspect(text)
    aspects.append(asp)
    flags.append(flag)

df["primary_aspect"] = aspects
df["aspect_flag"] = flags

# SAVE OUTPUT
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Aspect v3 labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect Distribution (v3) ---")
print(df["primary_aspect"].value_counts())

print("\n--- Aspect Flags (Top 20) ---")
print(df["aspect_flag"].value_counts().head(20))

# ============================================================
# FILE 1 : generate_emotion_v3.py
# PURPOSE: Phase-5 GOLD CLEANER — Reduce Frustrated Noise
# STRATEGY: Sharper Frustrated rules, calmer default
# ============================================================

import pandas as pd

INPUT_PATH  = r"D:/model_bert_copy/data/gold/cx_phase4/val.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase5/val_with_emotion_v3.csv"

VERY_ANGRY_KW = ["fraud","cheated","scam","court","legal","harassment","refund fraud","money stolen"]
ANGRY_KW = ["angry","furious","complaint","bad service","poor service","rude","refund pending","lost","damaged","wrong delivery"]
FRUSTRATED_KW = ["waiting","delay","pending","no update","follow up","concern","confused","still waiting"]
SATISFIED_KW = ["good","great","excellent","happy","satisfied","thanks","resolved","well done"]


def assign_emotion(text, sentiment, aspect, aspect_sentiment):
    t = text.lower()

    if any(k in t for k in VERY_ANGRY_KW):
        return 4

    if sentiment == 0 and aspect in [2,7] and aspect_sentiment == 0:
        return 4

    if any(k in t for k in ANGRY_KW):
        return 3

    if sentiment == 0 and aspect_sentiment == 0:
        return 3

    if any(k in t for k in FRUSTRATED_KW) and sentiment != 2:
        return 2

    if any(k in t for k in SATISFIED_KW) or (sentiment == 2 and aspect_sentiment == 2):
        return 1

    return 0


df = pd.read_csv(INPUT_PATH)

emotions = []
for _, row in df.iterrows():
    emotions.append(assign_emotion(row["text"], int(row["sentiment"]), int(row["primary_aspect"]), int(row["aspect_sentiment"])))

df["emotion"] = emotions

df.to_csv(OUTPUT_PATH, index=False)

print("✅ Emotion v3 generated")
print(df["emotion"].value_counts())

import pandas as pd
import re

# Load your existing train.csv
df = pd.read_csv("D:/model_bert_copy/data/gold/v2.3_multitask/val.csv")

def assign_intent(text, sentiment):
    t = text.lower()

    # Complaint rules
    if sentiment == 0 or any(k in t for k in [
        "delay", "late", "not delivered", "no contact", "no visit",
        "wrong", "issue", "problem", "bad", "worst", "rude",
        "missing", "damage", "damaged", "lost", "refund"
    ]):
        return 0   # Complaint

    # Praise rules
    if sentiment == 2 or any(k in t for k in [
        "good", "nice", "excellent", "happy", "thanks", "thank you",
        "great", "perfect", "smooth", "clear", "awesome"
    ]):
        return 2   # Praise

    # Inquiry rules
    if any(k in t for k in [
        "where", "when", "status", "track", "tracking", "update", "why"
    ]):
        return 1   # Inquiry

    # Default
    return 1       # Inquiry


# Apply
df["sentiment"] = df["label"]          # keep your mapping
df["customer_intent"] = df.apply(
    lambda x: assign_intent(x["text"], x["sentiment"]),
    axis=1
)

# Keep only required columns
out_df = df[["text", "sentiment", "customer_intent"]]

# Save new multitask dataset
out_path = "D:/model_bert_copy/data/gold/v2.3_multitask/val_multi.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8")

print("✅ Multitask dataset created:", out_path)
print(out_df["customer_intent"].value_counts())

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
