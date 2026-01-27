# ============================================================
# FILE    : generate_aspect_v1.py
# PURPOSE : Enterprise Aspect Rule Engine (Phase-3)
# INPUT   : text,sentiment,customer_intent
# OUTPUT  : text,sentiment,customer_intent,primary_aspect,aspect_flag
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

# ------------------------------------------------------------
# FIXED PATHS (CHANGE ONLY IF NEEDED)
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/v2.3_multitask/val_multi_v1.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/v2.3_multitask/val_with_aspect_v1.csv"

# ------------------------------------------------------------
# ASPECT RULE ENGINE
# ------------------------------------------------------------

def assign_aspect(text):
    t = text.lower()

    # --------------------
    # 2 = DAMAGE / LOST (HIGHEST PRIORITY)
    # --------------------
    if any(k in t for k in [
        "damaged", "damage", "broken", "cracked",
        "lost", "missing", "not received", "did not receive",
        "parcel missing", "item missing"
    ]):
        return 2, "damage_lost"

    # --------------------
    # 1 = WRONG DELIVERY
    # --------------------
    if any(k in t for k in [
        "wrong address", "wrong location", "delivered to wrong",
        "someone else", "unknown person", "wrong person",
        "maid", "neighbor", "not my address"
    ]):
        return 1, "wrong_delivery"

    # --------------------
    # 0 = DELAY
    # --------------------
    if any(k in t for k in [
        "late", "delay", "delayed", "one day late",
        "not on time", "after due", "pending for days",
        "still not delivered"
    ]):
        return 0, "delay"

    # --------------------
    # 3 = BEHAVIOUR
    # --------------------
    if any(k in t for k in [
        "rude", "misbehave", "behavior", "behaviour",
        "attitude", "unprofessional", "shouted", "angry",
        "delivery boy rude", "staff rude"
    ]):
        return 3, "behaviour"

    # --------------------
    # 7 = REFUND
    # --------------------
    if any(k in t for k in [
        "refund", "returned", "return process", "money back",
        "amount not credited", "return completed"
    ]):
        return 7, "refund"

    # --------------------
    # 6 = PRICING
    # --------------------
    if any(k in t for k in [
        "charged", "extra charge", "fee", "amount",
        "pricing", "cost", "payment", "cod issue"
    ]):
        return 6, "pricing"

    # --------------------
    # 5 = TRACKING
    # --------------------
    if any(k in t for k in [
        "track", "tracking", "status", "update",
        "awb", "waybill", "where is my order"
    ]):
        return 5, "tracking"

    # --------------------
    # 4 = SUPPORT (DEFAULT SERVICE ISSUES)
    # --------------------
    if any(k in t for k in [
        "call", "support", "customer care", "helpline",
        "no response", "not responding", "complaint desk"
    ]):
        return 4, "support"

    # --------------------
    # DEFAULT FALLBACK
    # --------------------
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

# Save

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Aspect labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect Distribution ---")
print(df["primary_aspect"].value_counts())

print("\n--- Aspect Flags (Top) ---")
print(df["aspect_flag"].value_counts().head(10))
