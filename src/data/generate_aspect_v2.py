# ============================================================
# FILE    : generate_aspect_v2.py
# PURPOSE : Enterprise Aspect Rule Engine (Balanced, Production)
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

INPUT_PATH  = r"D:/model_bert_copy/data/gold/v2.3_multitask/train_cx_v1.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/cx_phase3/train_with_aspect_v2.csv"

# ------------------------------------------------------------
# KEYWORD BANK (ENTERPRISE-GRADE)
# ------------------------------------------------------------

DELAY_KW = [
    "late", "delay", "delayed", "one day late", "two days late",
    "not on time", "not delivered on time", "delivery pending",
    "pending for days", "still waiting", "no delivery yet",
    "after due date", "delivery not attempted", "attempted late",
    "shipment delayed", "out for delivery but not delivered",
    "postponed", "rescheduled delivery"
]

WRONG_KW = [
    "wrong address", "wrong location", "wrong place",
    "delivered to wrong", "delivered somewhere else",
    "someone else received", "unknown person received",
    "maid received", "neighbor received", "security received",
    "not my order", "not my package", "not my parcel",
    "wrong pin code", "wrong building", "wrong flat",
    "misdelivered", "incorrect delivery"
]

DAMAGE_KW = [
    "damaged", "damage", "broken", "cracked", "leak", "leaked",
    "torn", "open packet", "opened parcel", "empty box",
    "missing item", "item missing", "parcel missing",
    "lost", "not received", "did not receive", "never received",
    "package lost", "shipment lost", "product missing",
    "stolen", "tampered", "seal broken"
]

BEHAVIOUR_KW = [
    "rude", "misbehave", "misbehavior", "behaviour", "behavior",
    "bad attitude", "unprofessional", "shouted", "argued",
    "angry delivery", "delivery boy rude", "staff rude",
    "threatened", "abused", "not polite", "very arrogant",
    "impolite", "bad manners", "harassment", "unacceptable behaviour"
]

TRACKING_KW = [
    "track", "tracking", "tracking not updated", "status not updated",
    "no update", "no tracking", "awb", "waybill", "consignment",
    "where is my order", "where is my parcel", "where is shipment",
    "no status", "status wrong", "location not updated",
    "showing delivered but not received", "fake delivery status",
    "system shows delivered"
]

PRICING_KW = [
    "charged", "extra charge", "overcharged", "wrong amount",
    "high charge", "pricing issue", "price mismatch",
    "fee", "service charge", "cod charge", "cash on delivery charge",
    "billing issue", "payment issue", "amount deducted",
    "double charged", "refund amount wrong"
]

REFUND_KW = [
    "refund", "refunded", "money back", "amount not credited",
    "return", "returned", "return process", "return completed",
    "refund pending", "refund delayed", "not refunded",
    "refund not received", "replacement", "exchange",
    "cancelled order refund", "reverse pickup"
]

SUPPORT_KW = [
    "call", "called", "contact", "support", "customer care",
    "helpline", "complaint", "registered complaint",
    "no response", "not responding", "no reply", "waiting for response",
    "ticket", "case id", "grievance", "escalation", "follow up",
    "no callback", "service issue", "helpdesk", "query"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT (PRIORITY ORDER)
# ------------------------------------------------------------

def assign_aspect(text):
    t = text.lower()

    # 2 = DAMAGE / LOST (HIGHEST SEVERITY)
    if any(k in t for k in DAMAGE_KW):
        return 2, "damage_lost"

    # 1 = WRONG DELIVERY
    if any(k in t for k in WRONG_KW):
        return 1, "wrong_delivery"

    # 0 = DELAY
    if any(k in t for k in DELAY_KW):
        return 0, "delay"

    # 7 = REFUND
    if any(k in t for k in REFUND_KW):
        return 7, "refund"

    # 6 = PRICING
    if any(k in t for k in PRICING_KW):
        return 6, "pricing"

    # 5 = TRACKING
    if any(k in t for k in TRACKING_KW):
        return 5, "tracking"

    # 3 = BEHAVIOUR
    if any(k in t for k in BEHAVIOUR_KW):
        return 3, "behaviour"

    # 4 = SUPPORT (ONLY IF SUPPORT WORDS EXIST)
    if any(k in t for k in SUPPORT_KW):
        return 4, "support"

    # DEFAULT FALLBACK (LAST RESORT)
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

print("✅ Aspect v2 labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect Distribution (v2) ---")
print(df["primary_aspect"].value_counts())

print("\n--- Aspect Flags (Top 20) ---")
print(df["aspect_flag"].value_counts().head(20))
