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
