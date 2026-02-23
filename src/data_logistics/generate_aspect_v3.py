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

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_aspect_2step.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_aspect_2step.csv"

# ------------------------------------------------------------
# KEYWORD BANK (ENGLISH + HINGLISH)
# ------------------------------------------------------------

DELAY_KW = [
    "late", "delay", "delayed", "delivery late", "delivery delay",
    "not on time", "not delivered on time", "delivery pending",
    "pending delivery", "still waiting", "no delivery yet",
    "after due date", "delivery not attempted",
    "shipment delayed", "out for delivery but not delivered",
    "delivery slow", "took long time", "taking too long",

    # Hinglish
    "bahut late", "abhi tak deliver nahi hua",
    "delivery abhi tak nahi aayi",
    "kab deliver hoga", "delivery pending hai",
    "der se aaya", "late aaya"
]

WRONG_KW = [
    "wrong address", "wrong location", "wrong place",
    "delivered to wrong", "someone else received",
    "unknown person received", "maid received",
    "neighbor received", "security received",
    "not my order", "not my parcel",
    "wrong pin code", "wrong flat", "misdelivered",

    # Hinglish
    "galat address", "galat jagah deliver",
    "kisi aur ko de diya",
    "galat parcel", "mera order nahi hai"
]

DAMAGE_KW = [
    "damaged", "damage", "broken", "cracked", "leaked",
    "torn", "open packet", "opened parcel", "empty box",
    "missing item", "item missing", "parcel missing",
    "lost", "not received", "never received",
    "stolen", "tampered", "seal broken",

    # Hinglish
    "toota hua", "khula hua packet",
    "samaan missing", "box khali tha",
    "parcel nahi mila", "samaan chori ho gaya"
]

BEHAVIOUR_KW = [
    "rude", "misbehave", "bad behaviour",
    "unprofessional", "argued", "abused",
    "delivery boy rude", "agent rude",
    "not polite", "harassment",
    "did not call", "no call", "not contacted",

    # Hinglish
    "bad behaviour", "misbehave kiya",
    "baat karne ka tareeka kharab",
    "delivery boy badtameez",
    "call nahi kiya", "baat nahi ki"
]

TRACKING_KW = [
    "track", "tracking", "tracking not updated",
    "status not updated", "no update",
    "awb", "waybill", "consignment",
    "showing delivered but not received",
    "fake delivery status", "stuck in transit",

    # Hinglish
    "tracking update nahi hai",
    "status galat dikha raha",
    "delivered dikha raha hai par mila nahi",
    "tracking issue hai"
]

PRICING_KW = [
    "charged", "extra charge", "overcharged",
    "wrong amount", "high charge",
    "cod charge", "service charge",
    "amount deducted", "double charged",

    # Hinglish
    "extra paisa liya",
    "zyada charge kiya",
    "galat amount kata",
    "paise zyada kat gaye"
]

REFUND_KW = [
    "refund", "refunded", "money back",
    "amount not credited", "refund pending",
    "refund delayed", "not refunded",
    "replacement", "return completed",

    # Hinglish
    "refund nahi mila",
    "paise wapas nahi aaye",
    "refund pending hai",
    "return ka paisa nahi aaya"
]

SUPPORT_KW = [
    "call", "contact", "support",
    "customer care", "helpline",
    "complaint", "ticket raised",
    "no response", "not responding",

    # Hinglish
    "customer care se baat nahi hui",
    "call connect nahi ho raha",
    "support se response nahi mila"
]

# ------------------------------------------------------------
# SOFT PATTERNS
# ------------------------------------------------------------

SOFT_DELAY_PATTERNS = [
    "delivery slow", "still not received",
    "abhi tak nahi mila"
]

SOFT_TRACKING_PATTERNS = [
    "no update yet", "tracking unclear",
    "status clear nahi hai"
]

SOFT_BEHAVIOUR_PATTERNS = [
    "not contacted", "call nahi aaya"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT
# ------------------------------------------------------------

def assign_aspect(text):
    if not isinstance(text, str) or not text.strip():
        return 4, "fallback_support"

    t = text.lower()

    if any(k in t for k in DAMAGE_KW):
        return 2, "damage_lost"

    if any(k in t for k in WRONG_KW):
        return 1, "wrong_delivery"

    if any(k in t for k in DELAY_KW):
        return 0, "delay"

    if any(k in t for k in REFUND_KW):
        return 7, "refund"

    if any(k in t for k in PRICING_KW):
        return 6, "pricing"

    if any(k in t for k in TRACKING_KW):
        return 5, "tracking"

    if any(k in t for k in BEHAVIOUR_KW):
        return 3, "behaviour"

    if any(k in t for k in SOFT_DELAY_PATTERNS):
        return 0, "soft_delay"

    if any(k in t for k in SOFT_TRACKING_PATTERNS):
        return 5, "soft_tracking"

    if any(k in t for k in SOFT_BEHAVIOUR_PATTERNS):
        return 3, "soft_behaviour"

    if any(k in t for k in SUPPORT_KW):
        return 4, "support"

    return 4, "fallback_support"

# ------------------------------------------------------------
# APPLY
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH, encoding="latin1", engine="python")

aspects, flags = [], []

for text in df["text"]:
    a, f = assign_aspect(text)
    aspects.append(a)
    flags.append(f)

df["primary_aspect"] = aspects
df["aspect_flag"] = flags

df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Logistics Aspect v3 (HINGLISH) generated")
print(df["primary_aspect"].value_counts())