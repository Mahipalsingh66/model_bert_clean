# ============================================================
# FILE    : logistics_aspect_engine.py
# PURPOSE : Logistics Industry Primary Aspect Classification
# AUTHOR  : Finalized for Production Use
#
# INDUSTRY : LOGISTICS
#
# PRIMARY ASPECT TAXONOMY (LOCKED)
# 0 = Delivery Failures
# 1 = Wrong Delivery
# 2 = Agent Behaviour
# 3 = Customer Support
# 4 = Tracking / Updates
# 5 = Damaged Package
# 6 = Payment Issues
# 7 = Service Quality
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_priority_5step.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/multitask_raw_data/train_multi_logistics_with new sentimen_6step.csv"

# ------------------------------------------------------------
# KEYWORD BANK (HIGH PRECISION, ENTERPRISE SAFE)
# ------------------------------------------------------------

DELIVERY_FAILURE_KW = [
    "late", "delay", "delayed", "not delivered", "delivery failed",
    "delivery pending", "pending delivery", "still waiting",
    "delivery attempt failed", "not attempted", "delivery not done",
    "delivery postponed", "delivery rescheduled",
    "out for delivery but not delivered",
    "shipment delayed", "delivery very slow", "took too long"
]

WRONG_DELIVERY_KW = [
    "wrong delivery", "wrong address", "wrong location", "wrong person",
    "delivered to someone else", "delivered to wrong person",
    "unknown person received", "maid received", "security received",
    "neighbor received", "not my order", "not my parcel",
    "wrong flat", "wrong building", "wrong pin code",
    "misdelivered", "incorrect delivery"
]

AGENT_BEHAVIOUR_KW = [
    "rude", "misbehave", "misbehavior", "unprofessional",
    "bad attitude", "shouted", "argued", "abused",
    "delivery boy rude", "agent rude", "staff rude",
    "not polite", "very arrogant", "harassment",
    "did not call", "no call", "not contacted",
    "no intimation", "did not inform", "no message",
    "agent not cooperative"
]

CUSTOMER_SUPPORT_KW = [
    "customer support", "customer care", "support team",
    "called support", "no response", "not responding",
    "no reply", "waiting for response", "complaint registered",
    "ticket raised", "case id", "grievance",
    "escalation", "follow up", "no callback",
    "helpline", "service center"
]

TRACKING_KW = [
    "tracking", "track", "tracking not updated",
    "status not updated", "no update", "no tracking",
    "awb", "waybill", "consignment",
    "where is my order", "where is my parcel",
    "showing delivered but not received",
    "fake delivery status", "tracking wrong",
    "stuck in transit", "no movement"
]

DAMAGED_PACKAGE_KW = [
    "damaged", "broken", "cracked", "leaked", "torn",
    "open packet", "opened parcel", "empty box",
    "item missing", "parcel missing", "content missing",
    "lost", "package lost", "shipment lost",
    "stolen", "tampered", "seal broken",
    "box damaged", "nothing inside"
]

PAYMENT_ISSUES_KW = [
    "payment issue", "extra charge", "charged extra",
    "overcharged", "wrong amount", "high charge",
    "cod charge", "cash on delivery charge",
    "amount deducted", "double charged",
    "refund issue", "refund pending", "refund delayed",
    "refund not received", "money not returned"
]

SERVICE_QUALITY_KW = [
    "poor service", "bad service", "service quality",
    "very bad experience", "disappointed",
    "not satisfied", "unsatisfactory service",
    "worst experience", "pathetic service",
    "service is bad"
]

# ------------------------------------------------------------
# ASPECT ASSIGNMENT (STRICT PRIORITY ORDER)
# ------------------------------------------------------------

def assign_logistics_aspect(text: str):
    """
    Returns:
        (primary_aspect_id, primary_aspect_name)
    """
    if not isinstance(text, str) or not text.strip():
        return 7, "service_quality"

    t = text.lower()

    if any(k in t for k in DAMAGED_PACKAGE_KW):
        return 5, "damaged_package"

    if any(k in t for k in WRONG_DELIVERY_KW):
        return 1, "wrong_delivery"

    if any(k in t for k in DELIVERY_FAILURE_KW):
        return 0, "delivery_failures"

    if any(k in t for k in TRACKING_KW):
        return 4, "tracking_updates"

    if any(k in t for k in AGENT_BEHAVIOUR_KW):
        return 2, "agent_behaviour"

    if any(k in t for k in PAYMENT_ISSUES_KW):
        return 6, "payment_issues"

    if any(k in t for k in CUSTOMER_SUPPORT_KW):
        return 3, "customer_support"

    if any(k in t for k in SERVICE_QUALITY_KW):
        return 7, "service_quality"

    return 7, "service_quality"

# ------------------------------------------------------------
# APPLY TO DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_cols = ["text", "sentiment", "customer_intent"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

aspect_ids = []
aspect_names = []

for text in df["text"]:
    asp_id, asp_name = assign_logistics_aspect(text)
    aspect_ids.append(asp_id)
    aspect_names.append(asp_name)

df["primary_aspect"] = aspect_ids
df["primary_aspect_name"] = aspect_names

# SAVE OUTPUT
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ Logistics aspect labels generated successfully")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Aspect Distribution ---")
print(df["primary_aspect"].value_counts())

print("\n--- Aspect Names (Top 20) ---")
print(df["primary_aspect_name"].value_counts().head(20))

# ------------------------------------------------------------
# QUICK TEST (OPTIONAL)
# ------------------------------------------------------------
if __name__ == "__main__":
    samples = [
        "Delivery agent was very rude and did not call",
        "Package was damaged and seal was broken",
        "Tracking shows delivered but I did not receive",
        "Wrong person received my parcel",
        "Refund not received after return",
        "Very bad service experience overall"
    ]

    for s in samples:
        print(s, "->", assign_logistics_aspect(s))
