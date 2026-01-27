import pandas as pd

# -------------------------------------------------
# FIXED PATHS (NO CONFUSION)
# -------------------------------------------------

INPUT_PATH  = r"D:/model_bert_copy/data/gold/v2.3_multitask/val_multi.csv"
OUTPUT_PATH = r"D:/model_bert_copy/data/gold/v2.3_multitask/val_multi_v1.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

# -------------------------------------------------
# CUSTOMER INTENT RULES
# -------------------------------------------------

def assign_intent(text, sentiment):
    t = text.lower()

    # Complaint
    if sentiment == 0 or any(k in t for k in [
        "delay", "late", "not delivered", "no contact", "no visit",
        "wrong", "issue", "problem", "bad", "worst", "rude",
        "missing", "damage", "damaged", "lost", "refund"
    ]):
        return 0   # Complaint

    # Praise
    if sentiment == 2 or any(k in t for k in [
        "good", "nice", "excellent", "happy", "thanks", "thank you",
        "great", "perfect", "smooth", "clear", "awesome"
    ]):
        return 2   # Praise

    # Inquiry
    if any(k in t for k in [
        "where", "when", "status", "track", "tracking", "update", "why"
    ]):
        return 1   # Inquiry

    return 1       # Default Inquiry


# -------------------------------------------------
# PRIMARY ASPECT RULES
# -------------------------------------------------

def assign_aspect(text):
    t = text.lower()

    # Behaviour
    if any(k in t for k in [
        "rude", "behaviour", "behavior", "attitude", "misbehave",
        "unprofessional", "abuse", "angry"
    ]):
        return 2   # Behaviour

    # Support
    if any(k in t for k in [
        "call", "calling", "support", "helpline", "customer care",
        "service center", "complaint desk", "ticket"
    ]):
        return 3   # Support

    # Pricing
    if any(k in t for k in [
        "charge", "charged", "extra", "amount", "fee", "refund",
        "payment", "cod", "price", "cost"
    ]):
        return 4   # Pricing

    # Tracking
    if any(k in t for k in [
        "track", "tracking", "status", "update", "awb", "waybill"
    ]):
        return 1   # Tracking

    # Default
    return 0       # Delivery


# -------------------------------------------------
# APPLY RULES
# -------------------------------------------------

df["sentiment"] = df["label"]

df["customer_intent"] = df.apply(
    lambda x: assign_intent(x["text"], x["sentiment"]),
    axis=1
)

df["primary_aspect"] = df["text"].apply(assign_aspect)

# Final CX Phase-1 dataset
out_df = df[["text", "sentiment", "customer_intent", "primary_aspect"]]

out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("✅ CX Phase-1 dataset created")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)

print("\n--- Intent distribution ---")
print(out_df["customer_intent"].value_counts())

print("\n--- Aspect distribution ---")
print(out_df["primary_aspect"].value_counts())
