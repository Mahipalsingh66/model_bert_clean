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
