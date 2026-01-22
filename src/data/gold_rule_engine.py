# src/data/gold_rule_engine.py

import yaml
import pandas as pd
from pathlib import Path

def load_rules(rule_path):
    with open(rule_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def apply_rules_to_row(text, label, rules):
    text_lower = text.lower()
    applied_rule = None
    flags = []

    # ---- HARD RULES ----
    for rule in rules.get("hard_rules", []):
        cond = rule["if"]
        if cond.get("label") == label:
            if "contains_any" in cond:
                if any(k.lower() in text_lower for k in cond["contains_any"]):
                    label = rule["then"]["set_label"]
                    applied_rule = rule["id"]
                    break  # stop at first hard rule

    # ---- SOFT RULES ----
    for rule in rules.get("soft_rules", []):
        cond = rule["if"]

        if "contains_any" in cond:
            if any(k.lower() in text_lower for k in cond["contains_any"]):
                flags.append(rule["then"]["flag"])

        if "token_length_lt" in cond:
            if len(text.split()) < cond["token_length_lt"]:
                flags.append(rule["then"]["flag"])

    return label, applied_rule, ",".join(flags) if flags else None

def generate_gold_v2(input_csv, rules_yaml, output_csv):
    rules = load_rules(rules_yaml)
    df = pd.read_csv(input_csv)

    out_rows = []

    for _, row in df.iterrows():
        final_label, rule_id, flags = apply_rules_to_row(
            row["text"],
            row["label"],
            rules
        )

        out_rows.append({
            "text": row["text"],
            "original_label": row["label"],
            "final_label": final_label,
            "rule_applied": rule_id,
            "flags": flags,
            "gold_version": "v2"
        })

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"✅ Gold_v2 generated → {output_csv}")
    print("Changed labels:",
          (out_df["original_label"] != out_df["final_label"]).sum())

if __name__ == "__main__":
    generate_gold_v2(
        input_csv="D:/model_bert_copy/data/gold/v1/train.csv",
        rules_yaml="D:/model_bert_copy/data/gold/v2/RULES.yaml",
        output_csv="data/gold/v2/train.csv"
    )
