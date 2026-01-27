# # src/data/gold_rule_engine.py

# import yaml
# import pandas as pd
# from pathlib import Path

# def load_rules(rule_path):
#     with open(rule_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

# def apply_rules_to_row(text, label, rules):
#     text_lower = text.lower()
#     applied_rule = None
#     flags = []

#     # ---- HARD RULES ----
#     for rule in rules.get("hard_rules", []):
#         cond = rule["if"]
#         if cond.get("label") == label:
#             if "contains_any" in cond:
#                 if any(k.lower() in text_lower for k in cond["contains_any"]):
#                     label = rule["then"]["set_label"]
#                     applied_rule = rule["id"]
#                     break  # stop at first hard rule

#     # ---- SOFT RULES ----
#     for rule in rules.get("soft_rules", []):
#         cond = rule["if"]

#         if "contains_any" in cond:
#             if any(k.lower() in text_lower for k in cond["contains_any"]):
#                 flags.append(rule["then"]["flag"])

#         if "token_length_lt" in cond:
#             if len(text.split()) < cond["token_length_lt"]:
#                 flags.append(rule["then"]["flag"])

#     return label, applied_rule, ",".join(flags) if flags else None

# def generate_gold_v2(input_csv, rules_yaml, output_csv):
#     rules = load_rules(rules_yaml)
#     df = pd.read_csv(input_csv)

#     out_rows = []

#     for _, row in df.iterrows():
#         final_label, rule_id, flags = apply_rules_to_row(
#             row["text"],
#             row["label"],
#             rules
#         )

#         out_rows.append({
#             "text": row["text"],
#             "original_label": row["label"],
#             "final_label": final_label,
#             "rule_applied": rule_id,
#             "flags": flags,
#             "gold_version": "v2"
#         })

#     out_df = pd.DataFrame(out_rows)
#     out_df.to_csv(output_csv, index=False, encoding="utf-8")

#     print(f"✅ Gold_v2 generated → {output_csv}")
#     print("Changed labels:",
#           (out_df["original_label"] != out_df["final_label"]).sum())

# if __name__ == "__main__":
#     generate_gold_v2(
#         input_csv="D:/model_bert_copy/data/gold/v2.1/train.csv",
#         rules_yaml="D:/model_bert_copy/data/gold/v2.2/RULES.yaml",
#         output_csv="data/gold/v2.2/train.csv"
#     )

# ============================================================
# GOLD RULE ENGINE — ENTERPRISE PRODUCTION VERSION
# Compatible with Gold_v2.3 rules.yaml
# Owner   : Mahipal Singh
# Purpose : Deterministic supervision for Golden Dataset
# ============================================================

# ============================================================
# GOLD RULE ENGINE — ENTERPRISE WEAK SUPERVISION VERSION
# Purpose : Correct only high-confidence errors, preserve signal
# Owner   : Mahipal Singh
# Version : Gold_v2.3 FINAL
# ============================================================

import yaml
import pandas as pd
import re
from pathlib import Path


# ------------------------------------------------------------
# LOAD RULE CONFIG
# ------------------------------------------------------------

def load_rules(rule_path):
    with open(rule_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_text(text: str, cfg: dict) -> str:
    if text is None:
        return ""

    t = str(text)

    if cfg.get("lowercase", True):
        t = t.lower()

    if cfg.get("remove_punctuation", True):
        t = re.sub(r"[^\w\s]", " ", t)

    if cfg.get("normalize_whitespace", True):
        t = re.sub(r"\s+", " ", t).strip()

    for rep in cfg.get("replace_patterns", []):
        t = re.sub(rep["pattern"], rep["replace"], t)

    return t


# ------------------------------------------------------------
# BUILD FLAT RULE LIST (ONLY REAL RULES)
# ------------------------------------------------------------

def build_rule_list(cfg: dict):
    rule_blocks = []

    for section_name, rules in cfg.items():

        if not isinstance(rules, list):
            continue

        for rule in rules:

            # Must be a valid rule dict
            if not isinstance(rule, dict):
                continue

            if "patterns" not in rule or "label" not in rule:
                continue

            rule_copy = rule.copy()
            rule_copy["section"] = section_name
            rule_blocks.append(rule_copy)

    # Strict priority ordering
    rule_blocks.sort(key=lambda r: r.get("priority", 999))

    print(f"🔹 Loaded {len(rule_blocks)} active rules from YAML")

    return rule_blocks


# ------------------------------------------------------------
# CONFLICT RESOLUTION (SIMPLE + SAFE)
# ------------------------------------------------------------

def resolve_conflicts(matches):
    # Already sorted by priority → lowest number wins
    return matches[0]


# ------------------------------------------------------------
# CORE RULE APPLICATION — SAFE OVERRIDE LOGIC
# ------------------------------------------------------------

def apply_rules_to_row(text, original_label, cfg, rule_list):

    normalized_text = normalize_text(text, cfg["normalization"])

    matches = []

    # Find matching rules
    for rule in rule_list:
        for pattern in rule.get("patterns", []):
            try:
                if re.search(pattern, normalized_text):
                    matches.append(rule)
                    break
            except re.error:
                continue

    # --------------------------------------------
    # NO MATCH → KEEP ORIGINAL LABEL COMPLETELY
    # --------------------------------------------
    if not matches:
        return {
            "final_label": original_label,
            "rule_id": None,
            "rule_applied": None,
            "priority": None,
            "flags": []
        }

    final_rule = resolve_conflicts(matches)
    section = final_rule["section"]
    new_label = final_rule["label"]

    # --------------------------------------------
    # ENTERPRISE OVERRIDE POLICY (VERY IMPORTANT)
    # --------------------------------------------

    OVERRIDE_SECTIONS = [
        "hard_negative",
        "hard_positive",
        "strong_negative",
        "strong_positive"
    ]

    # Case 1 — High-confidence rules may override
    if section in OVERRIDE_SECTIONS:

        # Only override if it actually changes sentiment
        if new_label != original_label:
            return {
                "final_label": new_label,
                "rule_id": final_rule.get("id"),
                "rule_applied": final_rule.get("rule_applied"),
                "priority": final_rule.get("priority"),
                "flags": final_rule.get("flags", [])
            }
        else:
            # Same label → just log rule
            return {
                "final_label": original_label,
                "rule_id": final_rule.get("id"),
                "rule_applied": "confirm_" + final_rule.get("rule_applied"),
                "priority": final_rule.get("priority"),
                "flags": final_rule.get("flags", [])
            }

    # --------------------------------------------
    # Case 2 — Neutral / Mixed rules ONLY act if original already Neutral
    # --------------------------------------------

    if section in ["neutral_override", "mixed_sentiment"]:
        if original_label == 1:
            return {
                "final_label": new_label,
                "rule_id": final_rule.get("id"),
                "rule_applied": final_rule.get("rule_applied"),
                "priority": final_rule.get("priority"),
                "flags": final_rule.get("flags", [])
            }
        else:
            # Do NOT override sentiment — only flag
            return {
                "final_label": original_label,
                "rule_id": final_rule.get("id"),
                "rule_applied": "flag_only_" + final_rule.get("rule_applied"),
                "priority": final_rule.get("priority"),
                "flags": final_rule.get("flags", [])
            }

    # --------------------------------------------
    # Case 3 — Soft / Domain / Fallback NEVER override
    # --------------------------------------------

    return {
        "final_label": original_label,
        "rule_id": final_rule.get("id"),
        "rule_applied": "flag_only_" + final_rule.get("rule_applied"),
        "priority": final_rule.get("priority"),
        "flags": final_rule.get("flags", [])
    }


# ------------------------------------------------------------
# BATCH GENERATOR — MAIN PIPELINE
# ------------------------------------------------------------

def generate_gold_v2_3(input_csv, rules_yaml, output_csv,
                       text_col="text",
                       label_col="label"):

    print("🔹 Loading rules...")
    cfg = load_rules(rules_yaml)
    rule_list = build_rule_list(cfg)

    print("🔹 Loading dataset...")
    df = pd.read_csv(input_csv)

    out_rows = []
    changed = 0

    print("🔹 Applying rule engine...")

    for _, row in df.iterrows():
        text = row[text_col]
        original_label = row[label_col]

        result = apply_rules_to_row(
            text=text,
            original_label=original_label,
            cfg=cfg,
            rule_list=rule_list
        )

        final_label = result["final_label"]

        if final_label != original_label:
            changed += 1

        out_rows.append({
            "text": text,
            "original_label": original_label,
            "final_label": final_label,
            "rule_id": result["rule_id"],
            "rule_applied": result["rule_applied"],
            "priority": result["priority"],
            "flags": ",".join(result["flags"]) if result["flags"] else None,
            "gold_version": "v2.3"
        })

    out_df = pd.DataFrame(out_rows)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    print("==============================================")
    print(f"✅ Gold_v2.3 generated → {output_csv}")
    print(f"Total rows      : {len(out_df)}")
    print(f"Labels changed  : {changed}")
    print(f"Change ratio    : {changed / len(out_df):.4f}")
    print("==============================================")


# ------------------------------------------------------------
# CLI ENTRY POINT (MATCHING YOUR STYLE)
# ------------------------------------------------------------

if __name__ == "__main__":

    generate_gold_v2_3(
        input_csv="D:/model_bert_copy/data/gold/v2.2/train.csv",
        rules_yaml="D:/model_bert_copy/data/gold/v2.3/rules.yaml",
        output_csv="D:/model_bert_copy/data/gold/v2.3/train.csv",
        text_col="text",
        label_col="label"
    )
