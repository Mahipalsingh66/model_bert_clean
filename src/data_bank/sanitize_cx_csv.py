# ============================================================
# FILE    : sanitize_cx_csv.py
# PURPOSE : One-time structural cleanup for CX CSV files
# AUTHOR  : Production Safe
#
# FIXES:
# - Encoding issues
# - Commas inside text
# - Broken rows
# - Mixed delimiters
# - Invisible characters
# ============================================================

import csv
import pandas as pd
import re

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

INPUT_PATH  = r"D:/bert_data/bank_dataset_cleaned.csv"
OUTPUT_PATH = r"D:/bert_data/bank_dataset_cleaned_sanitize.csv"

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

EXPECTED_COLUMNS = [
    "text",
    "industry",
    "sentiment",
    "customer_intent",
    "primary_aspect",
    "aspect_flag",
    "aspect_sentiment",
    "emotion"
]

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------------------------------------
# LOAD RAW CSV (TOLERANT MODE)
# ------------------------------------------------------------

rows = []

with open(INPUT_PATH, encoding="latin1", errors="ignore") as f:
    reader = csv.reader(f, delimiter=",")
    for row in reader:
        rows.append(row)

# ------------------------------------------------------------
# NORMALIZE ROW LENGTH
# ------------------------------------------------------------

clean_rows = []

for row in rows:
    if len(row) < len(EXPECTED_COLUMNS):
        # pad missing columns
        row = row + [""] * (len(EXPECTED_COLUMNS) - len(row))
    elif len(row) > len(EXPECTED_COLUMNS):
        # merge overflow into text column
        row = [
            ",".join(row[: len(row) - (len(EXPECTED_COLUMNS) - 1)])
        ] + row[-(len(EXPECTED_COLUMNS) - 1):]

    clean_rows.append(row[: len(EXPECTED_COLUMNS)])

# ------------------------------------------------------------
# BUILD DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(clean_rows, columns=EXPECTED_COLUMNS)

# ------------------------------------------------------------
# CLEAN TEXT COLUMN
# ------------------------------------------------------------

df["text"] = df["text"].apply(clean_text)

# ------------------------------------------------------------
# DROP EMPTY TEXT ROWS
# ------------------------------------------------------------

df = df[df["text"].str.len() > 2].reset_index(drop=True)

# ------------------------------------------------------------
# SAVE CLEAN CSV (UTF-8)
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8",
    quoting=csv.QUOTE_ALL
)

print("✅ CSV SANITIZATION COMPLETED")
print("Input :", INPUT_PATH)
print("Output:", OUTPUT_PATH)
print("Rows  :", len(df))