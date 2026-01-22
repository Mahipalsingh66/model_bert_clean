import pandas as pd
import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
CSV_PATH = "D:/model_bert_copy/data/raw/file.csv"   # change if needed
TEXT_COLUMN = "text"              # ⚠️ change if your column name is different
MAX_ROWS = 50000

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

CONF_THRESHOLD = 0.70
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
# ---------------------------------------- #

# Load data
df = pd.read_csv(CSV_PATH)

# Basic cleaning
df = df[[TEXT_COLUMN]].dropna()
df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42).reset_index(drop=True)

# Load sentiment pipeline
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=0 if torch.cuda.is_available() else -1
)

# Label mapping
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

def assign_label(result):
    label = LABEL_MAP[result["label"]]
    score = result["score"]

    if label in ["positive", "negative"] and score >= CONF_THRESHOLD:
        return label
    return "neutral"

# Run inference
labels = []
scores = []

for text in tqdm(df[TEXT_COLUMN], desc="Labelling sentiment"):
    result = sentiment_pipe(
    text,
    truncation=True,
    max_length=512
)[0]
  # truncate long text safely
    labels.append(assign_label(result))
    scores.append(result["score"])

df["label"] = labels
df["confidence"] = scores

# ---------------- SPLIT DATA ---------------- #
train_df, temp_df = train_test_split(
    df, test_size=(1 - TRAIN_RATIO), stratify=df["label"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
    stratify=temp_df["label"],
    random_state=42
)

# ---------------- SAVE ---------------- #
train_df.to_csv("D:/model_bert_copy/data_create/train.csv", index=False)
val_df.to_csv("D:/model_bert_copy/data_create/validation.csv", index=False)
test_df.to_csv("D:/model_bert_copy/data_create/test.csv", index=False)

print("✅ Dataset created successfully")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))
print("\nLabel distribution:")
print(df["label"].value_counts())
