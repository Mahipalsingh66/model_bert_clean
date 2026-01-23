import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

MODEL_DIR = "D:/model_bert_copy/model/model2.1"
VAL_CSV = "D:/model_bert_copy/data/gold/v2.1/val.csv"

TEXT_COL = "text"
LABEL_COL = "label"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

df = pd.read_csv(VAL_CSV)

all_preds = []
all_labels = []

print("Running evaluation...")

with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[TEXT_COL])
        label = int(row[LABEL_COL])

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

        all_preds.append(pred)
        all_labels.append(label)

acc = accuracy_score(all_labels, all_preds)

print("\n================ FINAL RESULTS ================\n")
print(f"✅ Accuracy: {acc:.4f}\n")

print("Classification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

print("Confusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
