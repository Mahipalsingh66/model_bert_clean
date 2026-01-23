from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_DIR = "D:/model_bert_copy/model/model2.1"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model.to(device)
model.eval()

print("✅ Model and tokenizer loaded successfully on", device)
