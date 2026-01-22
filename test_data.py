# import pandas as pd
# df = pd.read_csv("/media/data/7b192233-93d1-41b0-8a9a-4b626a44281a/project_bert/data/feedback.csv")
# print(df)
from src.features.tokenizer import load_tokenizer
from src.data.loader import SentimentDataset

tokenizer = load_tokenizer()

dataset = SentimentDataset(
    csv_path="data/processed/train.csv",
    tokenizer=tokenizer,
    max_length=128
)

print("Samples:", len(dataset))
sample = dataset[0]
print(sample.keys())
print(sample["input_ids"].shape)
print(sample["labels"])