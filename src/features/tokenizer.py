# tokenizer.py
from transformers import AutoTokenizer

MODEL_NAME = "xlm-roberta-base"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    return tokenizer
