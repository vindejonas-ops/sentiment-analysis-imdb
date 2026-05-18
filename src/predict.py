"""
Inference script.
Usage: python predict.py "This movie is fantastic!"
"""

import os
import sys
import numpy as np
import joblib
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLP_PATH = os.path.join(BASE_DIR, "model", "sentiment_mlp.joblib")


def extract_single(text, tokenizer, model):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mean_pooled = hidden.mean(dim=1)
        max_pooled = hidden.max(dim=1)[0]
        return torch.cat([mean_pooled, max_pooled], dim=-1).cpu().numpy()


def predict(text):
    if not os.path.exists(MLP_PATH):
        print(f"Error: Model not found at {MLP_PATH}")
        sys.exit(1)
    
    mlp = joblib.load(MLP_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    features = extract_single(text, tokenizer, model)
    prob = mlp.predict_proba(features)[0]
    pred = mlp.predict(features)[0]
    
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    
    print(f"\nText: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
    return sentiment, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your text here\"")
        sys.exit(1)
    
    predict(sys.argv[1])