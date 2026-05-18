"""
Feature extraction using DeBERTa-v3-small.
Extracts mean+max pooled features for MLP training.
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuration
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LENGTH = 512
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FEATURES_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(FEATURES_DIR, exist_ok=True)


def load_data():
    """Load IMDB dataset from local CSV or HuggingFace."""
    local_files = [
        os.path.join(DATA_DIR, "imdb_balanced_10k.csv"),
        os.path.join(DATA_DIR, "imdb_top_500.csv"),
    ]
    
    for f in local_files:
        if os.path.exists(f):
            print(f"Loading local data: {f}")
            return pd.read_csv(f)
    
    print("Downloading IMDB from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("imdb")
    train = ds["train"].to_pandas()
    
    df = pd.concat([
        train[train["label"] == 0].sample(5000, random_state=42),
        train[train["label"] == 1].sample(5000, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "imdb_balanced_10k.csv"), index=False)
    return df


def extract_features(texts, tokenizer, model, batch_size=BATCH_SIZE):
    """Extract mean+max pooled DeBERTa features."""
    features = []
    model.eval()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting features"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(DEVICE)
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            
            mean_pooled = hidden_states.mean(dim=1)
            max_pooled = hidden_states.max(dim=1)[0]
            pooled = torch.cat([mean_pooled, max_pooled], dim=-1)
            
            features.append(pooled.cpu().numpy())
    
    return np.vstack(features)


def main():
    print(f"Using device: {DEVICE}")
    
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    
    texts = df["text"].values
    X = extract_features(texts, tokenizer, model)
    y = df["label"].values
    
    print(f"Features shape: {X.shape}")
    
    np.save(os.path.join(FEATURES_DIR, "X_features.npy"), X)
    np.save(os.path.join(FEATURES_DIR, "y_labels.npy"), y)
    print(f"Saved to {FEATURES_DIR}")


if __name__ == "__main__":
    main()