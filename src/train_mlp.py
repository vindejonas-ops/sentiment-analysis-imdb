"""
MLP classifier for DeBERTa features.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(BASE_DIR, "model", "sentiment_mlp.joblib")
ACCURACY_PATH = os.path.join(BASE_DIR, "accuracy.txt")


def main():
    X = np.load(os.path.join(FEATURES_DIR, "X_features.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "y_labels.npy"))
    
    print(f"Features shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-4,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
        tol=1e-4
    )
    
    print("Training MLP...")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    
    joblib.dump(clf, MODEL_PATH)
    
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(acc))
    
    if acc < 0.95:
        print(f"WARNING: Accuracy {acc:.4f} below threshold")
        sys.exit(1)
    else:
        print(f"✅ Accuracy meets threshold (>= 0.95)")


if __name__ == "__main__":
    main()