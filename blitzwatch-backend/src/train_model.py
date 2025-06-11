# src/train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from lightgbm import LGBMClassifier
import joblib

from src.feature_engineering import label_blitz, engineer_features
from src.data_loader import load_cached_data


def train_and_evaluate(data_path, model_path=None):
    # 1) Load & label
    df = load_cached_data(data_path)
    df = label_blitz(df)
    X, y = engineer_features(df)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3) (Optional) Compute class weight or resample …
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale = neg / pos

    # 4) Initialize LightGBM with scale_pos_weight=scale (or whatever you prefer)
    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.1,
        scale_pos_weight=scale,
    )

    # 5) Train
    model.fit(X_train, y_train)

    # 1) Compute all P(blitz) on X_test
    y_proba = model.predict_proba(X_test)[:, 1]

    # 2) Print the maximum probability so you know how high it ever gets
    print(f"Maximum P(blitz) on test set: {y_proba.max():.4f}")

    # (Optional) Print a few sample probabilities
    print("Top 5 P(blitz) values:", sorted(y_proba, reverse=True)[:5])

    # 3) Choose a new threshold based on that max
    threshold = 0.70  # for example; see explanation below
    y_pred = (y_proba > threshold).astype(int)
    
    # 7) Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy at T={threshold:.2f}: {acc:.4f}")
    print(f"Classification Report at T={threshold:.2f}:")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix at T={threshold:.2f}:")
    print(confusion_matrix(y_test, y_pred))

    # 8) Save model
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Trained model saved to {model_path}")

    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    data_path = "data/raw_pass_plays.csv"
    model_path = "models/lgbm_blitz.pkl"
    train_and_evaluate(data_path, model_path)
