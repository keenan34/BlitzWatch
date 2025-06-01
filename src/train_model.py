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
    """
    Load data, train a LightGBM classifier, evaluate it, and optionally save the model.

    Returns:
    - model: Trained LightGBM model
    - X_test, y_test, y_pred: Test features, true labels, and predictions
    """
    # Load cached data
    df = load_cached_data(data_path)

    # Label blitz plays
    df = label_blitz(df)

    # Extract features and target
    X, y = engineer_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Compute ratio of negatives to positives
    pos = sum(y_train == 1)
    neg = sum(y_train == 0)
    scale = neg / pos  # roughly 10 in your data
    
    model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    n_estimators=100,
    learning_rate=0.1,
    scale_pos_weight=scale
)

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model if path provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Trained model saved to {model_path}")

    return model, X_test, y_test, y_pred


if __name__ == '__main__':
    # Example usage
    data_path = 'data/raw_pass_plays.csv'
    model_path = 'models/lgbm_blitz.pkl'
    train_and_evaluate(data_path, model_path)
