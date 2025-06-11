# src/visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.train_model import train_and_evaluate
from src.feature_engineering import label_blitz, engineer_features
from src.data_loader import load_cached_data

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_feature_importance(model, X_train, save_path=None, top_n=10):
    importances = model.feature_importances_
    names = X_train.columns
    feat_imp = pd.Series(importances, index=names).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp[:top_n], y=feat_imp.index[:top_n])
    plt.title('Top Feature Importances')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_shap_summary(model, X, save_path=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def run_all_plots(data_path, model, X_test, y_test, y_pred):
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Plot feature importance using training data (reload X)
    df = load_cached_data(data_path)
    df = label_blitz(df)
    X, y = engineer_features(df)
    plot_feature_importance(model, X)

    # Plot SHAP summary on test set
    plot_shap_summary(model, X_test)

if __name__ == '__main__':
    data_path = 'data/raw_pass_plays.csv'
    # Retrain (or load) the model and get test splits
    model, X_test, y_test, y_pred = train_and_evaluate(data_path)
    run_all_plots(data_path, model, X_test, y_test, y_pred)
