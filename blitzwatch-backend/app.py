# blitzwatch-backend/app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib, pandas as pd, io, matplotlib.pyplot as plt, shap, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# adjust these imports if your package layout differs
from src.data_loader import load_cached_data
from src.feature_engineering import label_blitz, engineer_features

app = Flask(__name__)
CORS(app)  # allow React localhost to call this

# load your trained model once
MODEL = joblib.load("models/lgbm_blitz.pkl")


# 1) Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Build a DataFrame just like manual_predict.py
    df = pd.DataFrame(
        [
            {
                "down": data["down"],
                "ydstogo": data["ydstogo"],
                "yardline_100": data["yardline_100"],
                "qtr": data["qtr"],
                "game_seconds_remaining": data["min_left"] * 60 + data["sec_left"],
                "posteam_score": data["posteam_score"],
                "defteam_score": data["defteam_score"],
                "pass_location": data["pass_location"],
                "pass_length": data["pass_length"],
                "shotgun": int(data["shotgun"]),
                "no_huddle": int(data["no_huddle"]),
            }
        ]
    )
    # engineer features
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]
    df["pass_location"] = df["pass_location"].map({"left": 0, "middle": 1, "right": 2})
    df["pass_length"] = (
        df["pass_length"].map({"short": 0, "deep": 1}).fillna(-1).astype(int)
    )
    feature_cols = [
        "down",
        "ydstogo",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
        "pass_location",
        "pass_length",
        "shotgun",
        "no_huddle",
    ]
    X_live = df[feature_cols]
    proba = MODEL.predict_proba(X_live)[0][1]
    return jsonify({"proba": proba})


# Helper to render a Matplotlib figure as PNG
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


# 2) Feature importance plot
@app.route("/insights/feature_importance")
def feature_importance():
    df_raw = load_cached_data("data/raw_pass_plays.csv")
    df_lab = label_blitz(df_raw)
    X, _ = engineer_features(df_lab)
    importances = MODEL.feature_importances_
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances, y=X.columns, ax=ax)
    ax.set_title("Feature Importances")
    return send_file(fig_to_png_bytes(fig), mimetype="image/png")


# 3) SHAP summary plot
@app.route("/insights/shap_summary")
def shap_summary():
    df_raw = load_cached_data("data/raw_pass_plays.csv")
    df_lab = label_blitz(df_raw)
    X, _ = engineer_features(df_lab)
    explainer = shap.TreeExplainer(MODEL)
    shap_vals = explainer.shap_values(X)
    fig = plt.figure()
    shap.summary_plot(shap_vals, X, show=False)
    return send_file(fig_to_png_bytes(fig), mimetype="image/png")


# 4) Confusion matrix plot
@app.route("/insights/confusion_matrix")
def conf_matrix():
    df_raw = load_cached_data("data/raw_pass_plays.csv")
    df_lab = label_blitz(df_raw)
    X, y = engineer_features(df_lab)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = MODEL.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return send_file(fig_to_png_bytes(fig), mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
