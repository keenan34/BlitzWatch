import sys
import joblib
import pandas as pd

# Load trained LightGBM model
try:
    MODEL = joblib.load("models/lgbm_blitz.pkl")
except FileNotFoundError:
    print(
        "Error: Trained model not found at models/lgbm_blitz.pkl. Please run 'python -m src.train_model' first."
    )
    sys.exit(1)


def prompt_int(prompt, min_val=None, max_val=None):
    """
    Prompt user for an integer within optional bounds.
    """
    while True:
        try:
            val = int(input(f"{prompt}: ").strip())
            if (min_val is not None and val < min_val) or (
                max_val is not None and val > max_val
            ):
                print(f" → Please enter an integer between {min_val} and {max_val}.")
                continue
            return val
        except ValueError:
            print(" → Invalid integer. Try again.")


def prompt_choice(prompt, options):
    """
    Prompt user to choose one value from the list of options.
    """
    opts_str = "/".join(options)
    options_lower = [o.lower() for o in options]
    while True:
        val = input(f"{prompt} ({opts_str}): ").strip().lower()
        if val in options_lower:
            # Return original casing from options list
            return options[options_lower.index(val)]
        print(f" → Choose one of: {opts_str}")


def main():
    print("\n=== BlitzWatch: Manual Pre-Snap Predictor ===\n")

    # Collect numeric inputs
    down = prompt_int("Down (1–4)", 1, 4)
    ydstogo = prompt_int("Yards to Go (1–99)", 1, 99)
    yardline = prompt_int("Yardline (yards from opponent end zone, 1–99)", 1, 99)
    qtr = prompt_int("Quarter (1–4)", 1, 4)

    min_left = prompt_int("Minutes left in Quarter (0–15)", 0, 15)
    sec_left = prompt_int("Seconds left in Quarter (0–59)", 0, 59)
    game_seconds_remaining = min_left * 60 + sec_left

    posteam_score = prompt_int("Your team’s current score", 0, 100)
    defteam_score = prompt_int("Opponent’s current score", 0, 100)

    # Collect categorical inputs
    pass_location = prompt_choice(
        "Pass Location", ["left", "middle", "right"]
    )  # returns exactly 'left','middle','right'
    pass_length_choice = prompt_choice("Pass Length", ["short", "deep", "none"])
    pass_length = pass_length_choice if pass_length_choice != "none" else None

    shotgun_input = prompt_choice("Shotgun formation?", ["yes", "no"])
    shotgun = True if shotgun_input.lower() == "yes" else False

    nohuddle_input = prompt_choice("No-Huddle offense?", ["yes", "no"])
    no_huddle = True if nohuddle_input.lower() == "yes" else False

    # Build feature dict
    features = {
        "down": down,
        "ydstogo": ydstogo,
        "yardline_100": yardline,
        "qtr": qtr,
        "game_seconds_remaining": game_seconds_remaining,
        "posteam_score": posteam_score,
        "defteam_score": defteam_score,
        "pass_location": pass_location,
        "pass_length": pass_length,
        "shotgun": shotgun,
        "no_huddle": no_huddle,
    }

    # Transform features as in inference
    df = pd.DataFrame([features])
    df["score_differential"] = df["posteam_score"] - df["defteam_score"]
    df["shotgun"] = df["shotgun"].astype(int)
    df["no_huddle"] = df["no_huddle"].astype(int)
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

    # Predict
    proba = MODEL.predict_proba(X_live)[0][1]

    # Print with a nice message
    print("\n--- Prediction Result ---")
    print(f"Blitz Probability: {proba:.2%}")
    if proba > 0.50:
        print("→ Model suggests a BLITZ.")
    else:
        print("→ Model suggests NO BLITZ.")
    print("--------------------------\n")


if __name__ == "__main__":
    main()
