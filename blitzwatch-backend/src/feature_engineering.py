import pandas as pd


def label_blitz(df):
    """
    Add a binary 'blitz' column. If 'pressure' is missing, treat it as 0.

    Parameters:
    - df: DataFrame containing 'qb_hit' (and optionally 'pressure') columns

    Returns:
    - DataFrame with new 'blitz' column (1 if blitz, 0 otherwise)
    """
    df = df.copy()

    # If 'pressure' isn’t in the columns, add it filled with zeros
    if 'pressure' not in df.columns:
        df['pressure'] = 0

    # Proxy blitz: any play with qb_hit or pressure
    df['blitz'] = ((df['qb_hit'] == 1) | (df['pressure'] == 1)).astype(int)
    return df


def engineer_features(df):
    """
    Create feature set for blitz prediction.

    Parameters:
    - df: DataFrame after labeling blitz

    Returns:
    - X: DataFrame of feature columns
    - y: Series of target 'blitz'
    """
    df = df.copy()
    # Score differential
    df['score_differential'] = df['posteam_score'] - df['defteam_score']

    # Binary encode shotgun and no_huddle
    df['shotgun'] = df['shotgun'].astype(int)
    df['no_huddle'] = df['no_huddle'].astype(int)

    # Use 'qtr' instead of 'quarter' if present
    if 'quarter' in df.columns:
        df.rename(columns={'quarter': 'qtr'}, inplace=True)

    # Map pass_location and pass_length to numeric
    # pass_location: left=0, middle=1, right=2
    df['pass_location'] = df['pass_location'].map({'left': 0, 'middle': 1, 'right': 2})
    # pass_length: short=0, deep=1, null/other => -1
    df['pass_length'] = df['pass_length'].map({'short': 0, 'deep': 1}).fillna(-1).astype(int)

    # Select features
    feature_cols = [
        'down', 'ydstogo', 'yardline_100', 'qtr',
        'game_seconds_remaining', 'score_differential',
        'pass_location', 'pass_length', 'shotgun', 'no_huddle'
    ]
    # Drop rows with missing values in features or 'blitz'
    df = df.dropna(subset=feature_cols + ['blitz'])

    X = df[feature_cols]
    y = df['blitz']
    return X, y


if __name__ == '__main__':
    # Example usage given a loaded DataFrame 'df'
    from src.data_loader import load_cached_data

    # Load cached raw data
    df_raw = load_cached_data('data/raw_pass_plays.csv')
    df_labeled = label_blitz(df_raw)
    X, y = engineer_features(df_labeled)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
