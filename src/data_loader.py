# src/data_loader.py

import nfl_data_py as nfl
import pandas as pd

def load_pass_play_data(seasons=None):
    """
    Load play-by-play data for given seasons, filter to passing plays, and return a DataFrame.

    Parameters:
    - seasons: list of years (e.g., [2018, 2019, 2020]). 
               If None, defaults to [2018, 2019, 2020, 2021, 2022, 2023].

    Returns:
    - pandas.DataFrame with passing plays and relevant columns.
    """
    if seasons is None:
        seasons = list(range(2018, 2024))

    # Import raw play-by-play data
    df = nfl.import_pbp_data(seasons)

    # Filter to passing plays
    df = df[df['play_type'] == 'pass']

    # Some column names changed in newer versions. We'll:
    #   - Use 'qtr' instead of 'quarter'
    #   - Include 'pressure' only if it exists
    #   - Always keep 'qb_hit' (if present)

    # Build a list of columns to keep that actually exist in df.columns
    wanted = [
        'game_id',
        'play_id',
        'posteam',
        'defteam',
        'qtr',               # was 'quarter'
        'down',
        'ydstogo',
        'yardline_100',
        'game_seconds_remaining',
        'posteam_score',
        'defteam_score',
        'pass_location',
        'pass_length',
        'shotgun',
        'no_huddle',
        'qb_hit'
    ]

    # If 'pressure' exists, keep it; otherwise skip it
    if 'pressure' in df.columns:
        wanted.append('pressure')

    # Filter to the intersection of df.columns and our wanted list
    cols_to_keep = [c for c in wanted if c in df.columns]
    df = df[cols_to_keep]

    # Reset index and return
    df = df.reset_index(drop=True)
    return df


def save_raw_data(df, path):
    """
    Save the raw DataFrame to a CSV file. Useful for caching.
    """
    df.to_csv(path, index=False)


def load_cached_data(path):
    """
    Load play-by-play data from a cached CSV file.

    Parameters:
    - path: str, path to the CSV file

    Returns:
    - pandas.DataFrame loaded from CSV
    """
    return pd.read_csv(path)


if __name__ == '__main__':
    # Example usage: download and save a CSV
    seasons = list(range(2018, 2024))
    df_pass = load_pass_play_data(seasons)
    print(f"Loaded {len(df_pass)} passing plays from seasons {seasons}.")
    save_raw_data(df_pass, 'data/raw_pass_plays.csv')
    print("Saved raw passing data to data/raw_pass_plays.csv")
