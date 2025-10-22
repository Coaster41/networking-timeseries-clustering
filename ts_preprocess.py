import os
import numpy as np
import pandas as pd

START_SEC = [12, 27, 42, 57]
VALUE_COLUMN = "rtt"
TIME_COLUMN = "timestamp"
FREQ = "10ms"
BLOCK_SECONDS = 15

def load_ts(path, folder):
    df_hours = []
    for hour in range(24):
        # csv_file_name = f"data/2024-01-24/irtt-10ms-1h-2024-01-24-{hour:0{2}d}-00-00.csv"

        csv_file_name = f"{path}/{folder}/irtt-10ms-1h-{folder}-{hour:0{2}d}-00-00.csv"
        if not os.path.exists(csv_file_name):
            continue
        try:
            df_hour = pd.read_csv(csv_file_name)
        except Exception as e:
            print(f"Skipping {csv_file_name}: {e}")
            continue

        # Convert ns to datetime and round to 10ms bins
        df_hour[TIME_COLUMN] = pd.to_datetime(df_hour[TIME_COLUMN], unit="ns").dt.round(FREQ)
        # Filter out non-positive RTT
        df_hour = df_hour[df_hour[VALUE_COLUMN] > 0].copy()
        # Keep only relevant columns to reduce memory
        df_hour = df_hour[[TIME_COLUMN, VALUE_COLUMN]]
        df_hours.append(df_hour)
    if not df_hours:
        raise RuntimeError("No hourly CSVs loaded. Check paths.")

    df_raw = pd.concat(df_hours, ignore_index=True)
    df_raw = df_raw.drop_duplicates(subset=[TIME_COLUMN]).sort_values(TIME_COLUMN)

    # Align to first starting second in [12, 27, 42, 57]
    start_time = df_raw.loc[df_raw[TIME_COLUMN].dt.second.isin(START_SEC), TIME_COLUMN].min()
    if pd.isna(start_time):
        raise RuntimeError("Could not find a start_time matching seconds in [12, 27, 42, 57].")

    df_raw = df_raw.loc[df_raw[TIME_COLUMN] >= start_time].copy()

    # Compute 15s block id and filter last, too-short blocks
    df_raw["id"] = ((df_raw[TIME_COLUMN] - start_time).dt.total_seconds() // BLOCK_SECONDS).astype(int)
    df_raw = df_raw.loc[df_raw["id"] != df_raw["id"].max()].copy()
    # Optional: keep only substantially populated blocks (your original threshold)
    df_raw = df_raw.groupby('id').filter(lambda x: len(x) > 1300).copy()
    df_raw = df_raw[[TIME_COLUMN, VALUE_COLUMN, "id"]]
    return df_raw, start_time


def load_clusters(path, df_raw):
    # 2) Load cluster labels; only process ids that exist in this file
    clusters = pd.read_csv(path)
    # Ensure timestamp is parsed (optional but useful)
    if "timestamp" in clusters.columns:
        clusters["timestamp"] = pd.to_datetime(clusters["timestamp"])
    # Keep only fields we need
    clusters = clusters[["id", "cluster"]].drop_duplicates()

    # Restrict df_raw to ids present in clusters
    df_raw = df_raw[df_raw["id"].isin(clusters["id"])].copy()
    return df_raw, clusters


def expected_index_for_id(id_val: int, start_time) -> pd.DatetimeIndex:
    # 3) Helper: build canonical index for a given block id
    start = start_time + pd.Timedelta(seconds=BLOCK_SECONDS * id_val)
    # Inclusive end at 15s - one step (so 1500 points for 10ms frequency)
    end = start + pd.Timedelta(seconds=BLOCK_SECONDS) - pd.Timedelta(milliseconds=10)
    return pd.date_range(start, end, freq=FREQ)


# 4) Helper: nearest-neighbor imputation choosing last/next by closest distance
def nearest_neighbor_fill_uniform(s: pd.Series) -> pd.Series:
    """
    s is a Series indexed by a uniformly spaced DatetimeIndex (10ms).
    Missing values are NaN.
    We choose the nearest observed sample (prev or next). Ties: prefer previous.
    """
    n = len(s)
    mask = s.isna().values
    if not mask.any():
        return s

    pos = np.arange(n)

    # Indices of last observed sample at or before each position
    prev_obs_pos = np.where(~mask, pos, np.nan)
    prev_obs_pos = pd.Series(prev_obs_pos).ffill().values  # positions or NaN at leading missing

    # Indices of next observed sample at or after each position
    next_obs_pos = np.where(~mask, pos, np.nan)
    next_obs_pos = pd.Series(next_obs_pos).bfill().values  # positions or NaN at trailing missing

    # Distances
    # Use large numbers where no prev/next exists
    dist_prev = pos - prev_obs_pos
    dist_prev = np.where(np.isnan(dist_prev), np.inf, dist_prev)
    dist_next = next_obs_pos - pos
    dist_next = np.where(np.isnan(dist_next), np.inf, dist_next)

    # Choose prev if strictly closer or tie (<=), else next
    use_prev = dist_prev <= dist_next

    s_ffill = s.ffill()
    s_bfill = s.bfill()

    filled_values = np.where(use_prev, s_ffill.values, s_bfill.values)

    out = s.copy()
    out[mask] = filled_values[mask]
    return out


def process_sequence(df_raw, start_time, clusters):
    # 5) Process each id into aligned sequences (long format)
    records = []
    feature_rows = []

    grouped = df_raw.groupby("id", sort=True)
    for id_val, grp in grouped:
        # Compute canonical index
        idx = expected_index_for_id(id_val, start_time)

        # Align this group's RTT to the canonical timeline
        s = grp.set_index(TIME_COLUMN)[VALUE_COLUMN].sort_index()
        s = s.reindex(idx)  # Missing samples become NaN
        # Token with -1 for missing
        s_token = s.fillna(-1)

        # Nearest neighbor fill (last or next whichever is closer; ties -> last)
        s_nearest = nearest_neighbor_fill_uniform(s)

        # Missing mask (based on original aligned series)
        missing_mask = s.isna()

        # Cluster label
        cluster = clusters.loc[clusters["id"] == id_val, "cluster"]
        cluster = int(cluster.iloc[0]) if len(cluster) else None

        # Build long-form rows
        df_block = pd.DataFrame({
            "id": id_val,
            "timestamp": idx,
            "rtt": s.values,
            "rtt_token": s_token.values,      # -1 indicates missing
            "rtt_nearest": s_nearest.values,  # nearest neighbor imputation
            "is_missing": missing_mask.values
        })

        # Append cluster if present
        df_block["cluster"] = cluster

        records.append(df_block)

        # Compute simple per-block features (good for training/eval)
        # Feel free to expand with more robust statistics
        valid = ~missing_mask
        valid_vals = s[valid]
        features = {
            "id": id_val,
            "cluster": cluster,
            "samples_expected": len(s),
            "samples_observed": int(valid.sum()),
            "missing_count": int(missing_mask.sum()),
            "missing_ratio": float(missing_mask.mean()),
            "rtt_mean": float(valid_vals.mean()) if len(valid_vals) else np.nan,
            "rtt_std": float(valid_vals.std(ddof=1)) if len(valid_vals) > 1 else np.nan,
            "rtt_median": float(valid_vals.median()) if len(valid_vals) else np.nan,
            "rtt_p10": float(valid_vals.quantile(0.10)) if len(valid_vals) else np.nan,
            "rtt_p90": float(valid_vals.quantile(0.90)) if len(valid_vals) else np.nan,
            # Compare imputation behavior
            "nearest_fill_delta_mean": float((s_nearest - s).abs().mean(skipna=True)),
            "nearest_fill_delta_max": float((s_nearest - s).abs().max(skipna=True)),
        }
        feature_rows.append(features)

    # Concatenate and save
    df_long = pd.concat(records, ignore_index=True)
    df_features = pd.DataFrame(feature_rows)
    return df_long, df_features


def save_sequence(df_long, path):
    df_long_clean = df_long[["id", "timestamp", "rtt_token", "rtt_nearest"]]
    df_long_clean.to_csv(f"{path}/clustering_data.csv", float_format="%.4f", index=False)
