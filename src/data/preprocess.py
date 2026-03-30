import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder


def find_common_columns(df_index):
    column_counter = Counter()
    file_column_counts = []

    for path in df_index["path"]:
        try:
            df_tmp = pd.read_csv(path, nrows=1, low_memory=False)
            cols = list(df_tmp.columns)

            file_column_counts.append({
                "path": path,
                "n_columns": len(cols)
            })

            for c in cols:
                column_counter[c] += 1

        except Exception as e:
            print(f"Error reading {path}: {e}")

    df_columns = pd.DataFrame({
        "column": list(column_counter.keys()),
        "appears_in_n_files": list(column_counter.values())
    }).sort_values(by=["appears_in_n_files", "column"], ascending=[False, True])

    df_file_cols = pd.DataFrame(file_column_counts)

    print("Total unique columns across all files:", len(df_columns))
    print("\nDifferent file widths found:")
    print(df_file_cols["n_columns"].value_counts().sort_index())

    all_file_cols = df_columns[df_columns["appears_in_n_files"] == len(df_index)]

    print("\nNumber of columns appearing in all files:", len(all_file_cols))

    common_columns = all_file_cols["column"].tolist()
    return common_columns


def build_clean_dataset(df_index, common_columns):
    clean_dfs = []

    for _, row in df_index.iterrows():
        try:
            df = pd.read_csv(row["path"], low_memory=False)

            df = df[common_columns].copy()

            df["city"] = row["city"]
            df["label"] = row["label"]
            df["source_file"] = row["file"]

            clean_dfs.append(df)

        except Exception as e:
            print(f"Error processing {row['path']}: {e}")

    df_all = pd.concat(clean_dfs, ignore_index=True)

    print("Combined shape:", df_all.shape)
    print("\nColumns:")
    print(df_all.columns.tolist())

    print("\nCities in combined data:")
    print(df_all["city"].value_counts())

    print("\nLabels in combined data:")
    print(df_all["label"].value_counts())

    return df_all


def preprocess_features(df_all):
    feature_cols = [c for c in df_all.columns if c not in ["city", "label", "source_file"]]

    for col in feature_cols:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all[feature_cols] = df_all[feature_cols].fillna(0)

    print("Remaining NaNs:", df_all[feature_cols].isna().sum().sum())

    return df_all, feature_cols


def encode_labels(df_all):
    le = LabelEncoder()
    df_all["label_encoded"] = le.fit_transform(df_all["label"])

    print("Label mapping:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))

    return df_all, le


def create_sliding_windows(df_all, feature_cols, window_size=50, stride=25):
    """
    Create sliding windows separately within each source file.
    Each window gets:
    - flattened features
    - one encoded label (majority vote)
    - city
    - source_file

    Parameters
    ----------
    df_all : pd.DataFrame
        Preprocessed full dataframe containing features + metadata
    feature_cols : list
        Feature column names
    window_size : int
        Number of rows per window
    stride : int
        Step size between consecutive windows

    Returns
    -------
    X_windows : np.ndarray
        Shape: (n_windows, window_size * n_features)
    y_windows : np.ndarray
        Shape: (n_windows,)
    meta_df : pd.DataFrame
        Metadata for each window
    """
    X_windows = []
    y_windows = []
    meta_rows = []

    grouped = df_all.groupby("source_file")

    total_windows = 0
    skipped_files = 0

    for source_file, df_file in grouped:
        df_file = df_file.reset_index(drop=True)

        if len(df_file) < window_size:
            skipped_files += 1
            continue

        city = df_file["city"].iloc[0]

        features_array = df_file[feature_cols].to_numpy(dtype=np.float32)
        labels_array = df_file["label_encoded"].to_numpy()

        for start in range(0, len(df_file) - window_size + 1, stride):
            end = start + window_size

            window_features = features_array[start:end]
            window_labels = labels_array[start:end]

            # Majority vote label for the window
            values, counts = np.unique(window_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]

            # Flatten window for feedforward model
            flattened_window = window_features.reshape(-1)

            X_windows.append(flattened_window)
            y_windows.append(majority_label)
            meta_rows.append({
                "city": city,
                "source_file": source_file,
                "start_idx": start,
                "end_idx": end
            })

            total_windows += 1

    X_windows = np.array(X_windows, dtype=np.float32)
    y_windows = np.array(y_windows, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    print("\nSliding window summary:")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")
    print(f"Total windows created: {total_windows}")
    print(f"Files skipped for being shorter than window_size: {skipped_files}")

    if len(meta_df) > 0:
        print("\nWindows per city:")
        print(meta_df["city"].value_counts())

        print("\nWindow label distribution:")
        print(pd.Series(y_windows).value_counts().sort_index())

        print("\nX_windows shape:", X_windows.shape)
        print("y_windows shape:", y_windows.shape)

    return X_windows, y_windows, meta_df
