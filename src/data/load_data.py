import os
import pandas as pd


def build_file_index(root):
    """
    Scan dataset folders and build an index of all CSV files.

    Assumes folder names follow:
    <label>_..._<city>

    Example:
    walking_something_jeddah
    -> label = walking
    -> city = jeddah
    """

    rows = []

    if not os.path.exists(root):
        raise FileNotFoundError(f"Root path does not exist: {root}")

    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)

        if not os.path.isdir(folder_path):
            continue

        parts = folder.split("_")

        # Robust extraction (works even if format slightly varies)
        label = parts[0]
        city = parts[-1]

        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                rows.append({
                    "folder": folder,
                    "file": file,
                    "city": city,
                    "label": label,
                    "path": os.path.join(folder_path, file)
                })

    df_index = pd.DataFrame(rows)
    return df_index


def print_index_summary(df_index):
    print("Number of CSV files:", len(df_index))

    if len(df_index) == 0:
        print("No files found.")
        return

    print("\nCities:")
    print(sorted(df_index["city"].unique()))

    print("\nLabels:")
    print(sorted(df_index["label"].unique()))

    print("\nFiles per city:")
    print(df_index["city"].value_counts())

    print("\nFiles per label:")
    print(df_index["label"].value_counts())