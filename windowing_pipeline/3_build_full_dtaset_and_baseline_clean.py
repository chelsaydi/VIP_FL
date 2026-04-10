import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.preprocessing import LabelEncoder
base_path = r"C:\Users\Administrator\Downloads\CellMob_AUB (1)\CellMob_AUB\CellMob"

window_size = 100
stride = 50

X_all = []
y_all = []

def get_label(folder_name):
    folder_name = folder_name.lower()
    if folder_name.startswith("car"):
        return "car"
    elif folder_name.startswith("bus"):
        return "bus"
    elif folder_name.startswith("walk"):
        return "walk"
    elif folder_name.startswith("train"):
        return "train"
    else:
        return None

# ---------- first pass: find common columns ----------
all_column_sets = []

for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path, nrows=5, low_memory=False)
            cols = set(df.columns)
            cols.discard("Time")
            all_column_sets.append(cols)
        except Exception as e:
            print(f"Error reading columns from {file_name}: {e}")

common_columns = sorted(set.intersection(*all_column_sets))
print("Number of common columns:", len(common_columns))

# ---------- second pass: process files using only common columns ----------
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    if not os.path.isdir(folder_path):
        continue

    label = get_label(folder)
    if label is None:
        continue

    print(f"\nProcessing folder: {folder} -> label: {label}")

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(folder_path, file_name)

        try:
            df = pd.read_csv(file_path, low_memory=False)

            # keep only the same columns in all files
            df = df.reindex(columns=common_columns)

            # convert to numeric
            df = df.apply(pd.to_numeric, errors="coerce")

            # fill missing values
            df = df.fillna(0)

            if len(df) < window_size:
                continue

            for start in range(0, len(df) - window_size + 1, stride):
                end = start + window_size
                window = df.iloc[start:end].values

                mean_vals = np.mean(window, axis=0)
                std_vals  = np.std(window, axis=0)
                min_vals  = np.min(window, axis=0)
                max_vals  = np.max(window, axis=0)

                feature_vector = np.concatenate([mean_vals, std_vals, min_vals, max_vals])

                X_all.append(feature_vector)
                y_all.append(label)

        except Exception as e:
            print(f"Error in file {file_name}: {e}")

X_all = np.array(X_all)
y_all = np.array(y_all)

print("\nDone.")
print("X_all shape:", X_all.shape)
print("y_all shape:", y_all.shape)

unique, counts = np.unique(y_all, return_counts=True)
print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"{u}: {c}") 
dataset_df = pd.DataFrame(X_all)
dataset_df["label"] = y_all
dataset_df.to_csv("cellmob_windowed_dataset.csv", index=False)

print("Saved as cellmob_windowed_dataset.csv") 


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_all)

print("Label mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{cls} -> {i}") 

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))