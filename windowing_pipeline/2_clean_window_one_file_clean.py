import pandas as pd
import numpy as np

file_path = r"C:\Users\Administrator\Downloads\CellMob_AUB (1)\CellMob_AUB\CellMob\car_mekkah\car_mekkah_065.csv"
df = pd.read_csv(file_path, low_memory=False)

# remove Time for now
df = df.drop(columns=["Time"], errors="ignore")

# convert all remaining columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# fill missing values
df = df.fillna(0)

print(df.head())
print(df.shape) 
window_size = 100
stride = 50

windows = []

for start in range(0, len(df) - window_size + 1, stride):
    end = start + window_size
    window = df.iloc[start:end].values
    windows.append(window)

import numpy as np
windows = np.array(windows)

print("Windows shape:", windows.shape)
feature_windows = []

for window in windows:
    mean_vals = np.mean(window, axis=0)
    std_vals  = np.std(window, axis=0)
    min_vals  = np.min(window, axis=0)
    max_vals  = np.max(window, axis=0)
    
    feature_vector = np.concatenate([mean_vals, std_vals, min_vals, max_vals])
    feature_windows.append(feature_vector)

feature_windows = np.array(feature_windows)

print("Feature windows shape:", feature_windows.shape)
print("First sample shape:", feature_windows[0].shape)
labels = ["car"] * len(feature_windows)

print("Labels count:", len(labels))
print("First 5 labels:", labels[:5])