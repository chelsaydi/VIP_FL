# CellMob Windowing Pipeline

This folder contains the preprocessing and baseline scripts for the CellMob common dataset.

## Files
- `1_inspect_onecsv_clean.py`: inspect one CSV file and print columns and shape
- `2_clean_window_one_file_clean.py`: clean one file, apply windowing, and extract features
- `3_build_full_dataset_and_baseline_clean.py`: build the full common dataset from all folders and train a centralized baseline

## Method
- Common dataset: CellMob
- Common columns across all files: 53
- Window size: 100
- Stride: 50
- Features per window: mean, std, min, max
- Final classes: bus, car, train, walk

## Result
- Total samples: 81430
- Features per sample: 212
- Centralized baseline accuracy: about 98.97%
