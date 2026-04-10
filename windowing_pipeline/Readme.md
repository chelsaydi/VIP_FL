\# VIP Dataset Windowing



This project preprocesses the CellMob dataset for mobility classification.



\## Files

\- `1\_inspect\_onecsv\_clean.py`: inspect one CSV file and print columns and shape

\- `2\_clean\_window\_one\_file\_clean.py`: clean one file, apply windowing, and extract features

\- `3\_build\_full\_dataset\_and\_baseline\_clean.py`: build the full common dataset from all folders and train a centralized baseline



\## Method

\- Common dataset: CellMob

\- Common columns across all files: 53

\- Window size: 100

\- Stride: 50

\- Features per window: mean, std, min, max

\- Final classes: bus, car, train, walk



\## Result

\- Total samples: 81430

\- Features per sample: 212

\- Centralized baseline accuracy: about 98.97%

