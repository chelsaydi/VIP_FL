import pandas as pd

file_path = r"C:\Users\Administrator\Downloads\CellMob_AUB (1)\CellMob_AUB\CellMob\car_mekkah\car_mekkah_065.csv"
df = pd.read_csv(file_path)

print(df.head())
print(df.columns.tolist())
print(df.shape)