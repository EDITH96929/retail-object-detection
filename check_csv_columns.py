import pandas as pd

# Load the CSV
df = pd.read_csv('data/raw/SKU110K_fixed/annotations/annotations_train.csv')

print("CSV Columns:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nDataFrame shape:")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")