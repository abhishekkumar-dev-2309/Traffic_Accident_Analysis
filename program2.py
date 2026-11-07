
# PROGRAM 2: DATASET STRUCTURE ANALYSIS

import pandas as pd
from pathlib import Path

# Step 1: Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "US_Accidents_March23.csv"

# Step 2: Load first 1000 records
data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

print("========== DATASET STRUCTURE ==========")

# Step 3: Dataset dimensions
print("\nDataset Shape:", data.shape)

# Step 4: Column names
print("\nColumn Names:")
for number, column in enumerate(data.columns, start=1):
    print(f"{number}. {column}")

# Step 5: Data types
print("\n========== COLUMN DATA TYPES ==========")
print(data.dtypes)

# Step 6: Count columns by data type
print("\n========== DATA TYPE COUNTS ==========")
print(data.dtypes.value_counts())

# Step 7: Missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum().sort_values(ascending=False))

# Step 8: Duplicate records
print("\nDuplicate Rows:", data.duplicated().sum())

# Step 9: Unique accident IDs
print("Unique Accident IDs:", data["ID"].nunique())

print("\nProgram 2 completed successfully!")
