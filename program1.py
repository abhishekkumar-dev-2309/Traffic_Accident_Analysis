
# -----------------------------------------
# PROGRAM 1: DATASET LOADING
# -----------------------------------------

import pandas as pd
from pathlib import Path

# Step 1: Define dataset path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "US_Accidents_March23.csv"
)

# Step 2: Check dataset availability

if not DATASET_PATH.exists():
    print("Dataset file not found!")
    raise FileNotFoundError(DATASET_PATH)

print("Dataset found successfully!")

# Step 3: Load first 1000 records

print("\nLoading accident dataset...")

data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

# Step 4: Display dataset information

print("\n========== DATASET INFORMATION ==========")

print("Total rows:", len(data))

print("Total columns:", len(data.columns))

print("\nColumn names:")

print(data.columns.tolist())

# Step 5: Display first five records

print("\n========== FIRST FIVE RECORDS ==========")

print(data.head())

# Step 6: Display dataset structure

print("\n========== DATASET STRUCTURE ==========")

data.info()

print("\nProgram 1 completed successfully!")
