
# PROGRAM 8: DATA CLEANING AND PREPROCESSING

import pandas as pd
from pathlib import Path

# Step 1: Define project paths

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "US_Accidents_March23.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 2: Load dataset

print("Loading accident dataset...")

data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

print("\nOriginal Dataset Shape:", data.shape)

# Step 3: Check missing values

print("\n========== MISSING VALUES BEFORE CLEANING ==========")

print(
    data.isnull()
    .sum()
    .sort_values(ascending=False)
)

# Step 4: Remove columns with 100% missing values

missing_percentage = data.isnull().mean() * 100

columns_to_drop = missing_percentage[
    missing_percentage == 100
].index

data.drop(
    columns=columns_to_drop,
    inplace=True
)

print("\nColumns removed:", list(columns_to_drop))

# Step 5: Remove duplicate records

duplicate_count = data.duplicated().sum()

print("\nDuplicate records:", duplicate_count)

data.drop_duplicates(inplace=True)

# Step 6: Handle missing values

# Numerical columns: fill with median

numerical_columns = data.select_dtypes(
    include=["number"]
).columns

for column in numerical_columns:

    data[column] = data[column].fillna(
        data[column].median()
    )

# Categorical columns: fill with mode

categorical_columns = data.select_dtypes(
    include=["object", "str"]
).columns

for column in categorical_columns:

    mode_values = data[column].mode()

    if not mode_values.empty:
        data[column] = data[column].fillna(
            mode_values.iloc[0]
        )

# Step 7: Check missing values after cleaning

print("\n========== MISSING VALUES AFTER CLEANING ==========")

print(data.isnull().sum())

# Step 8: Display cleaned dataset information

print("\n========== CLEANED DATASET INFORMATION ==========")

print("Original columns:", len(missing_percentage))

print("Cleaned columns:", len(data.columns))

print("Remaining records:", len(data))

print("\nFinal Dataset Shape:", data.shape)

# Step 9: Save cleaned dataset

output_file = OUTPUT_DIR / "cleaned_accidents.csv"

data.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved successfully!")

print("Output:", output_file)

print("\nProgram 8 completed successfully!")
