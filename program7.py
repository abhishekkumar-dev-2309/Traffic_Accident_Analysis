
# ==========================================
# PROGRAM 8: BASIC DATA CLEANING
# ==========================================

import pandas as pd
from pathlib import Path

# Find project directory
project_folder = Path(__file__).resolve().parent

# Define dataset path
dataset_path = project_folder / "dataset" / "US_Accidents_March23.csv"

# Define output path
output_folder = project_folder / "output"
output_folder.mkdir(exist_ok=True)

# Load first 1000 records
df = pd.read_csv(dataset_path, nrows=1000)

# Display original dataset size
print("Original Dataset Shape:", df.shape)

# Step 1: Remove completely empty columns
df = df.dropna(axis=1, how='all')

print("\nAfter removing empty columns:")
print(df.shape)

# Step 2: Remove records without accident coordinates
df = df.dropna(subset=['Start_Lat', 'Start_Lng'])

print("\nAfter checking coordinates:")
print(df.shape)

# Step 3: Remove duplicate records using ID
df = df.drop_duplicates(subset=['ID'])

print("\nAfter removing duplicate records:")
print(df.shape)

# Step 4: Save cleaned dataset
cleaned_path = output_folder / "cleaned_accidents.csv"
df.to_csv(cleaned_path, index=False)

print("\nCleaned dataset saved successfully!")
print("File location:", cleaned_path)
