
# PROGRAM 9: GEOGRAPHICAL DATA PREPARATION

import pandas as pd
from pathlib import Path

# Find project folder
project_folder = Path(__file__).resolve().parent

# Load cleaned dataset
cleaned_path = project_folder / "output" / "cleaned_accidents.csv"

df = pd.read_csv(cleaned_path)

# Select latitude and longitude columns
location_data = df[['Start_Lat', 'Start_Lng']].copy()

# Display first five locations
print("\nFIRST FIVE ACCIDENT LOCATIONS:")
print(location_data.head())

# Display dataset dimensions
print("\nLOCATION DATA SHAPE:")
print(location_data.shape)

# Check missing coordinates
print("\nMISSING COORDINATES:")
print(location_data.isnull().sum())

# Display statistical summary
print("\nLOCATION STATISTICS:")
print(location_data.describe())

print("\nGeographical data preparation completed!")
