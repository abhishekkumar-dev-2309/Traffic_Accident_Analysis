
# PROGRAM 4: ACCIDENT DATA EXPLORATION

import pandas as pd
from pathlib import Path

# Step 1: Define dataset path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "US_Accidents_March23.csv"
)

# Step 2: Load first 1000 records

data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

print("========== ACCIDENT DATA EXPLORATION ==========")

# Step 3: Severity analysis

print("\n1. ACCIDENT SEVERITY DISTRIBUTION")

severity_count = data["Severity"].value_counts().sort_index()

print(severity_count)

# Step 4: City-wise accident analysis

print("\n2. TOP 10 CITIES WITH MOST ACCIDENTS")

city_count = data["City"].value_counts().head(10)

print(city_count)

# Step 5: State-wise accident analysis

print("\n3. TOP 10 STATES WITH MOST ACCIDENTS")

state_count = data["State"].value_counts().head(10)

print(state_count)

# Step 6: Weather condition analysis

print("\n4. TOP 10 WEATHER CONDITIONS")

weather_count = (
    data["Weather_Condition"]
    .value_counts()
    .head(10)
)

print(weather_count)

# Step 7: Day and night accident analysis

print("\n5. DAY VS NIGHT ACCIDENTS")

day_night = data["Sunrise_Sunset"].value_counts()

print(day_night)

# Step 8: Road feature analysis

print("\n6. TRAFFIC SIGNAL RELATED ACCIDENTS")

print(data["Traffic_Signal"].value_counts())

print("\n7. JUNCTION RELATED ACCIDENTS")

print(data["Junction"].value_counts())

# Step 9: Final summary

print("\n========== ANALYSIS COMPLETED ==========")

print("Total records analyzed:", len(data))

print("\nProgram 4 completed successfully!")
