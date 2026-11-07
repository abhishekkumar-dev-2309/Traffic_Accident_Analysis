
import pandas as pd
import numpy as np
import os

# -----------------------------------------
# STEP 1: Define file paths
# -----------------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"

output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

output_file = os.path.join(
    output_folder,
    "accident_prediction_dataset.csv"
)

# -----------------------------------------
# STEP 2: Select useful columns
# -----------------------------------------

selected_columns = [
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
    "Sunrise_Sunset",
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity"
]

# -----------------------------------------
# STEP 3: Read dataset in chunks
# -----------------------------------------

chunksize = 100000
sample_size = 5000

total_records = 0
sample_number = 0

# Remove old output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)

print("Starting dataset preparation...\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(
        file_path,
        usecols=selected_columns,
        chunksize=chunksize
    )
):

    total_records += len(chunk)

    # Select a random sample from each chunk
    sample = chunk.sample(
        n=min(sample_size, len(chunk)),
        random_state=42 + chunk_number
    ).copy()

    # Convert Start_Time into datetime
    sample["Start_Time"] = pd.to_datetime(
        sample["Start_Time"],
        errors="coerce",
        utc=True
    )

    # Extract time-related features
    sample["Hour"] = sample["Start_Time"].dt.hour

    sample["Day_of_Week"] = (
        sample["Start_Time"].dt.day_of_week
    )

    sample["Month"] = sample["Start_Time"].dt.month

    # Create weekend feature
    sample["Is_Weekend"] = (
        sample["Day_of_Week"] >= 5
    ).astype(int)

    # Remove original timestamp column
    sample.drop(
        columns=["Start_Time"],
        inplace=True
    )

    # Remove records with missing target or invalid time
    sample.dropna(
        subset=["Severity", "Hour", "Day_of_Week", "Month"],
        inplace=True
    )

    # Save sample into CSV
    sample.to_csv(
        output_file,
        mode="a",
        header=(chunk_number == 0),
        index=False
    )

    sample_number += len(sample)

    print(
        f"Chunk {chunk_number + 1} processed | "
        f"Records scanned: {total_records:,} | "
        f"Samples saved: {sample_number:,}"
    )

# -----------------------------------------
# STEP 4: Load prepared dataset
# -----------------------------------------

data = pd.read_csv(output_file)

print("\nDATASET PREPARATION COMPLETED!")

print("\nDataset shape:", data.shape)

print("\nSeverity distribution:")
print(data["Severity"].value_counts().sort_index())

print("\nMissing values:")
print(data.isnull().sum())

print("\nFirst five records:")
print(data.head())

print("\nPrepared dataset saved at:")
print(output_file)
