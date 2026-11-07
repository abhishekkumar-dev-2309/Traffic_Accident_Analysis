
import pandas as pd
from pathlib import Path

# -----------------------------------------
# STEP 1: Define project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "dataset" / "US_Accidents_March23.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "temporal_prediction_dataset.csv"

# -----------------------------------------
# STEP 2: Select required columns
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
# STEP 3: Read original dataset in chunks
# -----------------------------------------

chunksize = 100000
sample_size = 5000

sampled_chunks = []

print("Reading original accident dataset...")

chunk_number = 0

for chunk in pd.read_csv(
    DATASET_FILE,
    usecols=selected_columns,
    chunksize=chunksize,
    low_memory=False
):

    # Sample records from each chunk
    sample = chunk.sample(
        n=min(sample_size, len(chunk)),
        random_state=42 + chunk_number
    )

    # Convert Start_Time into datetime
    sample["Start_Time"] = pd.to_datetime(
        sample["Start_Time"],
        utc=True,
        errors="coerce"
    )

    # Remove invalid timestamps and missing target
    sample = sample.dropna(
        subset=["Start_Time", "Severity"]
    )

    # Extract temporal features
    sample["Year"] = sample["Start_Time"].dt.year

    sample["Hour"] = sample["Start_Time"].dt.hour

    sample["Day_of_Week"] = (
        sample["Start_Time"].dt.day_of_week
    )

    sample["Month"] = sample["Start_Time"].dt.month

    sample["Is_Weekend"] = (
        sample["Day_of_Week"] >= 5
    ).astype("int8")

    sampled_chunks.append(sample)

    chunk_number += 1

    print(
        f"Processed chunk: {chunk_number}, "
        f"Sampled records: {len(sample)}"
    )

# -----------------------------------------
# STEP 4: Combine all sampled records
# -----------------------------------------

print("\nCombining sampled records...")

final_data = pd.concat(
    sampled_chunks,
    ignore_index=True
)

# Sort records chronologically
final_data = final_data.sort_values(
    by="Start_Time"
).reset_index(drop=True)

# -----------------------------------------
# STEP 5: Check temporal information
# -----------------------------------------

print("\n========== TEMPORAL DATASET SUMMARY ==========")

print("Total records:", len(final_data))

print(
    "Earliest accident date:",
    final_data["Start_Time"].min()
)

print(
    "Latest accident date:",
    final_data["Start_Time"].max()
)

print("\nYear-wise record distribution:")

print(final_data["Year"].value_counts().sort_index())

# -----------------------------------------
# STEP 6: Save prepared dataset
# -----------------------------------------

final_data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nTemporal dataset saved successfully!")

print("Output file:", OUTPUT_FILE)

print("\nFinal dataset shape:", final_data.shape)
