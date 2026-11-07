
# ==========================================
# PROGRAM 16: FULL DATASET PREPARATION
# CHUNK-BASED DATA EXTRACTION
# ==========================================

import pandas as pd
from pathlib import Path

# Find project directory
project_folder = Path(__file__).resolve().parent

# Original full dataset
dataset_path = (
    project_folder / "dataset" / "US_Accidents_March23.csv"
)

# Output directory
output_folder = project_folder / "output"
output_folder.mkdir(exist_ok=True)

# Required columns only
required_columns = [
    'ID',
    'Severity',
    'Start_Time',
    'Start_Lat',
    'Start_Lng'
]

# Output file path
output_path = output_folder / "full_accident_locations.csv"

# Remove previous output if it exists
if output_path.exists():
    output_path.unlink()

# Initialize counters
total_saved = 0
chunk_size = 100000

print("FULL DATASET PREPARATION STARTED")

# Read dataset in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(
        dataset_path,
        usecols=required_columns,
        chunksize=chunk_size
    ),
    start=1
):

    # Convert coordinates into numeric values
    chunk['Start_Lat'] = pd.to_numeric(
        chunk['Start_Lat'],
        errors='coerce'
    )

    chunk['Start_Lng'] = pd.to_numeric(
        chunk['Start_Lng'],
        errors='coerce'
    )

    # Keep records with valid coordinates
    valid_mask = (
        chunk['Start_Lat'].between(-90, 90) &
        chunk['Start_Lng'].between(-180, 180)
    )

    clean_chunk = chunk.loc[valid_mask].copy()

    # Save current batch
    clean_chunk.to_csv(
        output_path,
        mode='a',
        header=(chunk_number == 1),
        index=False
    )

    # Update saved record count
    total_saved += len(clean_chunk)

    print(
        f"Chunk {chunk_number} processed | "
        f"Records saved: {total_saved:,}"
    )

# Final report
print("\n========== PREPARATION REPORT ==========")
print("Total valid records saved:", f"{total_saved:,}")
print("Output file:", output_path)

print("\nFull dataset preparation completed!")

