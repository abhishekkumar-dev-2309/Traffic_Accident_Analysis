
# ==========================================
# PROGRAM 15: FULL DATASET INSPECTION
# MEMORY-EFFICIENT CHUNK PROCESSING
# ==========================================

# Import Pandas for data processing
import pandas as pd

# Import Path for file management
from pathlib import Path

# 1. Find project directory
project_folder = Path(__file__).resolve().parent

# 2. Define full dataset path
dataset_path = (
    project_folder / "dataset" / "US_Accidents_March23.csv"
)

# 3. Check whether dataset exists
if not dataset_path.exists():
    raise FileNotFoundError("Dataset file not found!")

# 4. Display dataset file size
file_size_gb = dataset_path.stat().st_size / (1024 ** 3)

print("FULL DATASET INSPECTION")
print("-----------------------")
print("Dataset size:", round(file_size_gb, 2), "GB")

# 5. Select only required columns
required_columns = [
    'ID',
    'Severity',
    'Start_Time',
    'Start_Lat',
    'Start_Lng'
]

# 6. Initialize counters
total_records = 0
valid_coordinates = 0
missing_coordinates = 0
invalid_coordinates = 0

severity_counts = {}

# 7. Read CSV in batches of 100,000 rows
chunk_size = 100000

for chunk_number, chunk in enumerate(
    pd.read_csv(
        dataset_path,
        usecols=required_columns,
        chunksize=chunk_size
    ),
    start=1
):

    # Count total records
    total_records += len(chunk)

    # Convert coordinates into numeric values
    latitude = pd.to_numeric(
        chunk['Start_Lat'],
        errors='coerce'
    )

    longitude = pd.to_numeric(
        chunk['Start_Lng'],
        errors='coerce'
    )

    # Identify missing coordinates
    missing_mask = latitude.isna() | longitude.isna()

    missing_coordinates += missing_mask.sum()

    # Check valid geographical coordinate ranges
    valid_mask = (
        latitude.between(-90, 90) &
        longitude.between(-180, 180)
    )

    valid_coordinates += (
        (~missing_mask) & valid_mask
    ).sum()

    # Count invalid non-missing coordinates
    invalid_coordinates += (
        (~missing_mask) & (~valid_mask)
    ).sum()

    # Count accident severity levels
    counts = chunk['Severity'].value_counts()

    for severity, count in counts.items():
        severity_counts[severity] = (
            severity_counts.get(severity, 0) + count
        )

    # Display processing progress
    print(
        f"Processed chunk {chunk_number} | "
        f"Total records: {total_records:,}"
    )

# 8. Display final inspection report
print("\n========== FINAL DATASET REPORT ==========")

print("Total Records:", f"{total_records:,}")

print(
    "Valid Coordinates:",
    f"{valid_coordinates:,}"
)

print(
    "Missing Coordinates:",
    f"{missing_coordinates:,}"
)

print(
    "Invalid Coordinates:",
    f"{invalid_coordinates:,}"
)

print("\nSEVERITY DISTRIBUTION:")

for severity in sorted(severity_counts):
    print(
        f"Severity {severity}: "
        f"{severity_counts[severity]:,}"
    )

# 9. Save inspection report
output_folder = project_folder / "output"
output_folder.mkdir(exist_ok=True)

report = pd.DataFrame({
    'Metric': [
        'Total Records',
        'Valid Coordinates',
        'Missing Coordinates',
        'Invalid Coordinates'
    ],
    'Count': [
        total_records,
        valid_coordinates,
        missing_coordinates,
        invalid_coordinates
    ]
})

report_path = output_folder / "full_dataset_inspection.csv"

report.to_csv(report_path, index=False)

print("\nInspection completed successfully!")
print("Report saved at:", report_path)
