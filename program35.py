
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------
# STEP 1: Dataset path
# -----------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# -----------------------------------
# STEP 2: Road-related columns
# -----------------------------------

road_columns = [
    "Amenity",
    "Bump",
    "Crossing",
    "Give_Way",
    "Junction",
    "Railway",
    "Stop",
    "Traffic_Signal"
]

# Store counts for True and False
road_summary = pd.DataFrame(
    0,
    index=road_columns,
    columns=["True_Count", "False_Count", "Unknown_Count"]
)

total_records = 0
chunk_number = 0

# -----------------------------------
# STEP 3: Read dataset in chunks
# -----------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=road_columns,
    chunksize=100000
):

    chunk_number += 1
    total_records += len(chunk)

    for column in road_columns:

        # Convert values into consistent Boolean format
        values = (
            chunk[column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        road_summary.loc[column, "True_Count"] += (
            (values == "true").sum()
        )

        road_summary.loc[column, "False_Count"] += (
            (values == "false").sum()
        )

        road_summary.loc[column, "Unknown_Count"] += (
            values.isna().sum()
        )

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {total_records:,}"
    )

# -----------------------------------
# STEP 4: Calculate percentages
# -----------------------------------

road_summary["Present_Percentage"] = (
    road_summary["True_Count"] / total_records
) * 100

road_summary["Absent_Percentage"] = (
    road_summary["False_Count"] / total_records
) * 100

road_summary["Unknown_Percentage"] = (
    road_summary["Unknown_Count"] / total_records
) * 100

# Sort by True count
road_summary = road_summary.sort_values(
    "True_Count",
    ascending=False
)

# -----------------------------------
# STEP 5: Save analysis
# -----------------------------------

output_file = os.path.join(
    output_folder,
    "road_feature_analysis.csv"
)

road_summary.to_csv(output_file)

print("\nROAD FEATURE ANALYSIS COMPLETED!")

print("Total records:", f"{total_records:,}")

print("\nRoad Feature Summary:")
print(road_summary.round(2))

# -----------------------------------
# STEP 6: Plot True counts
# -----------------------------------

plt.figure(figsize=(12, 6))

road_summary["True_Count"].plot(
    kind="bar",
    color="steelblue"
)

plt.title("Accidents Recorded with Road Features")
plt.xlabel("Road Features")
plt.ylabel("Number of Accident Records")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "road_feature_counts.png")
)

plt.show()

# -----------------------------------
# STEP 7: Plot presence percentages
# -----------------------------------

plt.figure(figsize=(12, 6))

road_summary["Present_Percentage"].plot(
    kind="bar",
    color="orange"
)

plt.title("Percentage of Records with Road Features Present")
plt.xlabel("Road Features")
plt.ylabel("Presence Percentage (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "road_feature_percentages.png")
)

plt.show()

print("\nAll road analysis files saved successfully!")
