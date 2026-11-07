
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
# STEP 2: Select road features
# -----------------------------------

road_columns = [
    "Traffic_Signal",
    "Crossing",
    "Junction",
    "Stop",
    "Amenity",
    "Railway",
    "Give_Way",
    "Bump"
]

use_columns = road_columns + ["Severity"]

# Store severity counts
severity_summary = pd.DataFrame(
    0,
    index=road_columns,
    columns=["Severity_1", "Severity_2", "Severity_3", "Severity_4"]
)

total_records = 0
chunk_number = 0

# -----------------------------------
# STEP 3: Read dataset in chunks
# -----------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=use_columns,
    chunksize=100000
):

    chunk_number += 1
    total_records += len(chunk)

    # Analyze each road feature separately
    for feature in road_columns:

        # Convert values to consistent Boolean format
        values = (
            chunk[feature]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        # Select records where feature is present
        present_records = chunk[values == "true"]

        # Count severity categories
        counts = present_records["Severity"].value_counts()

        for severity in [1, 2, 3, 4]:

            severity_summary.loc[
                feature, f"Severity_{severity}"
            ] += counts.get(severity, 0)

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {total_records:,}"
    )

# -----------------------------------
# STEP 4: Calculate total and percentages
# -----------------------------------

severity_summary["Total_Accidents"] = (
    severity_summary.sum(axis=1)
)

severity_summary["Severity_4_Percentage"] = (
    severity_summary["Severity_4"]
    / severity_summary["Total_Accidents"]
) * 100

# Sort by total accidents
severity_summary = severity_summary.sort_values(
    "Total_Accidents",
    ascending=False
)

# -----------------------------------
# STEP 5: Save results
# -----------------------------------

output_file = os.path.join(
    output_folder,
    "road_feature_severity_analysis.csv"
)

severity_summary.to_csv(output_file)

print("\nROAD FEATURE SEVERITY ANALYSIS COMPLETED!")

print("\nSeverity Summary:")
print(severity_summary.round(2))

# -----------------------------------
# STEP 6: Severity distribution graph
# -----------------------------------

severity_columns = [
    "Severity_1",
    "Severity_2",
    "Severity_3",
    "Severity_4"
]

severity_summary[severity_columns].plot(
    kind="bar",
    figsize=(13, 7)
)

plt.title("Accident Severity by Road Features")
plt.xlabel("Road Features")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.legend(title="Severity")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "road_feature_severity.png")
)

plt.show()

# -----------------------------------
# STEP 7: Severity 4 percentage graph
# -----------------------------------

plt.figure(figsize=(12, 6))

severity_summary["Severity_4_Percentage"].plot(
    kind="bar"
)

plt.title("Severity 4 Percentage by Road Features")
plt.xlabel("Road Features")
plt.ylabel("Severity 4 Percentage (%)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "road_feature_severity4.png")
)

plt.show()

print("\nAll files saved successfully!")
