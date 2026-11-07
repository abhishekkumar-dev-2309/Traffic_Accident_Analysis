
import pandas as pd
import matplotlib.pyplot as plt
import os

# -------------------------------
# STEP 1: File paths
# -------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# STEP 2: Initialize variables
# -------------------------------

weather_summary = pd.DataFrame()

total_records = 0
chunk_number = 0

# -------------------------------
# STEP 3: Read dataset in chunks
# -------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=["Weather_Condition", "Severity"],
    chunksize=100000
):

    chunk_number += 1
    total_records += len(chunk)

    # Remove extra spaces from weather names
    chunk["Weather_Condition"] = (
        chunk["Weather_Condition"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    # Count accidents by weather and severity
    current_summary = pd.crosstab(
        chunk["Weather_Condition"],
        chunk["Severity"]
    )

    # Add current chunk results to previous results
    weather_summary = weather_summary.add(
        current_summary,
        fill_value=0
    )

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {total_records:,}"
    )

# -------------------------------
# STEP 4: Calculate statistics
# -------------------------------

# Ensure all severity columns exist
for severity in [1, 2, 3, 4]:
    if severity not in weather_summary.columns:
        weather_summary[severity] = 0

weather_summary = weather_summary[[1, 2, 3, 4]]

weather_summary = weather_summary.rename(
    columns={
        1: "Severity_1",
        2: "Severity_2",
        3: "Severity_3",
        4: "Severity_4"
    }
)

# Total accidents for each weather condition
weather_summary["Total_Accidents"] = (
    weather_summary.sum(axis=1)
)

# Percentage of Severity 4 accidents
weather_summary["Severity_4_Percentage"] = (
    weather_summary["Severity_4"]
    / weather_summary["Total_Accidents"]
) * 100

# Sort by total accidents
weather_summary = weather_summary.sort_values(
    "Total_Accidents",
    ascending=False
)

# -------------------------------
# STEP 5: Save complete analysis
# -------------------------------

output_file = os.path.join(
    output_folder,
    "weather_condition_analysis.csv"
)

weather_summary.to_csv(output_file)

print("\nWEATHER ANALYSIS COMPLETED!")
print("Total records processed:", f"{total_records:,}")

print("\nTop 10 Weather Conditions:")
print(weather_summary.head(10))

# -------------------------------
# STEP 6: Top 10 accident conditions
# -------------------------------

top_weather = weather_summary.head(10)

plt.figure(figsize=(12, 6))

top_weather["Total_Accidents"].plot(
    kind="bar",
    color="skyblue"
)

plt.title("Top 10 Weather Conditions by Accident Count")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "top_10_weather_accidents.png")
)

plt.show()

# -------------------------------
# STEP 7: Severity 4 percentage
# -------------------------------

# Ignore conditions with fewer than 1,000 records
reliable_weather = weather_summary[
    weather_summary["Total_Accidents"] >= 1000
]

top_severity_weather = reliable_weather.sort_values(
    "Severity_4_Percentage",
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

top_severity_weather["Severity_4_Percentage"].plot(
    kind="bar",
    color="orange"
)

plt.title("Top 10 Weather Conditions by Severity 4 Percentage")
plt.xlabel("Weather Condition")
plt.ylabel("Severity 4 Percentage (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "top_10_weather_severity4.png")
)

plt.show()

print("\nAll weather analysis files saved successfully!")
