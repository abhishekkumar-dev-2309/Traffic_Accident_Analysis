
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import os

# ---------------------------------------
# STEP 1: Dataset and output paths
# ---------------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

GRID_SIZE = 0.1

# ---------------------------------------
# STEP 2: Initialize variables
# ---------------------------------------

grid_summary = pd.DataFrame()

total_records = 0
chunk_number = 0

# ---------------------------------------
# STEP 3: Read dataset in chunks
# ---------------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=["Start_Lat", "Start_Lng", "Severity"],
    chunksize=100000
):

    chunk_number += 1
    total_records += len(chunk)

    # Remove records with missing coordinates
    chunk = chunk.dropna(
        subset=["Start_Lat", "Start_Lng", "Severity"]
    )

    # Assign geographic grid cells
    chunk["Latitude_Cell"] = np.floor(
        chunk["Start_Lat"] / GRID_SIZE
    )

    chunk["Longitude_Cell"] = np.floor(
        chunk["Start_Lng"] / GRID_SIZE
    )

    # Count severity categories in each grid cell
    current_grid = (
        chunk.groupby(
            ["Latitude_Cell", "Longitude_Cell", "Severity"]
        )
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all severity columns exist
    for severity in [1, 2, 3, 4]:
        if severity not in current_grid.columns:
            current_grid[severity] = 0

    current_grid = current_grid[[1, 2, 3, 4]]

    # Add current chunk results to previous results
    if grid_summary.empty:
        grid_summary = current_grid.copy()
    else:
        grid_summary = grid_summary.add(
            current_grid,
            fill_value=0
        )

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {total_records:,}"
    )

# ---------------------------------------
# STEP 4: Prepare final summary
# ---------------------------------------

grid_summary = grid_summary.reset_index()

grid_summary = grid_summary.rename(
    columns={
        1: "Severity_1",
        2: "Severity_2",
        3: "Severity_3",
        4: "Severity_4"
    }
)

# Calculate total accidents in each cell
severity_columns = [
    "Severity_1",
    "Severity_2",
    "Severity_3",
    "Severity_4"
]

grid_summary["Total_Accidents"] = (
    grid_summary[severity_columns].sum(axis=1)
)

# Calculate Severity 4 percentage
grid_summary["Severity_4_Percentage"] = (
    grid_summary["Severity_4"]
    / grid_summary["Total_Accidents"]
) * 100

# Calculate grid centre coordinates
grid_summary["Latitude"] = (
    grid_summary["Latitude_Cell"] + 0.5
) * GRID_SIZE

grid_summary["Longitude"] = (
    grid_summary["Longitude_Cell"] + 0.5
) * GRID_SIZE

# ---------------------------------------
# STEP 5: Save complete grid analysis
# ---------------------------------------

grid_summary = grid_summary.sort_values(
    "Total_Accidents",
    ascending=False
)

grid_summary.to_csv(
    os.path.join(
        output_folder,
        "grid_severity_analysis.csv"
    ),
    index=False
)

# ---------------------------------------
# STEP 6: Select reliable high-severity cells
# ---------------------------------------

# Minimum 1,000 accident records per cell
reliable_grid = grid_summary[
    grid_summary["Total_Accidents"] >= 1000
].copy()

# Sort by Severity 4 percentage
top_severity_grid = reliable_grid.sort_values(
    "Severity_4_Percentage",
    ascending=False
).head(20)

top_severity_grid.to_csv(
    os.path.join(
        output_folder,
        "top_20_severity_grid_cells.csv"
    ),
    index=False
)

print("\nGRID SEVERITY ANALYSIS COMPLETED!")

print("Total records processed:", f"{total_records:,}")

print("\nTop 20 Grid Cells by Severity 4 Percentage:")

print(
    top_severity_grid[
        [
            "Latitude",
            "Longitude",
            "Total_Accidents",
            "Severity_4",
            "Severity_4_Percentage"
        ]
    ].to_string(index=False)
)

# ---------------------------------------
# STEP 7: Plot Severity 4 percentage
# ---------------------------------------

plt.figure(figsize=(13, 7))

plt.bar(
    range(1, len(top_severity_grid) + 1),
    top_severity_grid["Severity_4_Percentage"]
)

plt.title("Top 20 Grid Cells by Severity 4 Percentage")
plt.xlabel("Grid Cell Rank")
plt.ylabel("Severity 4 Percentage (%)")
plt.xticks(range(1, len(top_severity_grid) + 1))

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "top_20_grid_severity4.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 8: Create interactive map
# ---------------------------------------

map_center = [
    top_severity_grid["Latitude"].mean(),
    top_severity_grid["Longitude"].mean()
]

severity_map = folium.Map(
    location=map_center,
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

for rank, (_, row) in enumerate(
    top_severity_grid.iterrows(),
    start=1
):

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=6 + (
            row["Severity_4_Percentage"] * 2
        ),
        color="red",
        fill=True,
        fill_opacity=0.6,
        popup=(
            f"Rank: {rank}<br>"
            f"Total Accidents: {int(row['Total_Accidents']):,}<br>"
            f"Severity 4: {int(row['Severity_4']):,}<br>"
            f"Severity 4 Percentage: "
            f"{row['Severity_4_Percentage']:.2f}%"
        )
    ).add_to(severity_map)

map_file = os.path.join(
    output_folder,
    "grid_severity_hotspot_map.html"
)

severity_map.save(map_file)

print("\nAll grid severity files saved successfully!")
print("Map saved at:", map_file)
