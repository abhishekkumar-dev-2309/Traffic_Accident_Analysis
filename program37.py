
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

# Grid size in degrees
GRID_SIZE = 0.1

# ---------------------------------------
# STEP 2: Initialize variables
# ---------------------------------------

grid_summary = pd.DataFrame(
    columns=["Latitude_Cell", "Longitude_Cell", "Total_Accidents"]
)

total_records = 0
chunk_number = 0

# ---------------------------------------
# STEP 3: Read dataset in chunks
# ---------------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=["Start_Lat", "Start_Lng"],
    chunksize=100000
):

    chunk_number += 1
    total_records += len(chunk)

    # Remove records with missing coordinates
    chunk = chunk.dropna(
        subset=["Start_Lat", "Start_Lng"]
    )

    # Assign each accident to a grid cell
    chunk["Latitude_Cell"] = np.floor(
        chunk["Start_Lat"] / GRID_SIZE
    )

    chunk["Longitude_Cell"] = np.floor(
        chunk["Start_Lng"] / GRID_SIZE
    )

    # Count accidents in each grid cell
    current_grid = (
        chunk.groupby(
            ["Latitude_Cell", "Longitude_Cell"]
        )
        .size()
        .reset_index(name="Total_Accidents")
    )

    # Combine current chunk with previous counts
    grid_summary = pd.concat(
        [grid_summary, current_grid],
        ignore_index=True
    )

    # Recalculate combined grid counts
    grid_summary = (
        grid_summary.groupby(
            ["Latitude_Cell", "Longitude_Cell"],
            as_index=False
        )["Total_Accidents"]
        .sum()
    )

    print(
        f"Processed chunk {chunk_number} | "
        f"Records: {total_records:,}"
    )

# ---------------------------------------
# STEP 4: Calculate grid centre coordinates
# ---------------------------------------

grid_summary["Latitude"] = (
    grid_summary["Latitude_Cell"] + 0.5
) * GRID_SIZE

grid_summary["Longitude"] = (
    grid_summary["Longitude_Cell"] + 0.5
) * GRID_SIZE

# Sort grid cells by accident count
grid_summary = grid_summary.sort_values(
    "Total_Accidents",
    ascending=False
)

# ---------------------------------------
# STEP 5: Save complete grid analysis
# ---------------------------------------

grid_file = os.path.join(
    output_folder,
    "accident_grid_density.csv"
)

grid_summary.to_csv(grid_file, index=False)

# Select top 20 densest grid cells
top_20 = grid_summary.head(20).copy()

top_file = os.path.join(
    output_folder,
    "top_20_accident_density_locations.csv"
)

top_20.to_csv(top_file, index=False)

print("\nGRID DENSITY ANALYSIS COMPLETED!")

print("Total records processed:", f"{total_records:,}")

print("\nTop 20 Accident Density Grid Cells:")
print(
    top_20[
        ["Latitude", "Longitude", "Total_Accidents"]
    ].to_string(index=False)
)

# ---------------------------------------
# STEP 6: Visualize top 20 grid cells
# ---------------------------------------

plt.figure(figsize=(12, 7))

plt.bar(
    range(1, len(top_20) + 1),
    top_20["Total_Accidents"]
)

plt.title("Top 20 Geographic Grid Cells by Accident Count")
plt.xlabel("Grid Cell Rank")
plt.ylabel("Number of Accident Records")
plt.xticks(range(1, len(top_20) + 1))

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "top_20_accident_density.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 7: Create interactive density map
# ---------------------------------------

map_center = [
    top_20["Latitude"].mean(),
    top_20["Longitude"].mean()
]

accident_map = folium.Map(
    location=map_center,
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

for rank, (_, row) in enumerate(
    top_20.iterrows(), start=1
):

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=6 + (
            row["Total_Accidents"]
            / top_20["Total_Accidents"].max()
        ) * 12,
        color="red",
        fill=True,
        fill_opacity=0.6,
        popup=(
            f"Rank: {rank}<br>"
            f"Accidents: {int(row['Total_Accidents']):,}"
        )
    ).add_to(accident_map)

map_file = os.path.join(
    output_folder,
    "accident_grid_density_map.html"
)

accident_map.save(map_file)

print("\nAll grid density files saved successfully!")
print("Map saved at:", map_file)
