
import pandas as pd
import matplotlib.pyplot as plt
import folium
import os

# ---------------------------------------
# STEP 1: File paths
# ---------------------------------------

input_file = os.path.join(
    "output",
    "grid_severity_analysis.csv"
)

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------
# STEP 2: Load grid severity data
# ---------------------------------------

grid_data = pd.read_csv(input_file)

print("Total grid cells:", len(grid_data))

# ---------------------------------------
# STEP 3: Remove cells with insufficient data
# ---------------------------------------

# Minimum 1,000 accident records
grid_data = grid_data[
    grid_data["Total_Accidents"] >= 1000
].copy()

print("Grid cells after filtering:", len(grid_data))

# ---------------------------------------
# STEP 4: Calculate median thresholds
# ---------------------------------------

density_threshold = grid_data[
    "Total_Accidents"
].median()

severity_threshold = grid_data[
    "Severity_4_Percentage"
].median()

print("\nDensity Threshold:", density_threshold)
print("Severity Threshold:", severity_threshold)

# ---------------------------------------
# STEP 5: Classify grid cells
# ---------------------------------------

def classify_hotspot(row):

    high_density = (
        row["Total_Accidents"] >= density_threshold
    )

    high_severity = (
        row["Severity_4_Percentage"] >= severity_threshold
    )

    if high_density and high_severity:
        return "High Density + High Severity"

    elif high_density and not high_severity:
        return "High Density + Low Severity"

    elif not high_density and high_severity:
        return "Low Density + High Severity"

    else:
        return "Low Density + Low Severity"


grid_data["Hotspot_Category"] = grid_data.apply(
    classify_hotspot,
    axis=1
)

# ---------------------------------------
# STEP 6: Save classified data
# ---------------------------------------

output_file = os.path.join(
    output_folder,
    "combined_hotspot_classification.csv"
)

grid_data.to_csv(output_file, index=False)

print("\nHOTSPOT CLASSIFICATION COMPLETED!")

print("\nCategory Distribution:")

print(
    grid_data["Hotspot_Category"].value_counts()
)

# ---------------------------------------
# STEP 7: Category-wise summary
# ---------------------------------------

category_summary = (
    grid_data.groupby("Hotspot_Category")
    .agg(
        Total_Grid_Cells=("Total_Accidents", "count"),
        Total_Accidents=("Total_Accidents", "sum"),
        Average_Severity4_Percentage=(
            "Severity_4_Percentage", "mean"
        )
    )
    .reset_index()
)

category_summary.to_csv(
    os.path.join(
        output_folder,
        "hotspot_category_summary.csv"
    ),
    index=False
)

print("\nCategory Summary:")
print(category_summary.round(2))

# ---------------------------------------
# STEP 8: Category distribution graph
# ---------------------------------------

category_counts = grid_data[
    "Hotspot_Category"
].value_counts()

plt.figure(figsize=(12, 6))

category_counts.plot(
    kind="bar",
    color="steelblue"
)

plt.title("Combined Accident Hotspot Classification")
plt.xlabel("Hotspot Category")
plt.ylabel("Number of Grid Cells")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "combined_hotspot_classification.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 9: Create interactive map
# ---------------------------------------

map_center = [
    grid_data["Latitude"].mean(),
    grid_data["Longitude"].mean()
]

hotspot_map = folium.Map(
    location=map_center,
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

# Category colours
colors = {
    "High Density + High Severity": "red",
    "High Density + Low Severity": "orange",
    "Low Density + High Severity": "blue",
    "Low Density + Low Severity": "green"
}

# Add classified grid cells to map
for _, row in grid_data.iterrows():

    category = row["Hotspot_Category"]

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=4,
        color=colors[category],
        fill=True,
        fill_color=colors[category],
        fill_opacity=0.6,
        popup=(
            f"Category: {category}<br>"
            f"Accidents: {int(row['Total_Accidents']):,}<br>"
            f"Severity 4: "
            f"{row['Severity_4_Percentage']:.2f}%"
        )
    ).add_to(hotspot_map)

# Add map legend
legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
z-index: 9999;
background-color: white;
padding: 12px;
border: 2px solid grey;
border-radius: 5px;
">
<b>Hotspot Categories</b><br>
<span style="color:red;">●</span> High Density + High Severity<br>
<span style="color:orange;">●</span> High Density + Low Severity<br>
<span style="color:blue;">●</span> Low Density + High Severity<br>
<span style="color:green;">●</span> Low Density + Low Severity
</div>
"""

hotspot_map.get_root().html.add_child(
    folium.Element(legend_html)
)

map_file = os.path.join(
    output_folder,
    "combined_hotspot_map.html"
)

hotspot_map.save(map_file)

print("\nAll classification files saved successfully!")
print("Map saved at:", map_file)
