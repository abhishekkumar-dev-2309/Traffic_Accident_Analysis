
import pandas as pd
import matplotlib.pyplot as plt
import folium
import os

# ---------------------------------------
# STEP 1: File paths
# ---------------------------------------

input_file = os.path.join(
    "output",
    "combined_hotspot_classification.csv"
)

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------
# STEP 2: Load classified grid data
# ---------------------------------------

grid_data = pd.read_csv(input_file)

print("Total grid cells loaded:", len(grid_data))

# ---------------------------------------
# STEP 3: Calculate percentile rankings
# ---------------------------------------

# Rank accident density from 0 to 100
grid_data["Density_Score"] = (
    grid_data["Total_Accidents"].rank(pct=True) * 100
)

# Rank Severity 4 percentage from 0 to 100
grid_data["Severity_Score"] = (
    grid_data["Severity_4_Percentage"].rank(pct=True) * 100
)

# ---------------------------------------
# STEP 4: Calculate combined priority score
# ---------------------------------------

grid_data["Priority_Score"] = (
    0.5 * grid_data["Density_Score"]
    + 0.5 * grid_data["Severity_Score"]
)

# ---------------------------------------
# STEP 5: Assign priority levels
# ---------------------------------------

def assign_priority(score):

    if score >= 80:
        return "Very High"

    elif score >= 60:
        return "High"

    elif score >= 40:
        return "Medium"

    else:
        return "Low"


grid_data["Priority_Level"] = (
    grid_data["Priority_Score"].apply(assign_priority)
)

# ---------------------------------------
# STEP 6: Sort by priority score
# ---------------------------------------

grid_data = grid_data.sort_values(
    "Priority_Score",
    ascending=False
)

# Select top 20 priority locations
top_20 = grid_data.head(20).copy()

# ---------------------------------------
# STEP 7: Save complete results
# ---------------------------------------

grid_data.to_csv(
    os.path.join(
        output_folder,
        "final_priority_analysis.csv"
    ),
    index=False
)

top_20.to_csv(
    os.path.join(
        output_folder,
        "top_20_priority_hotspots.csv"
    ),
    index=False
)

print("\nFINAL PRIORITY ANALYSIS COMPLETED!")

print("\nTop 20 Priority Hotspots:")

print(
    top_20[
        [
            "Latitude",
            "Longitude",
            "Total_Accidents",
            "Severity_4_Percentage",
            "Density_Score",
            "Severity_Score",
            "Priority_Score",
            "Priority_Level"
        ]
    ].round(2).to_string(index=False)
)

# ---------------------------------------
# STEP 8: Priority level distribution
# ---------------------------------------

priority_counts = (
    grid_data["Priority_Level"].value_counts()
    .reindex(
        ["Very High", "High", "Medium", "Low"],
        fill_value=0
    )
)

print("\nPriority Level Distribution:")
print(priority_counts)

plt.figure(figsize=(10, 6))

priority_counts.plot(
    kind="bar",
    color="steelblue"
)

plt.title("Accident Hotspot Priority Distribution")
plt.xlabel("Priority Level")
plt.ylabel("Number of Grid Cells")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "priority_level_distribution.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 9: Top 20 priority score graph
# ---------------------------------------

plt.figure(figsize=(12, 7))

plt.bar(
    range(1, len(top_20) + 1),
    top_20["Priority_Score"],
    color="orange"
)

plt.title("Top 20 Accident Hotspots by Priority Score")
plt.xlabel("Hotspot Rank")
plt.ylabel("Priority Score (0-100)")
plt.xticks(range(1, len(top_20) + 1))
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "top_20_priority_scores.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 10: Create interactive priority map
# ---------------------------------------

map_center = [
    top_20["Latitude"].mean(),
    top_20["Longitude"].mean()
]

priority_map = folium.Map(
    location=map_center,
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

priority_colors = {
    "Very High": "red",
    "High": "orange",
    "Medium": "blue",
    "Low": "green"
}

for rank, (_, row) in enumerate(
    top_20.iterrows(),
    start=1
):

    color = priority_colors[row["Priority_Level"]]

    folium.CircleMarker(
        location=[
            row["Latitude"],
            row["Longitude"]
        ],
        radius=6 + (
            row["Priority_Score"] / 10
        ),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=(
            f"Rank: {rank}<br>"
            f"Priority: {row['Priority_Level']}<br>"
            f"Priority Score: {row['Priority_Score']:.2f}<br>"
            f"Total Accidents: {int(row['Total_Accidents']):,}<br>"
            f"Severity 4: {row['Severity_4_Percentage']:.2f}%"
        )
    ).add_to(priority_map)

map_file = os.path.join(
    output_folder,
    "final_priority_hotspot_map.html"
)

priority_map.save(map_file)

print("\nAll priority analysis files saved successfully!")
print("Map saved at:", map_file)
