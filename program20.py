# ==========================================
# PROGRAM 20: FULL DATASET HOTSPOT MAP
# ==========================================

import pandas as pd
import folium

from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = project_folder / "output" / "full_cluster_summary.csv"
output_path = project_folder / "output" / "full_accident_hotspot_map.html"

# Read cluster summary
df = pd.read_csv(input_path)

# Create base map
accident_map = folium.Map(
    location=[38, -98],
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

# Cluster marker colours
colors = {
    0: "red",
    1: "blue",
    2: "green"
}

# Add cluster markers
for _, row in df.iterrows():

    cluster_id = int(row["Cluster"])

    latitude = row["Latitude"]
    longitude = row["Longitude"]

    accidents = int(row["Total_Accidents"])

    # Scale circle radius according to accident count
    radius = max(8, (accidents / 100000) ** 0.5 * 5)

    popup_text = f"""
    <b>Cluster:</b> {cluster_id}<br>
    <b>Total Accidents:</b> {accidents:,}<br>
    <b>Percentage:</b> {row['Percentage']}%<br>
    <b>Average Severity:</b> {row['Average_Severity']}<br>
    <b>Latitude:</b> {latitude}<br>
    <b>Longitude:</b> {longitude}
    """

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=radius,
        color=colors[cluster_id],
        fill=True,
        fill_color=colors[cluster_id],
        fill_opacity=0.6,
        popup=folium.Popup(popup_text, max_width=300),
        tooltip=f"Cluster {cluster_id}"
    ).add_to(accident_map)

# Save interactive map
accident_map.save(output_path)

print("HOTSPOT MAP CREATED SUCCESSFULLY!")
print("Map saved at:", output_path)