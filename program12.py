
# ==========================================
# PROGRAM 12: INTERACTIVE ACCIDENT HOTSPOT MAP
# ==========================================

# Import required libraries
import pandas as pd
import folium

from pathlib import Path

# Find project directory
project_folder = Path(__file__).resolve().parent

# Load clustered accident dataset
dataset_path = project_folder / "output" / "clustered_accidents.csv"
df = pd.read_csv(dataset_path)

# Define map center
center_lat = df['Start_Lat'].mean()
center_lon = df['Start_Lng'].mean()

# Create interactive map using CartoDB tiles

# Create interactive map using Esri basemap
accident_map = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles="Esri.WorldStreetMap"
)
# Define colours for clusters
cluster_colors = {
    0: 'purple',
    1: 'orange',
    2: 'green'
}

# Add accident locations to map
for _, row in df.iterrows():

    # Get cluster number
    cluster = int(row['Cluster'])

    # Get corresponding colour
    color = cluster_colors[cluster]

    # Add accident location marker
    folium.CircleMarker(
        location=[
            row['Start_Lat'],
            row['Start_Lng']
        ],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=f"Cluster: {cluster}"
    ).add_to(accident_map)

# Add cluster centroids
centroids = [
    (37.988167, -121.845748),
    (40.270874, -82.749951),
    (39.702301, -84.173790)
]

for i, (lat, lon) in enumerate(centroids):

    folium.Marker(
        location=[lat, lon],
        popup=f"Cluster {i} Centroid",
        tooltip=f"Centroid {i}",
        icon=folium.Icon(
            color='red',
            icon='star'
        )
    ).add_to(accident_map)

# Save interactive map
output_path = project_folder / "output" / "accident_hotspot_map.html"

accident_map.save(str(output_path))

print("\nInteractive hotspot map created successfully!")
print("Map saved at:", output_path)
