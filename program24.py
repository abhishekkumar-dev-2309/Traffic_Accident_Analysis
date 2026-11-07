
# ==========================================
# PROGRAM 24: LOCALIZED ACCIDENT HOTSPOT MAP
# ==========================================

import pandas as pd
import folium

from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = (
    project_folder / "output" /
    "localized_cluster_summary.csv"
)

output_path = (
    project_folder / "output" /
    "localized_accident_hotspot_map.html"
)

# Load cluster summary
df = pd.read_csv(input_path)

# Create USA base map
accident_map = folium.Map(
    location=[39, -98],
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

# Define marker colours based on Severity 4 percentage
def get_color(value):

    if value >= 5:
        return "red"

    elif value >= 3:
        return "orange"

    else:
        return "green"


# Add each cluster to the map
for _, row in df.iterrows():

    cluster_id = int(row["Cluster"])

    latitude = row["Latitude"]
    longitude = row["Longitude"]

    accidents = int(row["Total_Accidents"])

    severity4 = row["Severity_4_Percentage"]

    # Circle size based on accident count
    radius = max(
        7,
        (accidents / 100000) ** 0.5 * 5
    )

    # Select marker colour
    color = get_color(severity4)

    # Popup information
    popup_text = f"""
    <b>Cluster:</b> {cluster_id}<br>
    <b>Total Accidents:</b> {accidents:,}<br>
    <b>Percentage:</b> {row['Percentage']}%<br>
    <b>Average Severity:</b> {row['Average_Severity']}<br>
    <b>Severity 4 Count:</b> {row['Severity_4']:,}<br>
    <b>Severity 4 Percentage:</b> {severity4}%<br>
    <b>Latitude:</b> {latitude}<br>
    <b>Longitude:</b> {longitude}
    """

    # Create circle marker
    folium.CircleMarker(
        location=[latitude, longitude],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.65,
        weight=2,
        popup=folium.Popup(
            popup_text,
            max_width=350
        ),
        tooltip=f"Cluster {cluster_id}"
    ).add_to(accident_map)


# Add map legend
legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 220px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
padding: 10px;
font-size: 14px;
">
<b>Severity 4 Percentage</b><br>
<span style="color:green;">●</span> Below 3%<br>
<span style="color:orange;">●</span> 3% to below 5%<br>
<span style="color:red;">●</span> 5% and above<br>
<small>Circle size = Accident count</small>
</div>
"""

accident_map.get_root().html.add_child(
    folium.Element(legend_html)
)

# Save interactive map
accident_map.save(output_path)

print("LOCALIZED HOTSPOT MAP CREATED!")
print("Total clusters:", len(df))
print("Map saved at:", output_path)
