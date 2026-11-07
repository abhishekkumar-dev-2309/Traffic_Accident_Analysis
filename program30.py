
import pandas as pd
import folium
import numpy as np

# -----------------------------------
# STEP 1: LOAD CLUSTER SUMMARY
# -----------------------------------

file_path = "output/k10_k15_comparison.csv"

data = pd.read_csv(file_path)

# Select only K=15 clusters
data = data[data["K"] == 15].copy()

print("Total K=15 clusters:", len(data))

# -----------------------------------
# STEP 2: CREATE BASE MAP
# -----------------------------------

# Approximate centre of USA
m = folium.Map(
    location=[39.5, -98.35],
    zoom_start=4,
    tiles="Esri.WorldStreetMap"
)

# -----------------------------------
# STEP 3: ADD CLUSTER MARKERS
# -----------------------------------

for _, row in data.iterrows():

    latitude = row["Latitude"]
    longitude = row["Longitude"]

    accidents = row["Total_Accidents"]
    severity = row["Average_Severity"]
    severity4 = row["Severity_4_Percentage"]

    # Select marker colour
    if severity4 >= 5:
        color = "red"

    elif severity4 >= 3:
        color = "orange"

    else:
        color = "green"

    # Circle size based on accident count
    radius = max(
        5,
        min(20, accidents / 60000)
    )

    # Popup information
    popup_text = f"""
    <b>Cluster:</b> {int(row['Cluster'])}<br>
    <b>Total Accidents:</b> {accidents:,}<br>
    <b>Average Severity:</b> {severity:.3f}<br>
    <b>Severity 4:</b> {severity4:.2f}%<br>
    <b>Latitude:</b> {latitude:.4f}<br>
    <b>Longitude:</b> {longitude:.4f}
    """

    # Add circle marker
    folium.CircleMarker(
        location=[latitude, longitude],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        popup=folium.Popup(
            popup_text,
            max_width=300
        ),
        tooltip=f"Cluster {int(row['Cluster'])}"
    ).add_to(m)

# -----------------------------------
# STEP 4: ADD MAP LEGEND
# -----------------------------------

legend_html = """
<div style="
position: fixed;
bottom: 30px;
left: 30px;
width: 210px;
background-color: white;
border: 2px solid grey;
z-index: 9999;
padding: 10px;
font-size: 14px;
">
<b>Severity 4 Percentage</b><br>
<span style="color:green;">●</span> Below 3%<br>
<span style="color:orange;">●</span> 3% to below 5%<br>
<span style="color:red;">●</span> 5% and above
</div>
"""

m.get_root().html.add_child(
    folium.Element(legend_html)
)

# -----------------------------------
# STEP 5: SAVE INTERACTIVE MAP
# -----------------------------------

output_path = "output/k15_accident_hotspot_map.html"

m.save(output_path)

print("\nK=15 HOTSPOT MAP CREATED!")
print("Total clusters:", len(data))
print("Map saved at:", output_path)
