# ==============================================
# PROGRAM 19: FULL DATASET CLUSTER ANALYSIS
# ==============================================

import pandas as pd
import numpy as np

from pathlib import Path

# Project folder
project_folder = Path(__file__).resolve().parent

input_path = project_folder / "output" / "full_clustered_accidents.csv"
output_path = project_folder / "output" / "full_cluster_summary.csv"

chunk_size = 100000

# Initialize cluster statistics
cluster_stats = {}

for cluster_id in range(3):

    cluster_stats[cluster_id] = {
        "count": 0,
        "severity": {1: 0, 2: 0, 3: 0, 4: 0},
        "x_sum": 0.0,
        "y_sum": 0.0,
        "z_sum": 0.0
    }

print("FULL DATASET CLUSTER ANALYSIS STARTED")

# Read data in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(input_path, chunksize=chunk_size),
    start=1
):

    # Convert latitude and longitude into radians
    lat = np.radians(chunk["Start_Lat"].to_numpy())
    lon = np.radians(chunk["Start_Lng"].to_numpy())

    # Convert coordinates into 3D
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    chunk["X"] = x
    chunk["Y"] = y
    chunk["Z"] = z

    # Analyze each cluster
    for cluster_id in range(3):

        data = chunk[chunk["Cluster"] == cluster_id]

        stats = cluster_stats[cluster_id]

        stats["count"] += len(data)

        stats["severity"].update({
            severity: stats["severity"][severity]
            + int((data["Severity"] == severity).sum())
            for severity in range(1, 5)
        })

        stats["x_sum"] += data["X"].sum()
        stats["y_sum"] += data["Y"].sum()
        stats["z_sum"] += data["Z"].sum()

    print(f"Chunk {chunk_number} analyzed")

# Create final summary
summary = []

for cluster_id, stats in cluster_stats.items():

    # Average 3D coordinates
    x = stats["x_sum"] / stats["count"]
    y = stats["y_sum"] / stats["count"]
    z = stats["z_sum"] / stats["count"]

    # Normalize average vector
    magnitude = np.sqrt(x*x + y*y + z*z)

    x = x / magnitude
    y = y / magnitude
    z = z / magnitude

    # Convert 3D coordinates into latitude-longitude
    latitude = np.degrees(np.arcsin(z))
    longitude = np.degrees(np.arctan2(y, x))

    # Average numerical severity
    severity_sum = sum(
        severity * count
        for severity, count in stats["severity"].items()
    )

    average_severity = severity_sum / stats["count"]

    summary.append({
        "Cluster": cluster_id,
        "Total_Accidents": stats["count"],
        "Percentage": round(
            stats["count"] / sum(
                s["count"] for s in cluster_stats.values()
            ) * 100, 2
        ),
        "Latitude": round(latitude, 6),
        "Longitude": round(longitude, 6),
        "Average_Severity": round(average_severity, 3),
        "Severity_1": stats["severity"][1],
        "Severity_2": stats["severity"][2],
        "Severity_3": stats["severity"][3],
        "Severity_4": stats["severity"][4]
    })

# Save summary
summary_df = pd.DataFrame(summary)

summary_df.to_csv(output_path, index=False)

print("\nFINAL CLUSTER SUMMARY")
print(summary_df.to_string(index=False))

print("\nSummary saved at:", output_path)