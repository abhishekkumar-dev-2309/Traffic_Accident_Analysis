
# ==============================================
# PROGRAM 23: LOCALIZED CLUSTER ANALYSIS
# ==============================================

import pandas as pd
import numpy as np

from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = (
    project_folder / "output" /
    "localized_clustered_accidents.csv"
)

output_path = (
    project_folder / "output" /
    "localized_cluster_summary.csv"
)

chunk_size = 100000
K = 10

# Initialize statistics for 10 clusters
stats = {}

for cluster_id in range(K):
    stats[cluster_id] = {
        "count": 0,
        "severity": {1: 0, 2: 0, 3: 0, 4: 0},
        "x_sum": 0.0,
        "y_sum": 0.0,
        "z_sum": 0.0
    }

print("LOCALIZED CLUSTER ANALYSIS STARTED")

# Read data chunk-wise
for chunk_number, chunk in enumerate(
    pd.read_csv(input_path, chunksize=chunk_size),
    start=1
):

    # Convert latitude and longitude into radians
    lat = np.radians(chunk["Start_Lat"].to_numpy())
    lon = np.radians(chunk["Start_Lng"].to_numpy())

    # Convert geographical coordinates into 3D
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    chunk["X"] = x
    chunk["Y"] = y
    chunk["Z"] = z

    # Calculate statistics for each cluster
    for cluster_id in range(K):

        data = chunk[chunk["Cluster"] == cluster_id]

        s = stats[cluster_id]

        s["count"] += len(data)

        # Accumulate severity counts
        for severity in range(1, 5):
            s["severity"][severity] += int(
                (data["Severity"] == severity).sum()
            )

        # Accumulate geographical coordinates
        s["x_sum"] += data["X"].sum()
        s["y_sum"] += data["Y"].sum()
        s["z_sum"] += data["Z"].sum()

    print(f"Chunk {chunk_number} analyzed")

# Calculate total records
total_records = sum(
    s["count"] for s in stats.values()
)

# Generate summary
summary = []

for cluster_id, s in stats.items():

    # Average 3D geographical coordinates
    x = s["x_sum"] / s["count"]
    y = s["y_sum"] / s["count"]
    z = s["z_sum"] / s["count"]

    # Normalize the vector
    magnitude = np.sqrt(x*x + y*y + z*z)

    x /= magnitude
    y /= magnitude
    z /= magnitude

    # Convert back to latitude and longitude
    latitude = np.degrees(np.arcsin(z))
    longitude = np.degrees(np.arctan2(y, x))

    # Calculate average numerical severity
    severity_total = sum(
        level * count
        for level, count in s["severity"].items()
    )

    average_severity = severity_total / s["count"]

    # Severity 4 percentage
    severity4_percentage = (
        s["severity"][4] / s["count"]
    ) * 100

    summary.append({
        "Cluster": cluster_id,
        "Total_Accidents": s["count"],
        "Percentage": round(
            s["count"] / total_records * 100, 2
        ),
        "Latitude": round(latitude, 6),
        "Longitude": round(longitude, 6),
        "Average_Severity": round(average_severity, 3),
        "Severity_1": s["severity"][1],
        "Severity_2": s["severity"][2],
        "Severity_3": s["severity"][3],
        "Severity_4": s["severity"][4],
        "Severity_4_Percentage": round(
            severity4_percentage, 2
        )
    })

# Save final summary
summary_df = pd.DataFrame(summary)

summary_df.to_csv(output_path, index=False)

print("\nFINAL LOCALIZED CLUSTER SUMMARY")
print(summary_df.to_string(index=False))

print("\nTotal records analyzed:", total_records)
print("Summary saved at:", output_path)
