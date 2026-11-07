
# ==========================================
# PROGRAM 22: FULL DATASET LOCALIZED CLUSTERING
# ==========================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = project_folder / "output" / "full_accident_locations.csv"

output_path = (
    project_folder / "output" /
    "localized_clustered_accidents.csv"
)

model_path = (
    project_folder / "output" /
    "localized_kmeans_model.pkl"
)

# Load trained K-Means model
model = joblib.load(model_path)

chunk_size = 100000
total_records = 0

# Initialize cluster counts
cluster_counts = {i: 0 for i in range(10)}

# Remove previous output
if output_path.exists():
    output_path.unlink()

print("LOCALIZED CLUSTER PREDICTION STARTED")

# Read full dataset in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(input_path, chunksize=chunk_size),
    start=1
):

    # Convert coordinates into radians
    lat = np.radians(chunk["Start_Lat"].to_numpy())
    lon = np.radians(chunk["Start_Lng"].to_numpy())

    # Convert into 3D geographical coordinates
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    X = np.column_stack((x, y, z))

    # Predict cluster labels
    chunk["Cluster"] = model.predict(X)

    # Count cluster records
    counts = chunk["Cluster"].value_counts()

    for cluster_id, count in counts.items():
        cluster_counts[int(cluster_id)] += int(count)

    # Save results chunk-wise
    chunk.to_csv(
        output_path,
        mode="a",
        index=False,
        header=(chunk_number == 1)
    )

    total_records += len(chunk)

    print(
        f"Chunk {chunk_number} completed | "
        f"Records: {total_records}"
    )

print("\nLOCALIZED CLUSTERING COMPLETED!")

print("Total records:", total_records)

print("\nCLUSTER DISTRIBUTION:")

for cluster_id, count in cluster_counts.items():

    percentage = count / total_records * 100

    print(
        f"Cluster {cluster_id}: "
        f"{count:,} accidents ({percentage:.2f}%)"
    )

print("\nOutput saved at:", output_path)
