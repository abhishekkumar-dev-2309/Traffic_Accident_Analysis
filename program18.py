# ==========================================
# PROGRAM 18: FULL DATASET CLUSTER PREDICTION
# ==========================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = project_folder / "output" / "full_accident_locations.csv"
output_path = project_folder / "output" / "full_clustered_accidents.csv"
model_path = project_folder / "output" / "full_kmeans_model.pkl"

# Load trained model
model = joblib.load(model_path)

chunk_size = 100000
total_records = 0

# Cluster count initialize
cluster_counts = {0: 0, 1: 0, 2: 0}

# Remove previous output if it exists
if output_path.exists():
    output_path.unlink()

print("FULL DATASET CLUSTER PREDICTION STARTED")

# Read dataset in chunks
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

    X = np.column_stack((x, y, z))

    # Predict cluster
    chunk["Cluster"] = model.predict(X)

    # Count records in each cluster
    counts = chunk["Cluster"].value_counts()

    for cluster_id, count in counts.items():
        cluster_counts[int(cluster_id)] += int(count)

    # Save chunk
    chunk.to_csv(
        output_path,
        mode="a",
        index=False,
        header=(chunk_number == 1)
    )

    total_records += len(chunk)

    print(f"Chunk {chunk_number} completed | Records: {total_records}")

print("\nCLUSTER PREDICTION COMPLETED!")

print("Total records:", total_records)

print("\nCLUSTER DISTRIBUTION:")

for cluster_id, count in cluster_counts.items():
    percentage = (count / total_records) * 100

    print(
        f"Cluster {cluster_id}: "
        f"{count} accidents ({percentage:.2f}%)"
    )

print("\nOutput saved at:", output_path)