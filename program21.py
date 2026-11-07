
# ==========================================
# PROGRAM 21: LOCALIZED K-MEANS (K = 10)
# ==========================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = project_folder / "output" / "full_accident_locations.csv"
output_folder = project_folder / "output"
model_path = output_folder / "localized_kmeans_model.pkl"

# Model configuration
K = 10
chunk_size = 100000
sample_fraction = 0.02

# Store random samples
sample_list = []

print("RANDOM SAMPLING STARTED")

# Read full dataset in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_path,
        usecols=["Start_Lat", "Start_Lng"],
        chunksize=chunk_size
    ),
    start=1
):

    # Select 2% random records from each chunk
    sample = chunk.sample(
        frac=sample_fraction,
       random_state=42 + chunk_number
    )

    # Convert latitude and longitude into radians
    lat = np.radians(sample["Start_Lat"].to_numpy())
    lon = np.radians(sample["Start_Lng"].to_numpy())

    # Convert geographical coordinates into 3D
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    X = np.column_stack((x, y, z))

    sample_list.append(X)

    print(f"Sampling chunk {chunk_number} completed")

# Combine all random samples
X_sample = np.vstack(sample_list)

# Shuffle sample rows
rng = np.random.default_rng(42)
rng.shuffle(X_sample)

print("\nRANDOM SAMPLING COMPLETED")
print("Total sample records:", len(X_sample))

# Train MiniBatch K-Means
print("\nK-MEANS TRAINING STARTED")

model = MiniBatchKMeans(
    n_clusters=K,
    batch_size=10000,
    random_state=42,
    n_init=10
)

model.fit(X_sample)

# Save trained model
joblib.dump(model, model_path)

print("\nLOCALIZED K-MEANS TRAINING COMPLETED!")
print("Number of clusters:", K)
print("Training samples:", len(X_sample))
print("Model saved at:", model_path)

print("\nCLUSTER CENTRES (3D):")
print(model.cluster_centers_)
