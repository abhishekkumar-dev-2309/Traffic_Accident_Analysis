
# ==========================================
# PROGRAM 17: FULL DATASET MINI-BATCH K-MEANS
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# Find project folder
project_folder = Path(__file__).resolve().parent

# Input dataset
dataset_path = (
    project_folder / "output" / "full_accident_locations.csv"
)

# Output folder
output_folder = project_folder / "output"
output_folder.mkdir(exist_ok=True)

# Model output path
model_path = output_folder / "full_kmeans_model.pkl"

# Set number of clusters
K = 3

# Set batch size
chunk_size = 100000

# Create MiniBatchKMeans model
model = MiniBatchKMeans(
    n_clusters=K,
    batch_size=10000,
    random_state=42,
    n_init=10
)

print("FULL DATASET CLUSTERING STARTED")

# Read dataset in batches
for chunk_number, chunk in enumerate(
    pd.read_csv(
        dataset_path,
        usecols=['Start_Lat', 'Start_Lng'],
        chunksize=chunk_size
    ),
    start=1
):

    # Convert latitude and longitude into radians
    lat = np.radians(chunk['Start_Lat'].to_numpy())
    lon = np.radians(chunk['Start_Lng'].to_numpy())

    # Convert coordinates into 3D unit-sphere coordinates
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # Combine coordinates into feature array
    X = np.column_stack((x, y, z))

    # Train model incrementally
    model.partial_fit(X)

    print(
        f"Training chunk {chunk_number} completed"
    )

# Save trained model
joblib.dump(model, model_path)

print("\nFULL DATASET TRAINING COMPLETED!")
print("Number of clusters:", K)
print("Model saved at:", model_path)

# Display cluster centres
print("\nCLUSTER CENTRES (3D):")
print(model.cluster_centers_)
