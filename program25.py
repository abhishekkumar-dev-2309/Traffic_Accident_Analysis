
# ==========================================
# PROGRAM 25: SILHOUETTE SCORE EVALUATION
# ==========================================

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import silhouette_score

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = (
    project_folder / "output" /
    "localized_clustered_accidents.csv"
)

output_path = (
    project_folder / "output" /
    "silhouette_evaluation.csv"
)

chunk_size = 100000
sample_list = []

print("SILHOUETTE EVALUATION STARTED")

# Read dataset in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(input_path, chunksize=chunk_size),
    start=1
):

    # Select random records from each chunk
    sample = chunk.sample(
        frac=0.02,
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

    # Get cluster labels
    labels = sample["Cluster"].to_numpy()

    # Store coordinates and labels
    sample_list.append(
        np.column_stack((X, labels))
    )

    print(f"Chunk {chunk_number} sampled")

# Combine all samples
all_samples = np.vstack(sample_list)

# Shuffle the sample
rng = np.random.default_rng(42)
rng.shuffle(all_samples)

# Select 5000 records for evaluation
evaluation_data = all_samples[:5000]

# Separate coordinates and cluster labels
X_eval = evaluation_data[:, :3]
labels_eval = evaluation_data[:, 3].astype(int)

print("\nEvaluation records:", len(X_eval))
print("Clusters represented:", np.unique(labels_eval))

# Calculate Silhouette Score
score = silhouette_score(
    X_eval,
    labels_eval,
    metric="euclidean"
)

# Save result
result = pd.DataFrame({
    "Metric": ["Silhouette Score"],
    "Value": [score],
    "Evaluation_Samples": [len(X_eval)],
    "Number_of_Clusters": [len(np.unique(labels_eval))]
})

result.to_csv(output_path, index=False)

print("\nSILHOUETTE EVALUATION COMPLETED!")
print("Silhouette Score:", round(score, 4))
print("Result saved at:", output_path)
