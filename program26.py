
# ==========================================
# PROGRAM 26: OPTIMAL K VALUE COMPARISON
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Project paths
project_folder = Path(__file__).resolve().parent

input_path = (
    project_folder / "output" /
    "full_accident_locations.csv"
)

output_folder = project_folder / "output"

chunk_size = 100000
sample_fraction = 0.02

K_values = [3, 5, 10, 15, 20]

sample_list = []

print("OPTIMAL K VALUE ANALYSIS STARTED")

# Collect representative random samples
for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_path,
        usecols=["Start_Lat", "Start_Lng"],
        chunksize=chunk_size
    ),
    start=1
):

    sample = chunk.sample(
        frac=sample_fraction,
        random_state=42 + chunk_number
    )

    lat = np.radians(sample["Start_Lat"].to_numpy())
    lon = np.radians(sample["Start_Lng"].to_numpy())

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    X = np.column_stack((x, y, z))

    sample_list.append(X)

    print(f"Chunk {chunk_number} sampled")

# Combine samples
X_sample = np.vstack(sample_list)

# Shuffle samples
rng = np.random.default_rng(42)
rng.shuffle(X_sample)

# Fixed evaluation sample
X_eval = X_sample[:5000]

results = []

print("\nCOMPARING DIFFERENT K VALUES")

for K in K_values:

    print(f"\nTraining model for K = {K}")

    model = MiniBatchKMeans(
        n_clusters=K,
        batch_size=10000,
        random_state=42,
        n_init=10
    )

    # Train model
    model.fit(X_sample)

    # Predict evaluation sample labels
    labels = model.predict(X_eval)

    # Calculate silhouette score
    score = silhouette_score(
        X_eval,
        labels,
        metric="euclidean"
    )

    # Store results
    results.append({
        "K": K,
        "Inertia": model.inertia_,
        "Silhouette_Score": score
    })

    print("Inertia:", round(model.inertia_, 4))
    print("Silhouette Score:", round(score, 4))

# Create result table
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv(
    output_folder / "optimal_k_comparison.csv",
    index=False
)

print("\nFINAL COMPARISON")
print(results_df.to_string(index=False))

# Plot Elbow Method
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["Inertia"],
    marker="o"
)

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)

plt.savefig(
    output_folder / "optimal_k_elbow.png",
    dpi=300
)

plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["Silhouette_Score"],
    marker="o"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)

plt.savefig(
    output_folder / "optimal_k_silhouette.png",
    dpi=300
)

plt.show()

print("\nGRAPHS SAVED SUCCESSFULLY!")

