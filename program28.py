import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

# Dataset path
file_path = "output/full_accident_locations.csv"

# Read dataset in chunks
chunks = []
chunk_number = 0

for chunk in pd.read_csv(file_path, chunksize=100000):

    sample = chunk.sample(
        frac=0.02,
        random_state=42 + chunk_number
    )

    chunks.append(sample)
    chunk_number += 1

# Combine sampled data
data = pd.concat(chunks, ignore_index=True)

print("Total sampled records:", len(data))

# Convert latitude and longitude into radians
lat = np.radians(data["Start_Lat"].to_numpy())
lon = np.radians(data["Start_Lng"].to_numpy())

# Convert geographic coordinates into 3D coordinates
X = np.column_stack((
    np.cos(lat) * np.cos(lon),
    np.cos(lat) * np.sin(lon),
    np.sin(lat)
))

# Split data into training and testing sets
X_train, X_test = train_test_split(
    X,
    test_size=0.20,
    random_state=42
)

# Select a fixed evaluation sample
rng = np.random.default_rng(42)

indices = rng.choice(
    len(X_test),
    size=5000,
    replace=False
)

X_eval = X_test[indices]

# Test different random seeds
seeds = [21, 42, 63, 84, 105]

results = []

for k in [10, 15]:

    for seed in seeds:

        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=10000,
            random_state=seed,
            n_init=10
        )

        # Train model
        model.fit(X_train)

        # Predict test labels
        labels = model.predict(X_eval)

        # Calculate silhouette score
        score = silhouette_score(
            X_eval,
            labels
        )

        results.append({
            "K": k,
            "Random_Seed": seed,
            "Silhouette_Score": score
        })

        print(
            f"K={k}, Seed={seed}, "
            f"Silhouette Score={score:.6f}"
        )

# Convert results into DataFrame
results_df = pd.DataFrame(results)

# Save individual results
results_df.to_csv(
    "output/clustering_stability_results.csv",
    index=False
)

# Calculate average and standard deviation
summary = results_df.groupby("K")[
    "Silhouette_Score"
].agg(["mean", "std"])

summary.to_csv(
    "output/clustering_stability_summary.csv"
)

print("\nSTABILITY SUMMARY")
print(summary)

# Plot comparison
summary.plot(
    kind="bar",
    y="mean",
    yerr="std",
    capsize=5,
    legend=False
)

plt.title("Clustering Stability: K=10 vs K=15")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Mean Silhouette Score")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "output/clustering_stability.png"
)

plt.show()

print("\nPROGRAM 28 COMPLETED SUCCESSFULLY!")