
# ==========================================
# PROGRAM 27: INDEPENDENT K-MEANS EVALUATION
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

project_folder = Path(__file__).resolve().parent

input_path = (
    project_folder / "output" /
    "full_accident_locations.csv"
)

output_folder = project_folder / "output"

sample_list = []
chunk_size = 100000

K_values = [3, 5, 10, 15, 20]

print("COLLECTING RANDOM SAMPLES")

# Read dataset in chunks
for chunk_number, chunk in enumerate(
    pd.read_csv(
        input_path,
        usecols=["Start_Lat", "Start_Lng"],
        chunksize=chunk_size
    ),
    start=1
):

    sample = chunk.sample(
        frac=0.02,
        random_state=42 + chunk_number
    )

    lat = np.radians(sample["Start_Lat"].to_numpy())
    lon = np.radians(sample["Start_Lng"].to_numpy())

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    X = np.column_stack((x, y, z))

    sample_list.append(X)

    print(f"Chunk {chunk_number} completed")

# Combine samples
X_sample = np.vstack(sample_list)

# Split data into training and testing sets
X_train, X_test = train_test_split(
    X_sample,
    test_size=0.20,
    random_state=42
)

# Select 5000 independent test records
rng = np.random.default_rng(42)

test_indices = rng.choice(
    len(X_test),
    size=min(5000, len(X_test)),
    replace=False
)

X_eval = X_test[test_indices]

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Evaluation records:", len(X_eval))

results = []

# Compare different K values
for K in K_values:

    print(f"\nEvaluating K = {K}")

    model = MiniBatchKMeans(
        n_clusters=K,
        batch_size=10000,
        random_state=42,
        n_init=10
    )

    # Train only on training data
    model.fit(X_train)

    # Predict unseen evaluation records
    labels = model.predict(X_eval)

    # Calculate independent silhouette score
    score = silhouette_score(
        X_eval,
        labels,
        metric="euclidean"
    )

    results.append({
        "K": K,
        "Inertia": model.inertia_,
        "Silhouette_Score": score
    })

    print("Inertia:", round(model.inertia_, 4))
    print("Silhouette Score:", round(score, 4))

# Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    output_folder / "independent_k_evaluation.csv",
    index=False
)

print("\nFINAL INDEPENDENT EVALUATION")
print(results_df.to_string(index=False))

# Plot silhouette comparison
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["K"],
    results_df["Silhouette_Score"],
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Independent K-Means Evaluation")
plt.grid(True)

plt.savefig(
    output_folder / "independent_k_evaluation.png",
    dpi=300
)

plt.show()

print("\nEVALUATION COMPLETED!")
