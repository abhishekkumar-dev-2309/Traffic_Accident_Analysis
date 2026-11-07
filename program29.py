
import pandas as pd
import numpy as np
import joblib

from sklearn.cluster import MiniBatchKMeans

# ----------------------------------
# STEP 1: DATASET AND MODEL PATHS
# ----------------------------------

data_path = "output/full_accident_locations.csv"

# Load existing K=10 model
model_10 = joblib.load(
    "output/localized_kmeans_model.pkl"
)

# ----------------------------------
# STEP 2: PREPARE TRAINING SAMPLE
# ----------------------------------

chunks = []
chunk_number = 0

print("Preparing training sample...")

for chunk in pd.read_csv(
    data_path,
    chunksize=100000
):

    sample = chunk.sample(
        frac=0.02,
        random_state=42 + chunk_number
    )

    chunks.append(sample)
    chunk_number += 1

data = pd.concat(chunks, ignore_index=True)

print("Training sample:", len(data))

# Convert latitude and longitude into radians
lat = np.radians(data["Start_Lat"].to_numpy())
lon = np.radians(data["Start_Lng"].to_numpy())

# Convert coordinates into 3D unit-sphere coordinates
X = np.column_stack((
    np.cos(lat) * np.cos(lon),
    np.cos(lat) * np.sin(lon),
    np.sin(lat)
))

# ----------------------------------
# STEP 3: TRAIN K=15 MODEL
# ----------------------------------

print("\nTraining K=15 model...")

model_15 = MiniBatchKMeans(
    n_clusters=15,
    batch_size=10000,
    random_state=42,
    n_init=10
)

model_15.fit(X)

joblib.dump(
    model_15,
    "output/k15_kmeans_model.pkl"
)

# ----------------------------------
# STEP 4: INITIALIZE STATISTICS
# ----------------------------------

models = {
    10: model_10,
    15: model_15
}

statistics = {}

for k in models:

    statistics[k] = {
        "count": np.zeros(k, dtype=np.int64),
        "severity_sum": np.zeros(k),
        "severity_count": np.zeros((k, 4), dtype=np.int64),
        "x_sum": np.zeros(k),
        "y_sum": np.zeros(k),
        "z_sum": np.zeros(k)
    }

# ----------------------------------
# STEP 5: PROCESS FULL DATASET
# ----------------------------------

print("\nAnalyzing full dataset...")

total_records = 0

for chunk in pd.read_csv(
    data_path,
    chunksize=100000
):

    lat = np.radians(
        chunk["Start_Lat"].to_numpy()
    )

    lon = np.radians(
        chunk["Start_Lng"].to_numpy()
    )

    X_chunk = np.column_stack((
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat)
    ))

    severity = chunk["Severity"].to_numpy()

    # Predict clusters using both models
    for k, model in models.items():

        labels = model.predict(X_chunk)

        stats = statistics[k]

        # Count accidents and severity
        stats["count"] += np.bincount(
            labels,
            minlength=k
        )

        stats["severity_sum"] += np.bincount(
            labels,
            weights=severity,
            minlength=k
        )

        # Count each severity category
        for level in range(1, 5):

            mask = severity == level

            stats["severity_count"][:, level - 1] += (
                np.bincount(
                    labels[mask],
                    minlength=k
                )
            )

        # Sum 3D coordinates for geographic centroid
        stats["x_sum"] += np.bincount(
            labels,
            weights=X_chunk[:, 0],
            minlength=k
        )

        stats["y_sum"] += np.bincount(
            labels,
            weights=X_chunk[:, 1],
            minlength=k
        )

        stats["z_sum"] += np.bincount(
            labels,
            weights=X_chunk[:, 2],
            minlength=k
        )

    total_records += len(chunk)

    print(
        "Processed records:",
        total_records
    )

# ----------------------------------
# STEP 6: CREATE SUMMARY
# ----------------------------------

all_summaries = []

for k, stats in statistics.items():

    for cluster in range(k):

        count = stats["count"][cluster]

        if count == 0:
            continue

        # Normalize average 3D coordinate
        x = stats["x_sum"][cluster] / count
        y = stats["y_sum"][cluster] / count
        z = stats["z_sum"][cluster] / count

        magnitude = np.sqrt(x*x + y*y + z*z)

        x /= magnitude
        y /= magnitude
        z /= magnitude

        # Convert centroid back to latitude/longitude
        latitude = np.degrees(
            np.arcsin(np.clip(z, -1, 1))
        )

        longitude = np.degrees(
            np.arctan2(y, x)
        )

        severity_counts = (
            stats["severity_count"][cluster]
        )

        average_severity = (
            stats["severity_sum"][cluster] / count
        )

        severity_4_percentage = (
            severity_counts[3] / count
        ) * 100

        all_summaries.append({

            "K": k,
            "Cluster": cluster,
            "Total_Accidents": count,
            "Percentage": count / total_records * 100,
            "Latitude": latitude,
            "Longitude": longitude,
            "Average_Severity": average_severity,
            "Severity_1": severity_counts[0],
            "Severity_2": severity_counts[1],
            "Severity_3": severity_counts[2],
            "Severity_4": severity_counts[3],
            "Severity_4_Percentage": severity_4_percentage
        })

# ----------------------------------
# STEP 7: SAVE RESULTS
# ----------------------------------

summary_df = pd.DataFrame(all_summaries)

summary_df.to_csv(
    "output/k10_k15_comparison.csv",
    index=False
)

print("\nCLUSTER COMPARISON COMPLETED!")
print(summary_df.to_string(index=False))

print("\nTotal records analyzed:", total_records)
print("Output: output/k10_k15_comparison.csv")
