
# ==========================================
# PROGRAM 11: K-MEANS ACCIDENT CLUSTERING
# ==========================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from pathlib import Path

# Find project folder
project_folder = Path(__file__).resolve().parent

# Load cleaned accident dataset
dataset_path = project_folder / "output" / "cleaned_accidents.csv"
df = pd.read_csv(dataset_path)

# Select latitude and longitude
locations = df[['Start_Lat', 'Start_Lng']].copy()

# Convert latitude and longitude into radians
# Radians are required for trigonometric functions
lat = np.radians(locations['Start_Lat'].to_numpy())
lon = np.radians(locations['Start_Lng'].to_numpy())

# Convert geographical coordinates into 3D unit-sphere coordinates
x = np.cos(lat) * np.cos(lon)
y = np.cos(lat) * np.sin(lon)
z = np.sin(lat)

# Create 3D coordinate array
X = np.column_stack((x, y, z))

# Select number of clusters
K = 3

# Create K-Means model
kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

# Train the model and assign cluster labels
df['Cluster'] = kmeans.fit_predict(X)

# Count accidents in each cluster
cluster_counts = df['Cluster'].value_counts().sort_index()

print("\nACCIDENT COUNT IN EACH CLUSTER:")
print(cluster_counts)

# Convert cluster centroids back to latitude and longitude
centroids = kmeans.cluster_centers_

# Normalize centroids to unit length
centroids = centroids / np.linalg.norm(
    centroids, axis=1, keepdims=True
)

# Calculate centroid latitude
centroid_lat = np.degrees(
    np.arcsin(centroids[:, 2])
)

# Calculate centroid longitude
centroid_lon = np.degrees(
    np.arctan2(centroids[:, 1], centroids[:, 0])
)

# Display centroid locations
centroid_data = pd.DataFrame({
    'Cluster': range(K),
    'Latitude': centroid_lat,
    'Longitude': centroid_lon
})

print("\nCLUSTER CENTROIDS:")
print(centroid_data)

# Save accident records with cluster labels
output_path = project_folder / "output" / "clustered_accidents.csv"
df.to_csv(output_path, index=False)

# Visualize accident clusters
plt.figure(figsize=(10, 6))

plt.scatter(
    df['Start_Lng'],
    df['Start_Lat'],
    c=df['Cluster'],
    cmap='viridis',
    s=15,
    alpha=0.6
)

# Plot cluster centroids
plt.scatter(
    centroid_lon,
    centroid_lat,
    marker='X',
    color='red',
    s=200,
    label='Centroids'
)

plt.title("Traffic Accident Clusters using K-Means")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# Save cluster visualization
graph_path = project_folder / "output" / "kmeans_clusters.png"
plt.savefig(graph_path, dpi=300)

plt.show()

print("\nK-Means clustering completed successfully!")
print("Clustered dataset saved:", output_path)
print("Cluster graph saved:", graph_path)
