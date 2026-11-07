
# ==========================================
# PROGRAM 10: ELBOW METHOD USING K-MEANS
# ==========================================

# Import Pandas for handling dataset
import pandas as pd

# Import Matplotlib for graph plotting
import matplotlib.pyplot as plt

# Import KMeans machine learning algorithm
from sklearn.cluster import KMeans

# Import Path for file locations
from pathlib import Path

# Find the main project folder
project_folder = Path(__file__).resolve().parent

# Load cleaned accident dataset
dataset_path = project_folder / "output" / "cleaned_accidents.csv"
df = pd.read_csv(dataset_path)

# Select latitude and longitude
X = df[['Start_Lat', 'Start_Lng']]

# Store WCSS values
wcss = []

# Test different cluster numbers from 1 to 10
for k in range(1, 11):

    # Create K-Means model
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # Train the model using location data
    model.fit(X)

    # Store the WCSS (inertia) value
    wcss.append(model.inertia_)

    print("K =", k, "WCSS =", model.inertia_)

# Plot the Elbow graph
plt.figure(figsize=(9, 5))

plt.plot(
    range(1, 11),
    wcss,
    marker='o'
)

# Add graph title and labels
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS (Inertia)")

# Save graph
output_path = project_folder / "output" / "elbow_method.png"
plt.savefig(output_path)

# Display graph
plt.show()

print("\nElbow Method completed successfully!")
print("Graph saved at:", output_path)
