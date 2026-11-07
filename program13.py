
# ==========================================
# PROGRAM 13: CLUSTER-WISE SEVERITY ANALYSIS
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# Find project folder
project_folder = Path(__file__).resolve().parent

# Load clustered accident dataset
dataset_path = project_folder / "output" / "clustered_accidents.csv"

df = pd.read_csv(dataset_path)

# Calculate average severity for each cluster
severity_analysis = df.groupby('Cluster')['Severity'].mean()

print("\nAVERAGE SEVERITY BY CLUSTER:")
print(severity_analysis)

# Count severity levels in each cluster
severity_counts = pd.crosstab(
    df['Cluster'],
    df['Severity']
)

print("\nSEVERITY DISTRIBUTION:")
print(severity_counts)

# Plot severity distribution
severity_counts.plot(
    kind='bar',
    figsize=(9, 5)
)

plt.title("Accident Severity by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.legend(title="Severity")

# Save graph
output_path = project_folder / "output" / "cluster_severity.png"

plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.show()

print("\nCluster severity analysis completed!")
print("Graph saved at:", output_path)
