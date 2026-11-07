
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------
# STEP 1: LOAD CLUSTER DATA
# ----------------------------------

file_path = "output/k10_k15_comparison.csv"

data = pd.read_csv(file_path)

# Separate K=10 and K=15 data
k10 = data[data["K"] == 10].copy()
k15 = data[data["K"] == 15].copy()

print("K=10 clusters:", len(k10))
print("K=15 clusters:", len(k15))

# ----------------------------------
# STEP 2: ACCIDENT COUNT GRAPH
# ----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    k10["Cluster"].astype(str),
    k10["Total_Accidents"]
)

plt.title("Accident Distribution Across K=10 Clusters")
plt.xlabel("Cluster Number")
plt.ylabel("Total Accidents")
plt.tight_layout()

plt.savefig("output/k10_accident_distribution.png")
plt.show()

# K=15 graph
plt.figure(figsize=(12, 6))

plt.bar(
    k15["Cluster"].astype(str),
    k15["Total_Accidents"]
)

plt.title("Accident Distribution Across K=15 Clusters")
plt.xlabel("Cluster Number")
plt.ylabel("Total Accidents")
plt.tight_layout()

plt.savefig("output/k15_accident_distribution.png")
plt.show()

# ----------------------------------
# STEP 3: SEVERITY 4 PERCENTAGE
# ----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    k10["Cluster"].astype(str),
    k10["Severity_4_Percentage"]
)

plt.title("Severity 4 Percentage - K=10")
plt.xlabel("Cluster Number")
plt.ylabel("Severity 4 Percentage (%)")
plt.tight_layout()

plt.savefig("output/k10_severity4_comparison.png")
plt.show()

# K=15 severity graph
plt.figure(figsize=(12, 6))

plt.bar(
    k15["Cluster"].astype(str),
    k15["Severity_4_Percentage"]
)

plt.title("Severity 4 Percentage - K=15")
plt.xlabel("Cluster Number")
plt.ylabel("Severity 4 Percentage (%)")
plt.tight_layout()

plt.savefig("output/k15_severity4_comparison.png")
plt.show()

# ----------------------------------
# STEP 4: AVERAGE SEVERITY
# ----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(
    k10["Cluster"].astype(str),
    k10["Average_Severity"]
)

plt.title("Average Severity - K=10")
plt.xlabel("Cluster Number")
plt.ylabel("Average Severity")
plt.tight_layout()

plt.savefig("output/k10_average_severity.png")
plt.show()

plt.figure(figsize=(12, 6))

plt.bar(
    k15["Cluster"].astype(str),
    k15["Average_Severity"]
)

plt.title("Average Severity - K=15")
plt.xlabel("Cluster Number")
plt.ylabel("Average Severity")
plt.tight_layout()

plt.savefig("output/k15_average_severity.png")
plt.show()

# ----------------------------------
# STEP 5: PRINT IMPORTANT FINDINGS
# ----------------------------------

print("\nK=10 HIGHEST ACCIDENT CLUSTER")
print(
    k10.loc[
        k10["Total_Accidents"].idxmax()
    ]
)

print("\nK=15 HIGHEST ACCIDENT CLUSTER")
print(
    k15.loc[
        k15["Total_Accidents"].idxmax()
    ]
)

print("\nK=10 HIGHEST SEVERITY 4 PERCENTAGE")
print(
    k10.loc[
        k10["Severity_4_Percentage"].idxmax()
    ]
)

print("\nK=15 HIGHEST SEVERITY 4 PERCENTAGE")
print(
    k15.loc[
        k15["Severity_4_Percentage"].idxmax()
    ]
)

print("\nPROGRAM 31 COMPLETED SUCCESSFULLY!")
