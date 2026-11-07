
# ==========================================
# PROGRAM 14: FINAL CLUSTER SUMMARY
# ==========================================

import pandas as pd
from pathlib import Path

# Find project directory
project_folder = Path(__file__).resolve().parent

# Load clustered dataset
dataset_path = project_folder / "output" / "clustered_accidents.csv"
df = pd.read_csv(dataset_path)

# Calculate total accidents in each cluster
total_accidents = df.groupby('Cluster').size()

# Calculate percentage of accidents
percentage = (total_accidents / len(df)) * 100

# Calculate average severity
average_severity = df.groupby('Cluster')['Severity'].mean()

# Calculate average latitude
latitude = df.groupby('Cluster')['Start_Lat'].mean()

# Calculate average longitude
longitude = df.groupby('Cluster')['Start_Lng'].mean()

# Create summary table
summary = pd.DataFrame({
    'Total Accidents': total_accidents,
    'Accident Percentage': percentage,
    'Average Severity': average_severity,
    'Average Latitude': latitude,
    'Average Longitude': longitude
})

# Add severity-wise accident counts
severity_counts = pd.crosstab(
    df['Cluster'],
    df['Severity']
)

severity_counts.columns = [
    f'Severity {level} Count'
    for level in severity_counts.columns
]

# Combine summary and severity counts
summary = summary.join(severity_counts)

# Display final report
print("\n========== FINAL CLUSTER SUMMARY ==========")
print(summary.round(3))

# Save summary report
output_path = project_folder / "output" / "final_cluster_summary.csv"

summary.to_csv(output_path)

print("\nFinal cluster summary generated successfully!")
print("Report saved at:", output_path)
