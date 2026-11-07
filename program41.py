
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------
# STEP 1: Load hotspot data
# ---------------------------------------

input_file = os.path.join(
    "output",
    "combined_hotspot_classification.csv"
)

output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

data = pd.read_csv(input_file)

# Use the same minimum-record threshold as Program 40
data = data[
    data["Total_Accidents"] >= 1000
].copy()

print("Total grid cells:", len(data))

# ---------------------------------------
# STEP 2: Calculate percentile scores
# ---------------------------------------

data["Density_Score"] = (
    data["Total_Accidents"].rank(pct=True) * 100
)

data["Severity_Score"] = (
    data["Severity_4_Percentage"].rank(pct=True) * 100
)

# ---------------------------------------
# STEP 3: Calculate three weighted models
# ---------------------------------------

# Model A: Equal weights
data["Score_50_50"] = (
    0.50 * data["Density_Score"]
    + 0.50 * data["Severity_Score"]
)

# Model B: Density receives more weight
data["Score_60_40"] = (
    0.60 * data["Density_Score"]
    + 0.40 * data["Severity_Score"]
)

# Model C: Severity receives more weight
data["Score_40_60"] = (
    0.40 * data["Density_Score"]
    + 0.60 * data["Severity_Score"]
)

# ---------------------------------------
# STEP 4: Generate rankings
# ---------------------------------------

score_columns = [
    "Score_50_50",
    "Score_60_40",
    "Score_40_60"
]

for column in score_columns:

    data[column.replace("Score", "Rank")] = (
        data[column]
        .rank(method="min", ascending=False)
        .astype(int)
    )

# ---------------------------------------
# STEP 5: Select top 20 locations
# ---------------------------------------

top_50_50 = data.nsmallest(20, "Rank_50_50")
top_60_40 = data.nsmallest(20, "Rank_60_40")
top_40_60 = data.nsmallest(20, "Rank_40_60")

# Save separate rankings
top_50_50.to_csv(
    os.path.join(output_folder, "top20_model_50_50.csv"),
    index=False
)

top_60_40.to_csv(
    os.path.join(output_folder, "top20_model_60_40.csv"),
    index=False
)

top_40_60.to_csv(
    os.path.join(output_folder, "top20_model_40_60.csv"),
    index=False
)

# ---------------------------------------
# STEP 6: Compare top 20 overlap
# ---------------------------------------

def get_location_set(frame):

    return set(
        zip(
            frame["Latitude"],
            frame["Longitude"]
        )
    )


set_50 = get_location_set(top_50_50)
set_60 = get_location_set(top_60_40)
set_40 = get_location_set(top_40_60)

comparison = pd.DataFrame({
    "Model_Comparison": [
        "50:50 vs 60:40",
        "50:50 vs 40:60",
        "60:40 vs 40:60"
    ],
    "Common_Top20_Locations": [
        len(set_50 & set_60),
        len(set_50 & set_40),
        len(set_60 & set_40)
    ]
})

comparison["Jaccard_Similarity"] = [
    len(set_50 & set_60) / len(set_50 | set_60),
    len(set_50 & set_40) / len(set_50 | set_40),
    len(set_60 & set_40) / len(set_60 | set_40)
]

comparison.to_csv(
    os.path.join(
        output_folder,
        "priority_model_overlap.csv"
    ),
    index=False
)

# ---------------------------------------
# STEP 7: Compare ranking correlations
# ---------------------------------------

rank_columns = [
    "Rank_50_50",
    "Rank_60_40",
    "Rank_40_60"
]

rank_correlation = data[rank_columns].corr(
    method="spearman"
)

rank_correlation.to_csv(
    os.path.join(
        output_folder,
        "priority_rank_correlation.csv"
    )
)

# ---------------------------------------
# STEP 8: Print results
# ---------------------------------------

print("\nPRIORITY SENSITIVITY ANALYSIS COMPLETED!")

print("\nTop 20 Overlap Comparison:")
print(comparison.round(3).to_string(index=False))

print("\nSpearman Rank Correlation:")
print(rank_correlation.round(4))

print("\nTop 10 Locations – Model A (50:50):")

print(
    top_50_50[
        [
            "Latitude",
            "Longitude",
            "Rank_50_50",
            "Score_50_50"
        ]
    ].head(10).round(2).to_string(index=False)
)

# ---------------------------------------
# STEP 9: Visualize top 20 overlap
# ---------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model_Comparison"],
    comparison["Common_Top20_Locations"]
)

plt.title("Top 20 Hotspot Ranking Stability")
plt.xlabel("Model Comparison")
plt.ylabel("Common Locations")
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "priority_model_stability.png"
    )
)

plt.show()

# ---------------------------------------
# STEP 10: Compare score distributions
# ---------------------------------------

plt.figure(figsize=(10, 6))

for column in score_columns:

    plt.hist(
        data[column],
        bins=30,
        alpha=0.4,
        label=column
    )

plt.title("Priority Score Distribution Across Models")
plt.xlabel("Priority Score")
plt.ylabel("Number of Grid Cells")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "priority_score_comparison.png"
    )
)

plt.show()

print("\nAll sensitivity analysis files saved successfully!")
