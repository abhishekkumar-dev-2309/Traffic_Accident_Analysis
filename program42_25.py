
import pandas as pd
import os

# Step 1: Create final project results
results = {
    "Category": [
        "Dataset",
        "Dataset",
        "Clustering",
        "Clustering",
        "Clustering",
        "Hotspot Analysis",
        "Hotspot Analysis",
        "Machine Learning",
        "Machine Learning",
        "Machine Learning",
        "Temporal Evaluation",
        "Temporal Evaluation",
        "Prediction System"
    ],

    "Parameter": [
        "Total Accident Records",
        "Total Dataset Columns",
        "K-Means K=3",
        "K-Means K=10",
        "K-Means K=15",
        "Grid Size",
        "Priority Weighting",
        "Selected Model Candidate",
        "Random Test Accuracy",
        "Random Test Macro F1",
        "Temporal Accuracy",
        "Temporal Macro F1",
        "Interactive Prediction"
    ],

    "Result": [
        "7,728,394",
        "46",
        "3 Clusters",
        "10 Clusters",
        "15 Clusters",
        "0.1 Degree",
        "50:50",
        "Tuned Random Forest",
        "71.27%",
        "40.49%",
        "87.45%",
        "26.37%",
        "Successfully Tested"
    ]
}

# Step 2: Convert into DataFrame
df = pd.DataFrame(results)

# Step 3: Display final results
print("\n========== FINAL PROJECT RESULTS ==========")
print(df.to_string(index=False))

# Step 4: Save consolidated results
os.makedirs("output", exist_ok=True)

output_path = "output/final_consolidated_project_results.csv"

df.to_csv(output_path, index=False)

print("\nFinal results saved successfully!")

# Step 5: Create important research findings
findings = [
    "The dataset contains 7,728,394 accident records.",
    "Geographic accident clustering was performed using K-Means.",
    "Grid-based analysis was used to identify accident hotspots.",
    "A 50:50 density and severity weighting strategy was used for priority scoring.",
    "Tuned Random Forest achieved 71.27% accuracy on the random test set.",
    "Temporal Random Forest achieved 87.45% accuracy on the later-period test set.",
    "Temporal model minority-class recall remains very low.",
    "Model accuracy should be interpreted alongside Macro F1 and Balanced Accuracy.",
    "The prediction system classifies accident severity, not accident occurrence probability."
]

with open(
    "output/final_research_findings.txt",
    "w",
    encoding="utf-8"
) as file:

    for number, finding in enumerate(findings, start=1):
        file.write(f"{number}. {finding}\n")

print("Research findings saved successfully!")

print("\nProgram 42.25 completed successfully!")
