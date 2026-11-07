
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Create dashboard figure
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

fig.suptitle(
    "TRAFFIC ACCIDENT ANALYSIS AND PREDICTION DASHBOARD",
    fontsize=17,
    fontweight="bold"
)

# Step 2: Accident severity distribution
severity = [67366, 6156981, 1299337, 204710]
labels = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]

axes[0, 0].pie(
    severity,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

axes[0, 0].set_title("Accident Severity Distribution")

# Step 3: Top 10 accident states
states = [
    "CA", "FL", "TX", "SC", "NY",
    "NC", "VA", "PA", "MN", "OR"
]

accidents = [
    1741433, 880192, 582837, 382557, 347960,
    338199, 303301, 296620, 192084, 179660
]

axes[0, 1].barh(states[::-1], accidents[::-1])

axes[0, 1].set_title("Top 10 Accident States")
axes[0, 1].set_xlabel("Number of Accidents")

# Step 4: Top 10 priority hotspots
hotspots = [
    "33.65, -84.45",
    "38.85, -76.95",
    "33.75, -84.35",
    "33.75, -84.45",
    "41.85, -87.65",
    "26.05, -80.15",
    "39.25, -76.65",
    "39.95, -75.15",
    "33.85, -84.25",
    "33.75, -84.25"
]

scores = [
    91.67, 91.38, 90.80, 90.69, 90.25,
    90.21, 89.83, 89.34, 88.68, 88.28
]

axes[1, 0].barh(hotspots[::-1], scores[::-1])

axes[1, 0].set_title("Top 10 Priority Hotspots")
axes[1, 0].set_xlabel("Priority Score")
axes[1, 0].set_xlim(0, 100)

# Step 5: Model performance comparison
models = ["Tuned RF", "Temporal RF"]

accuracy = [71.2694, 87.45]
macro_f1 = [40.4904, 26.37]

x = np.arange(len(models))
width = 0.35

axes[1, 1].bar(
    x - width / 2,
    accuracy,
    width,
    label="Accuracy"
)

axes[1, 1].bar(
    x + width / 2,
    macro_f1,
    width,
    label="Macro F1"
)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(models)
axes[1, 1].set_ylim(0, 100)
axes[1, 1].set_ylabel("Score (%)")
axes[1, 1].set_title("Model Performance Comparison")
axes[1, 1].legend()

# Step 6: Add total accident records
fig.text(
    0.5, 0.02,
    "Total Accident Records: 7,728,394 | "
    "Analysis: US Accidents Dataset",
    ha="center",
    fontsize=11
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Step 7: Save dashboard
plt.savefig(
    "output/final_project_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFinal dashboard generated successfully!")
print("File: output/final_project_dashboard.png")

# Step 8: Save project summary
summary = pd.DataFrame({
    "Project_Metric": [
        "Total Accident Records",
        "Primary Model Accuracy",
        "Primary Model Macro F1",
        "Temporal Model Accuracy",
        "Temporal Model Macro F1",
        "Number of K10 Clusters",
        "Number of K15 Clusters"
    ],
    "Value": [
        7728394,
        0.712694,
        0.404904,
        0.8745,
        0.2637,
        10,
        15
    ]
})

summary.to_csv(
    "output/final_project_summary.csv",
    index=False
)

print("Project summary saved successfully!")
print("Program 42.23 completed successfully!")
