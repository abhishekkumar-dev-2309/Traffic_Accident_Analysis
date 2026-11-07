
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Store model evaluation results
data = {
    "Model": [
        "Tuned Random Forest",
        "Temporal Random Forest"
    ],
    "Accuracy": [0.712694, 0.8745],
    "Balanced_Accuracy": [0.448158, 0.2652],
    "Macro_F1": [0.404904, 0.2637],
    "Weighted_F1": [0.721461, 0.8311]
}

df = pd.DataFrame(data)

print("========== MODEL COMPARISON ==========")
print(df.to_string(index=False))

# Step 2: Save comparison results
df.to_csv(
    "output/temporal_vs_random_model_comparison.csv",
    index=False
)

# Step 3: Prepare metrics for plotting
metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
    "Weighted_F1"
]

x = np.arange(len(metrics))
width = 0.35

# Step 4: Create comparison graph
plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    df.loc[0, metrics],
    width,
    label="Tuned Random Forest"
)

plt.bar(
    x + width / 2,
    df.loc[1, metrics],
    width,
    label="Temporal Random Forest"
)

plt.xticks(
    x,
    ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1"]
)

plt.ylabel("Score")
plt.title("Random Test vs Temporal Model Comparison")
plt.ylim(0, 1)

plt.legend()
plt.tight_layout()

# Step 5: Save graph
plt.savefig(
    "output/temporal_vs_random_model_comparison.png"
)

plt.show()

print("\nComparison files saved successfully!")
print("Program 42.17 completed successfully!")
