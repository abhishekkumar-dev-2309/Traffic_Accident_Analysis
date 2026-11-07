
import pandas as pd

# Step 1: Store model performance
data = {
    "Model": [
        "Tuned Random Forest",
        "Temporal Random Forest"
    ],
    "Accuracy": [0.712694, 0.874500],
    "Balanced_Accuracy": [0.448158, 0.265200],
    "Macro_F1": [0.404904, 0.263700],
    "Weighted_F1": [0.721461, 0.831100]
}

df = pd.DataFrame(data)

# Step 2: Add model selection recommendation
df["Role"] = [
    "Primary Model Candidate",
    "Temporal Evaluation Model"
]

df["Selection_Reason"] = [
    "Higher Macro F1 and Balanced Accuracy",
    "Evaluated on later-period data (2022-2023)"
]

# Step 3: Display final selection
print("\n========== FINAL MODEL SELECTION ==========")
print(df.to_string(index=False))

# Step 4: Save results
df.to_csv(
    "output/final_model_selection.csv",
    index=False
)

# Step 5: Display selected model
print("\nPrimary Model Candidate:")
print("Tuned Random Forest")

print("\nImportant Limitation:")
print("Minority class prediction remains weak.")
print("Further validation is required before deployment.")

print("\nProgram 42.18 completed successfully!")
