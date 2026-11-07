
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score

# --------------------------------------------------
# STEP 1: Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

test_file = OUTPUT_DIR / "testing_dataset.csv"
model_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

# --------------------------------------------------
# STEP 2: Load model and testing data
# --------------------------------------------------

print("Loading trained model...")
model = joblib.load(model_file)

print("Loading testing dataset...")
test_data = pd.read_csv(test_file)

X_test = test_data.drop(columns=["Severity"])
y_test = test_data["Severity"]

# Convert Boolean features into integers
boolean_features = [
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity"
]

for column in boolean_features:
    X_test[column] = X_test[column].astype("int8")

# --------------------------------------------------
# STEP 3: Select a sample for faster evaluation
# --------------------------------------------------

sample_size = min(10000, len(X_test))

X_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

y_sample = y_test.loc[X_sample.index]

print("\nSample records:", len(X_sample))

# --------------------------------------------------
# STEP 4: Calculate permutation importance
# --------------------------------------------------

print("\nCalculating permutation importance...")
print("Please wait...")

result = permutation_importance(
    model,
    X_sample,
    y_sample,
    scoring="balanced_accuracy",
    n_repeats=3,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# STEP 5: Create importance table
# --------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X_sample.columns,
    "Importance_Mean": result.importances_mean,
    "Importance_Std": result.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
)

print("\nTop 15 Permutation Important Features:")
print(importance_df.head(15).to_string(index=False))

# --------------------------------------------------
# STEP 6: Save results
# --------------------------------------------------

importance_df.to_csv(
    OUTPUT_DIR / "permutation_feature_importance.csv",
    index=False
)

# --------------------------------------------------
# STEP 7: Plot top 15 features
# --------------------------------------------------

top_features = importance_df.head(15).sort_values(
    by="Importance_Mean",
    ascending=True
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance_Mean",
    y="Feature"
)

plt.title("Permutation Feature Importance")
plt.xlabel("Decrease in Balanced Accuracy")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "permutation_feature_importance.png",
    dpi=300
)

plt.close()

print("\nPermutation analysis completed!")
print("Results saved successfully.")
