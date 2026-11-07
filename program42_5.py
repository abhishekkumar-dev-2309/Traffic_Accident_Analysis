
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# --------------------------------------------------
# STEP 1: Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

model_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

# --------------------------------------------------
# STEP 2: Load trained model
# --------------------------------------------------

print("Loading trained Random Forest model...")

model = joblib.load(model_file)

# --------------------------------------------------
# STEP 3: Extract preprocessing and classifier
# --------------------------------------------------

preprocessor = model.named_steps["preprocessing"]
classifier = model.named_steps["classifier"]

# --------------------------------------------------
# STEP 4: Get transformed feature names
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

# --------------------------------------------------
# STEP 5: Extract feature importance
# --------------------------------------------------

importance_values = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
})

# --------------------------------------------------
# STEP 6: Sort features by importance
# --------------------------------------------------

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Important Features:")
print(importance_df.head(20).to_string(index=False))

# --------------------------------------------------
# STEP 7: Save feature importance
# --------------------------------------------------

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# --------------------------------------------------
# STEP 8: Plot top 20 features
# --------------------------------------------------

top_features = importance_df.head(20).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 8))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest - Top 20 Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature importance analysis completed!")
print("Files saved successfully in output folder.")
