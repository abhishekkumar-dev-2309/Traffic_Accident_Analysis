
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load confusion matrix
cm = pd.read_csv(
    "output/temporal_error_confusion_matrix.csv",
    index_col=0
)

print("Confusion Matrix:")
print(cm)

# Step 2: Plot confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Temporal Model - Confusion Matrix")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")

plt.tight_layout()
plt.savefig("output/temporal_confusion_matrix_heatmap.png")
plt.show()


# Step 3: Load class-wise prediction data
error_data = pd.read_csv(
    "output/temporal_error_analysis.csv"
)

print("\nClass-wise Error Analysis:")
print(error_data.head())

# Step 4: Calculate class-wise recall
class_recall = (
    error_data.groupby("Actual_Severity")["Correct_Prediction"]
    .mean()
)

print("\nClass-wise Recall:")
print(class_recall)

# Step 5: Plot class-wise recall
plt.figure(figsize=(8, 5))

plt.bar(
    class_recall.index,
    class_recall.values
)

plt.title("Class-wise Recall of Temporal Model")
plt.xlabel("Severity Class")
plt.ylabel("Recall")
plt.xticks([1, 2, 3, 4])
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("output/temporal_classwise_recall.png")
plt.show()

print("\nProgram 42.16 completed successfully!")
