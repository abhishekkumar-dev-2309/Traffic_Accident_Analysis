
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------
# STEP 1: DATASET PATH
# ------------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"
# Store state-wise statistics
state_summary = pd.DataFrame()

chunk_number = 0
total_records = 0

print("Starting State-wise Accident Analysis...")

# ------------------------------------
# STEP 2: READ DATASET IN CHUNKS
# ------------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=["State", "Severity"],
    chunksize=100000
):

    # Remove records with missing state
    chunk = chunk.dropna(subset=["State"])

    # Count accidents by State and Severity
    summary = pd.crosstab(
        chunk["State"],
        chunk["Severity"]
    )

    # Ensure all severity columns exist
    summary = summary.reindex(
        columns=[1, 2, 3, 4],
        fill_value=0
    )

    # Combine current chunk with previous results
    state_summary = state_summary.add(
        summary,
        fill_value=0
    )

    total_records += len(chunk)
    chunk_number += 1

    print(
        "Processed chunks:",
        chunk_number,
        "| Records:",
        total_records
    )

# ------------------------------------
# STEP 3: CALCULATE TOTAL ACCIDENTS
# ------------------------------------

state_summary = state_summary.fillna(0)

state_summary.columns = [
    "Severity_1",
    "Severity_2",
    "Severity_3",
    "Severity_4"
]

state_summary["Total_Accidents"] = (
    state_summary[
        ["Severity_1", "Severity_2",
         "Severity_3", "Severity_4"]
    ].sum(axis=1)
)

# Calculate Severity 4 percentage
state_summary["Severity_4_Percentage"] = (
    state_summary["Severity_4"]
    / state_summary["Total_Accidents"]
) * 100

# Sort by total accident count
state_summary = state_summary.sort_values(
    by="Total_Accidents",
    ascending=False
)

# ------------------------------------
# STEP 4: SAVE STATE-WISE REPORT
# ------------------------------------

state_summary.to_csv(
    "state_wise_accident_analysis.csv"
)

print("\nSTATE-WISE ACCIDENT ANALYSIS")
print(state_summary.to_string())

# ------------------------------------
# STEP 5: TOP 10 STATES GRAPH
# ------------------------------------

top_10 = state_summary.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10.index,
    top_10["Total_Accidents"]
)

plt.title("Top 10 States by Accident Count")
plt.xlabel("State")
plt.ylabel("Total Accidents")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_accident_states.png")

plt.show()

# ------------------------------------
# STEP 6: SEVERITY 4 GRAPH
# ------------------------------------

top_severity = state_summary.sort_values(
    by="Severity_4_Percentage",
    ascending=False
).head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_severity.index,
    top_severity["Severity_4_Percentage"]
)

plt.title("Top 10 States by Severity 4 Percentage")
plt.xlabel("State")
plt.ylabel("Severity 4 Percentage (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("top_10_severity4_states.png")

plt.show()

# ------------------------------------
# STEP 7: FINAL OUTPUT
# ------------------------------------

print("\nTOP 10 STATES BY ACCIDENT COUNT")
print(top_10[["Total_Accidents"]])

print("\nTOP 10 STATES BY SEVERITY 4 PERCENTAGE")
print(
    top_severity[
        ["Total_Accidents", "Severity_4_Percentage"]
    ]
)

print("\nPROGRAM 32 COMPLETED SUCCESSFULLY!")
