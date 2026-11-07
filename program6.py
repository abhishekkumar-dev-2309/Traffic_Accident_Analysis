
# PROGRAM 6: TIME-BASED ACCIDENT ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Step 1: Define paths

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "US_Accidents_March23.csv"
)

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Step 2: Load dataset

data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

# Step 3: Convert Start_Time into datetime

data["Start_Time"] = pd.to_datetime(
    data["Start_Time"],
    utc=True,
    errors="coerce"
)

# Remove invalid timestamps

data = data.dropna(subset=["Start_Time"]).copy()

# Step 4: Extract time features

data["Hour"] = data["Start_Time"].dt.hour

data["Day"] = data["Start_Time"].dt.day_name()

data["Month"] = data["Start_Time"].dt.month_name()

# -----------------------------------
# GRAPH 1: HOURLY ACCIDENT ANALYSIS
# -----------------------------------

hourly_count = data["Hour"].value_counts().sort_index()

plt.figure(figsize=(10, 5))

sns.barplot(
    x=hourly_count.index,
    y=hourly_count.values,
    color="steelblue"
)

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Accidents")

plt.xticks(range(24))

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "hourly_analysis.png")

plt.show()

# -----------------------------------
# GRAPH 2: DAILY ACCIDENT ANALYSIS
# -----------------------------------

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

daily_count = (
    data["Day"]
    .value_counts()
    .reindex(day_order, fill_value=0)
)

plt.figure(figsize=(10, 5))

sns.barplot(
    x=daily_count.index,
    y=daily_count.values,
    color="orange"
)

plt.title("Accidents by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "day_analysis.png")

plt.show()

# -----------------------------------
# GRAPH 3: MONTHLY ACCIDENT ANALYSIS
# -----------------------------------

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

monthly_count = (
    data["Month"]
    .value_counts()
    .reindex(month_order, fill_value=0)
)

plt.figure(figsize=(11, 5))

sns.barplot(
    x=monthly_count.index,
    y=monthly_count.values,
    color="seagreen"
)

plt.title("Accidents by Month")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "monthly_analysis.png")

plt.show()

# -----------------------------------
# DISPLAY ANALYSIS
# -----------------------------------

print("\n========== HOURLY ANALYSIS ==========")
print(hourly_count)

print("\n========== DAILY ANALYSIS ==========")
print(daily_count)

print("\n========== MONTHLY ANALYSIS ==========")
print(monthly_count)

print("\nAll time-based graphs saved successfully!")

print("\nProgram 6 completed successfully!")
