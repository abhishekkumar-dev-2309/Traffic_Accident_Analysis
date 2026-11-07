
# PROGRAM 5: ACCIDENT DATA VISUALIZATION

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

sns.set_theme(style="whitegrid")

# -----------------------------------
# GRAPH 1: Severity Distribution
# -----------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=data,
    x="Severity",
    hue="Severity",
    palette="viridis",
    legend=False
)

plt.title("Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Accidents")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "severity_analysis.png")
plt.show()

# -----------------------------------
# GRAPH 2: Top 10 Cities
# -----------------------------------

plt.figure(figsize=(10, 6))

top_cities = data["City"].value_counts().head(10)

sns.barplot(
    x=top_cities.values,
    y=top_cities.index,
    hue=top_cities.index,
    palette="magma",
    legend=False
)

plt.title("Top 10 Cities by Accident Count")
plt.xlabel("Number of Accidents")
plt.ylabel("City")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "city_analysis.png")
plt.show()

# -----------------------------------
# GRAPH 3: Weather Conditions
# -----------------------------------

plt.figure(figsize=(10, 6))

top_weather = (
    data["Weather_Condition"]
    .value_counts()
    .head(10)
)

sns.barplot(
    x=top_weather.values,
    y=top_weather.index,
    hue=top_weather.index,
    palette="coolwarm",
    legend=False
)

plt.title("Top 10 Weather Conditions")
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "weather_analysis.png")
plt.show()

# -----------------------------------
# GRAPH 4: Missing Values
# -----------------------------------

plt.figure(figsize=(12, 6))

missing_values = (
    data.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_values = missing_values[missing_values > 0]

sns.barplot(
    x=missing_values.values,
    y=missing_values.index,
    hue=missing_values.index,
    palette="rocket",
    legend=False
)

plt.title("Missing Values in Accident Dataset")
plt.xlabel("Number of Missing Values")
plt.ylabel("Columns")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "missing_values.png")
plt.show()

print("\nAll four visualizations created successfully!")
print("Graphs saved in output folder.")

print("\nProgram 5 completed successfully!")
