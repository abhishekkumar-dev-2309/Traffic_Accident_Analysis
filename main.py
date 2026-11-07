
# ==========================================
# PROGRAM 6: TIME-BASED ACCIDENT ANALYSIS
# ==========================================

# Import Pandas for data analysis
import pandas as pd

# Import Matplotlib for graphs
import matplotlib.pyplot as plt

# Import Seaborn for attractive graphs
import seaborn as sns

# Import Path to manage file locations
from pathlib import Path

# Find project folder automatically
project_folder = Path(__file__).resolve().parent

# Dataset file location
dataset_path = project_folder / "dataset" / "US_Accidents_March23.csv"

# Output folder location
output_folder = project_folder / "output"
output_folder.mkdir(exist_ok=True)

# Load first 1000 accident records
df = pd.read_csv(dataset_path, nrows=1000)

# Convert Start_Time column into datetime format
# errors='coerce' converts invalid dates into NaT
df['Start_Time'] = pd.to_datetime(
    df['Start_Time'], errors='coerce', format='mixed'
)

# Remove rows having invalid date/time
df = df.dropna(subset=['Start_Time'])

# Extract hour from accident starting time
df['Hour'] = df['Start_Time'].dt.hour

# Extract weekday name
df['Day'] = df['Start_Time'].dt.day_name()

# Extract month name
df['Month'] = df['Start_Time'].dt.month_name()

# Set graph style
sns.set_theme(style="whitegrid")

# ------------------------------------------
# 1. HOURLY ACCIDENT ANALYSIS
# ------------------------------------------

hour_counts = df['Hour'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
sns.barplot(x=hour_counts.index, y=hour_counts.values)

plt.title("Accidents by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of Accidents")

plt.savefig(output_folder / "hourly_analysis.png")
plt.show()

# ------------------------------------------
# 2. DAY-WISE ACCIDENT ANALYSIS
# ------------------------------------------

day_order = [
    'Monday', 'Tuesday', 'Wednesday',
    'Thursday', 'Friday', 'Saturday', 'Sunday'
]

day_counts = df['Day'].value_counts().reindex(day_order)

plt.figure(figsize=(10, 5))
sns.barplot(x=day_counts.index, y=day_counts.values)

plt.title("Accidents by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30)

plt.savefig(output_folder / "day_analysis.png")
plt.show()

# ------------------------------------------
# 3. MONTHLY ACCIDENT ANALYSIS
# ------------------------------------------

month_order = [
    'January', 'February', 'March', 'April',
    'May', 'June', 'July', 'August',
    'September', 'October', 'November', 'December'
]

month_counts = df['Month'].value_counts().reindex(month_order)

plt.figure(figsize=(12, 5))
sns.barplot(x=month_counts.index, y=month_counts.values)

plt.title("Accidents by Month")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)

plt.savefig(output_folder / "monthly_analysis.png")
plt.show()

print("Time-based analysis completed successfully!")
print("Three graphs saved in the output folder.")



