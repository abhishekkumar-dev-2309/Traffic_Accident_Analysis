
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# STEP 1: DATASET PATH
# -----------------------------------

file_path = r"C:\Users\abhis\Traffic_Accident_Analysis\dataset\US_Accidents_March23.csv"

# Initialize counters
hour_count = pd.Series(0, index=range(24), dtype="int64")

day_count = pd.Series(
    0,
    index=[
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ],
    dtype="int64"
)

month_count = pd.Series(0, index=range(1, 13), dtype="int64")

weekend_count = {"Weekday": 0, "Weekend": 0}

period_count = {
    "Morning": 0,
    "Afternoon": 0,
    "Evening": 0,
    "Night": 0
}

total_records = 0

print("Starting Time-based Accident Analysis...")

# -----------------------------------
# STEP 2: READ DATA IN CHUNKS
# -----------------------------------

for chunk in pd.read_csv(
    file_path,
    usecols=["Start_Time"],
    chunksize=100000
):

    # Convert Start_Time into datetime
    time = pd.to_datetime(
        chunk["Start_Time"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    # Remove invalid timestamps
    time = time.dropna()

    # Extract hour, day and month
    hour = time.dt.hour
    day = time.dt.dayofweek
    month = time.dt.month

    # Count accidents by hour
    hour_count += hour.value_counts().reindex(
        range(24), fill_value=0
    )

    # Count accidents by weekday
    day_count += day.value_counts().reindex(
        range(7), fill_value=0
    ).set_axis(day_count.index)

    # Count accidents by month
    month_count += month.value_counts().reindex(
        range(1, 13), fill_value=0
    )

    # Weekday vs Weekend
    weekend_count["Weekday"] += (day < 5).sum()
    weekend_count["Weekend"] += (day >= 5).sum()

    # Time period classification
    period_count["Morning"] += ((hour >= 5) & (hour < 12)).sum()
    period_count["Afternoon"] += ((hour >= 12) & (hour < 17)).sum()
    period_count["Evening"] += ((hour >= 17) & (hour < 21)).sum()
    period_count["Night"] += ((hour >= 21) | (hour < 5)).sum()

    total_records += len(time)

    print("Processed valid timestamps:", total_records)

# -----------------------------------
# STEP 3: SAVE ANALYSIS
# -----------------------------------

hour_count.to_csv("hourly_accident_analysis.csv")
day_count.to_csv("daily_accident_analysis.csv")
month_count.to_csv("monthly_accident_analysis.csv")

pd.Series(weekend_count).to_csv(
    "weekday_weekend_analysis.csv"
)

pd.Series(period_count).to_csv(
    "time_period_analysis.csv"
)

# -----------------------------------
# STEP 4: HOURLY GRAPH
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(hour_count.index, hour_count.values)

plt.title("Hourly Accident Distribution")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.xticks(range(24))

plt.tight_layout()
plt.savefig("hourly_accident_analysis.png")
plt.show()

# -----------------------------------
# STEP 5: DAILY GRAPH
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(day_count.index, day_count.values)

plt.title("Day-wise Accident Distribution")
plt.xlabel("Day")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("daily_accident_analysis.png")
plt.show()

# -----------------------------------
# STEP 6: MONTHLY GRAPH
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.bar(month_count.index, month_count.values)

plt.title("Monthly Accident Distribution")
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(range(1, 13))

plt.tight_layout()
plt.savefig("monthly_accident_analysis.png")
plt.show()

# -----------------------------------
# STEP 7: WEEKDAY VS WEEKEND
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    weekend_count.keys(),
    weekend_count.values()
)

plt.title("Weekday vs Weekend Accidents")
plt.ylabel("Number of Accidents")

plt.tight_layout()
plt.savefig("weekday_weekend_analysis.png")
plt.show()

# -----------------------------------
# STEP 8: TIME PERIOD GRAPH
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    period_count.keys(),
    period_count.values()
)

plt.title("Accidents by Time Period")
plt.ylabel("Number of Accidents")

plt.tight_layout()
plt.savefig("time_period_analysis.png")
plt.show()

# -----------------------------------
# STEP 9: FINAL RESULTS
# -----------------------------------

print("\nTOTAL VALID RECORDS:", total_records)

print("\nHIGHEST ACCIDENT HOUR:")
print(hour_count.idxmax(), "Hour")

print("\nHIGHEST ACCIDENT DAY:")
print(day_count.idxmax())

print("\nHIGHEST ACCIDENT MONTH:")
print(month_count.idxmax())

print("\nWEEKDAY VS WEEKEND:")
print(weekend_count)

print("\nTIME PERIOD ANALYSIS:")
print(period_count)

print("\nPROGRAM 33 COMPLETED SUCCESSFULLY!")
