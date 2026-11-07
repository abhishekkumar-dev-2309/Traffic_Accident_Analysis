
import pandas as pd
import joblib
import os
from datetime import datetime

# Step 1: Load trained model
model = joblib.load(
    "output/final_accident_severity_model.pkl"
)

print("\n======================================")
print(" IMPROVED ACCIDENT SEVERITY PREDICTION")
print("======================================")


# Step 2: Numeric input with validation
def get_number(label, default, minimum, maximum, integer=False):

    while True:
        value = input(f"Enter {label} [{default}]: ").strip()

        if value == "":
            return default

        try:
            number = float(value)

            if not minimum <= number <= maximum:
                print(f"Enter a value between {minimum} and {maximum}.")
                continue

            if integer:
                if not number.is_integer():
                    print("Please enter a whole number.")
                    continue
                return int(number)

            return number

        except ValueError:
            print("Invalid input! Please enter a number.")


# Step 3: Binary input validation
def get_binary(label, default):

    while True:
        value = input(f"Enter {label} (0/1) [{default}]: ").strip()

        if value == "":
            return default

        if value in ["0", "1"]:
            return int(value)

        print("Please enter only 0 or 1.")


# Step 4: Text input
def get_text(label, default):

    value = input(f"Enter {label} [{default}]: ").strip()

    if value == "":
        return default

    return value


# Step 5: Collect user inputs
print("\nEnter Accident-Related Details")
print("Press Enter to use default values.\n")

accident_data = {
    "Start_Lat": get_number("Latitude", 40.7128, -90, 90),
    "Start_Lng": get_number("Longitude", -74.006, -180, 180),
    "Temperature(F)": get_number("Temperature (F)", 77, -100, 150),
    "Humidity(%)": get_number("Humidity (%)", 60, 0, 100),
    "Pressure(in)": get_number("Pressure (in)", 29.9, 0, 40),
    "Visibility(mi)": get_number("Visibility (mi)", 10, 0, 100),
    "Wind_Speed(mph)": get_number("Wind Speed (mph)", 5, 0, 300),
    "Precipitation(in)": get_number("Precipitation (in)", 0, 0, 100),
    "Hour": get_number("Hour (0-23)", 14, 0, 23, True),
    "Day_of_Week": get_number("Day of Week (0-6)", 4, 0, 6, True),
    "Month": get_number("Month (1-12)", 6, 1, 12, True),
    "Is_Weekend": get_binary("Is Weekend?", 0),
    "Traffic_Signal": get_binary("Traffic Signal?", 1),
    "Junction": get_binary("Junction?", 0),
    "Crossing": get_binary("Crossing?", 0),
    "Railway": get_binary("Railway?", 0),
    "Stop": get_binary("Stop Sign?", 0),
    "Bump": get_binary("Bump?", 0),
    "Amenity": get_binary("Amenity?", 0),
    "Weather_Condition": get_text("Weather Condition", "Clear"),
    "Sunrise_Sunset": get_text("Sunrise/Sunset (Day/Night)", "Day")
}

# Step 6: Validate Sunrise_Sunset
while accident_data["Sunrise_Sunset"].lower() not in ["day", "night"]:
    print("Please enter Day or Night.")
    accident_data["Sunrise_Sunset"] = get_text(
        "Sunrise/Sunset (Day/Night)", "Day"
    )

accident_data["Sunrise_Sunset"] = accident_data["Sunrise_Sunset"].capitalize()

# Step 7: Convert input into DataFrame
input_df = pd.DataFrame([accident_data])

# Step 8: Predict severity
prediction = model.predict(input_df)[0]

# Step 9: Display result
print("\n======================================")
print("          PREDICTION RESULT")
print("======================================")

print("Predicted Severity Class:", prediction)

# Step 10: Save prediction history
result = accident_data.copy()
result["Predicted_Severity"] = prediction
result["Prediction_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

output_file = "output/user_prediction_history.csv"

result_df = pd.DataFrame([result])

if os.path.exists(output_file):
    result_df.to_csv(
        output_file,
        mode="a",
        header=False,
        index=False
    )
else:
    result_df.to_csv(output_file, index=False)

print("\nPrediction saved successfully!")
print("File:", output_file)

print("\nProgram 42.22 completed successfully!")
