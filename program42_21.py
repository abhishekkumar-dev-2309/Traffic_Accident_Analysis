
import pandas as pd
import joblib

# Step 1: Load trained model
model = joblib.load(
    "output/final_accident_severity_model.pkl"
)

print("\n======================================")
print(" ACCIDENT SEVERITY PREDICTION SYSTEM")
print("======================================")

# Step 2: Create input function
def get_input(label, default):
    value = input(f"Enter {label} [{default}]: ")
    if value.strip() == "":
        return default
    return value


# Step 3: Collect user input
print("\nEnter Accident-Related Details")
print("(Press Enter to use the default value)\n")

accident_data = {
    "Start_Lat": float(get_input("Latitude", 40.7128)),
    "Start_Lng": float(get_input("Longitude", -74.0060)),
    "Temperature(F)": float(get_input("Temperature (F)", 77)),
    "Humidity(%)": float(get_input("Humidity (%)", 60)),
    "Pressure(in)": float(get_input("Pressure (in)", 29.9)),
    "Visibility(mi)": float(get_input("Visibility (mi)", 10)),
    "Wind_Speed(mph)": float(get_input("Wind Speed (mph)", 5)),
    "Precipitation(in)": float(get_input("Precipitation (in)", 0)),
    "Hour": int(get_input("Hour (0-23)", 14)),
    "Day_of_Week": int(get_input("Day of Week (0-6)", 4)),
    "Month": int(get_input("Month (1-12)", 6)),
    "Is_Weekend": int(get_input("Is Weekend? (0/1)", 0)),
    "Traffic_Signal": int(get_input("Traffic Signal? (0/1)", 1)),
    "Junction": int(get_input("Junction? (0/1)", 0)),
    "Crossing": int(get_input("Crossing? (0/1)", 0)),
    "Railway": int(get_input("Railway? (0/1)", 0)),
    "Stop": int(get_input("Stop Sign? (0/1)", 0)),
    "Bump": int(get_input("Bump? (0/1)", 0)),
    "Amenity": int(get_input("Amenity? (0/1)", 0)),
    "Weather_Condition": get_input("Weather Condition", "Clear"),
    "Sunrise_Sunset": get_input("Sunrise/Sunset (Day/Night)", "Day")
}

# Step 4: Convert input into DataFrame
input_df = pd.DataFrame([accident_data])

# Step 5: Predict severity
prediction = model.predict(input_df)

# Step 6: Display prediction
print("\n======================================")
print("          PREDICTION RESULT")
print("======================================")

print("Predicted Accident Severity:", prediction[0])

print("\nPrediction completed successfully!")
print("Program 42.21 completed successfully!")
