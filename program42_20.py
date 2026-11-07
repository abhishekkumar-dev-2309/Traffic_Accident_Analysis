
import pandas as pd
import joblib

# Step 1: Load the final trained model
model = joblib.load(
    "output/final_accident_severity_model.pkl"
)

print("Accident Severity Model Loaded Successfully!")

# Step 2: Create prediction function
def predict_accident_severity(input_data):

    # Convert input dictionary into DataFrame
    input_df = pd.DataFrame([input_data])

    # Predict severity
    prediction = model.predict(input_df)

    return prediction[0]


# Step 3: Provide sample input
accident_data = {
    "Start_Lat": 40.7128,
    "Start_Lng": -74.0060,
    "Temperature(F)": 77.0,
    "Humidity(%)": 60.0,
    "Pressure(in)": 29.9,
    "Visibility(mi)": 10.0,
    "Wind_Speed(mph)": 5.0,
    "Precipitation(in)": 0.0,
    "Hour": 14,
    "Day_of_Week": 4,
    "Month": 6,
    "Is_Weekend": 0,
    "Traffic_Signal": 1,
    "Junction": 0,
    "Crossing": 0,
    "Railway": 0,
    "Stop": 0,
    "Bump": 0,
    "Amenity": 0,
    "Weather_Condition": "Clear",
    "Sunrise_Sunset": "Day"
}

# Step 4: Call prediction function
result = predict_accident_severity(accident_data)

# Step 5: Display result
print("\n========== ACCIDENT SEVERITY PREDICTION ==========")

print("Input Location:")
print("Latitude:", accident_data["Start_Lat"])
print("Longitude:", accident_data["Start_Lng"])

print("\nPredicted Severity:", result)

print("\nProgram 42.20 completed successfully!")
