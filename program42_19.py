
import pandas as pd
import joblib
import os

# Step 1: Define file paths
model_path = "output/tuned_random_forest_model.pkl"
test_path = "output/testing_dataset.csv"

# Step 2: Load the tuned model
print("Loading Tuned Random Forest Model...")

model = joblib.load(model_path)

print("Model loaded successfully!")

# Step 3: Save model with final name
final_model_path = "output/final_accident_severity_model.pkl"

joblib.dump(model, final_model_path)

print("Final model saved successfully!")

# Step 4: Load testing dataset
print("\nLoading testing dataset...")

test_data = pd.read_csv(test_path)

print("Testing records:", len(test_data))

# Step 5: Select required input features
features = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Hour",
    "Day_of_Week",
    "Month",
    "Is_Weekend",
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity",
    "Weather_Condition",
    "Sunrise_Sunset"
]

X_test = test_data[features]

# Step 6: Take 20 sample records
X_sample = X_test.head(20)

actual = test_data["Severity"].head(20)

# Step 7: Load final saved model
final_model = joblib.load(final_model_path)

# Step 8: Predict accident severity
predictions = final_model.predict(X_sample)

# Step 9: Create result table
results = pd.DataFrame({
    "Actual_Severity": actual.values,
    "Predicted_Severity": predictions
})

results["Prediction_Status"] = (
    results["Actual_Severity"] == results["Predicted_Severity"]
).map({
    True: "Correct",
    False: "Incorrect"
})

# Step 10: Display predictions
print("\n========== SAMPLE PREDICTIONS ==========")
print(results.to_string(index=False))

# Step 11: Save prediction results
results.to_csv(
    "output/final_model_sample_predictions.csv",
    index=False
)

print("\nPrediction results saved successfully!")
print("Program 42.19 completed successfully!")
