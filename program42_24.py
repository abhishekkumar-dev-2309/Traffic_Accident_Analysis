
import os
import pandas as pd
import joblib

print("\n==========================================")
print(" FINAL PROJECT VERIFICATION SYSTEM")
print("==========================================")

# Step 1: Define important project files
files = {
    "Original Dataset": "dataset/US_Accidents_March23.csv",
    "Full Accident Locations": "output/full_accident_locations.csv",
    "Clustered Accident Data": "output/full_clustered_accidents.csv",
    "Hotspot Analysis": "output/final_priority_analysis.csv",
    "Model Comparison": "output/three_model_comparison.csv",
    "Final Model": "output/final_accident_severity_model.pkl",
    "Temporal Model": "output/temporal_random_forest_model.pkl",
    "Final Dashboard": "output/final_project_dashboard.png",
    "Project Summary": "output/final_project_summary.csv",
    "Prediction History": "output/user_prediction_history.csv"
}

# Step 2: Check file availability
print("\n========== FILE VERIFICATION ==========")

for name, path in files.items():

    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)

        print(f"[OK] {name} - Available ({size:.2f} MB)")

    else:
        print(f"[MISSING] {name} - Not Found")


# Step 3: Check important CSV files
print("\n========== CSV VERIFICATION ==========")

csv_files = [
    "output/full_accident_locations.csv",
    "output/full_clustered_accidents.csv",
    "output/final_priority_analysis.csv",
    "output/three_model_comparison.csv",
    "output/final_project_summary.csv"
]

for path in csv_files:

    if os.path.exists(path):

        try:
            sample = pd.read_csv(path, nrows=5)

            print("\nFile:", path)
            print("Columns:", len(sample.columns))
            print("Sample records:", len(sample))

        except Exception as e:
            print("CSV Error:", path, e)


# Step 4: Load final trained model
print("\n========== MODEL VERIFICATION ==========")

model_path = "output/final_accident_severity_model.pkl"

if os.path.exists(model_path):

    model = joblib.load(model_path)

    print("Final Random Forest model loaded successfully!")

else:
    raise FileNotFoundError("Final model file is missing!")


# Step 5: Test model with existing test data
print("\n========== PREDICTION TEST ==========")

test_path = "output/testing_dataset.csv"

if os.path.exists(test_path):

    test_data = pd.read_csv(test_path, nrows=5)

    features = [
        "Start_Lat", "Start_Lng",
        "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)",
        "Wind_Speed(mph)", "Precipitation(in)",
        "Hour", "Day_of_Week", "Month",
        "Is_Weekend", "Traffic_Signal",
        "Junction", "Crossing", "Railway",
        "Stop", "Bump", "Amenity",
        "Weather_Condition", "Sunrise_Sunset"
    ]

    X = test_data[features]

    predictions = model.predict(X)

    print("Sample predictions:", predictions)
    print("Model prediction test successful!")

else:
    print("Testing dataset not found!")


# Step 6: Final verification message
print("\n==========================================")
print(" FINAL VERIFICATION COMPLETED")
print("==========================================")
