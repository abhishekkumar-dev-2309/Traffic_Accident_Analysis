
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# STEP 1: Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

train_file = OUTPUT_DIR / "training_dataset.csv"
model_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

# --------------------------------------------------
# STEP 2: Load training dataset
# --------------------------------------------------

print("Loading training dataset...")

train_data = pd.read_csv(train_file)

X_train = train_data.drop(columns=["Severity"])
y_train = train_data["Severity"]

print("Training data loaded:", X_train.shape)

# --------------------------------------------------
# STEP 3: Define feature categories
# --------------------------------------------------

numerical_features = [
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
    "Amenity"
]

categorical_features = [
    "Weather_Condition",
    "Sunrise_Sunset"
]

# Convert Boolean road features into 0 and 1
boolean_features = [
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity"
]

for column in boolean_features:
    X_train[column] = X_train[column].astype("int8")

# --------------------------------------------------
# STEP 4: Create preprocessing pipelines
# --------------------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

print("Preprocessing pipeline created.")

# --------------------------------------------------
# STEP 5: Create Random Forest model
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# STEP 6: Combine preprocessing and model
# --------------------------------------------------

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", random_forest)
])

# --------------------------------------------------
# STEP 7: Train the model
# --------------------------------------------------

print("\nTraining Random Forest model...")
print("Please wait. This may take some time.")

model.fit(X_train, y_train)

print("\nModel training completed!")

# --------------------------------------------------
# STEP 8: Save trained model
# --------------------------------------------------

joblib.dump(model, model_file)

print("\nModel saved successfully!")
print("Saved at:", model_file)

print("\nClasses learned by the model:")
print(model.named_steps["classifier"].classes_)
