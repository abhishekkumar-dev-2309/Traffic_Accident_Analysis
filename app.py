
import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Traffic Accident Prediction",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(
        "output/final_accident_severity_model.pkl"
    )

model = load_model()

st.title("🚗 Traffic Accident Severity Prediction")
st.write("Enter accident-related details to predict severity.")

with st.form("prediction_form"):

    st.subheader("Location and Weather")

    col1, col2 = st.columns(2)

    with col1:
        lat = st.number_input("Latitude", value=40.7128)
        lng = st.number_input("Longitude", value=-74.006)
        temp = st.number_input("Temperature (F)", value=77.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
        pressure = st.number_input("Pressure (in)", value=29.9)
        visibility = st.number_input("Visibility (mi)", value=10.0)
        wind = st.number_input("Wind Speed (mph)", value=5.0)
        precipitation = st.number_input("Precipitation (in)", value=0.0)

    with col2:
        hour = st.number_input("Hour", 0, 23, 14)
        day = st.number_input("Day of Week", 0, 6, 4)
        month = st.number_input("Month", 1, 12, 6)

        weekend = st.selectbox("Weekend?", [0, 1])
        traffic = st.selectbox("Traffic Signal?", [0, 1])
        junction = st.selectbox("Junction?", [0, 1])
        crossing = st.selectbox("Crossing?", [0, 1])
        railway = st.selectbox("Railway?", [0, 1])
        stop = st.selectbox("Stop Sign?", [0, 1])
        bump = st.selectbox("Bump?", [0, 1])
        amenity = st.selectbox("Amenity?", [0, 1])

    weather = st.text_input("Weather Condition", "Clear")
    sunrise = st.selectbox("Sunrise/Sunset", ["Day", "Night"])

    submit = st.form_submit_button("Predict Accident Severity")

if submit:

    accident_data = {
        "Start_Lat": lat,
        "Start_Lng": lng,
        "Temperature(F)": temp,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure,
        "Visibility(mi)": visibility,
        "Wind_Speed(mph)": wind,
        "Precipitation(in)": precipitation,
        "Hour": hour,
        "Day_of_Week": day,
        "Month": month,
        "Is_Weekend": weekend,
        "Traffic_Signal": traffic,
        "Junction": junction,
        "Crossing": crossing,
        "Railway": railway,
        "Stop": stop,
        "Bump": bump,
        "Amenity": amenity,
        "Weather_Condition": weather,
        "Sunrise_Sunset": sunrise
    }

    input_df = pd.DataFrame([accident_data])

    try:
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Severity Class: {prediction}")

        result = accident_data.copy()
        result["Predicted_Severity"] = prediction
        result["Prediction_Time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        os.makedirs("output", exist_ok=True)
        file = "output/user_prediction_history.csv"

        pd.DataFrame([result]).to_csv(
            file,
            mode="a",
            header=not os.path.exists(file),
            index=False
        )

        st.info("Prediction history saved successfully.")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# =====================================
# PREDICTION HISTORY
# =====================================

st.divider()

st.header("📊 Prediction History")

history_file = "output/user_prediction_history.csv"

if os.path.exists(history_file):

    history_df = pd.read_csv(history_file)

    st.metric(
        "Total Predictions",
        len(history_df)
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction History",
        data=csv_data,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info("No prediction history available yet.")
