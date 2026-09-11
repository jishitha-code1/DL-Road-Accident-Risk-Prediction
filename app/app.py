
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# ==============================
# Load Model and Preprocessing
# ==============================

model = tf.keras.models.load_model(
    "models/improved_model.keras"
)

encoder = joblib.load(
    "models/encoder.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# ==============================
# Feature Configuration
# ==============================

categorical_features = [
    "city",
    "road_type",
    "weather",
    "visibility",
    "traffic_density",
    "day_of_week"
]

numerical_features = [
    "hour",
    "is_weekend",
    "is_peak_hour",
    "year",
    "month",
    "date_day",
    "time_hour",
    "time_minute"
]


# ==============================
# Risk Category
# ==============================

def get_risk_category(risk_score):

    if risk_score < 0.25:
        return "Low Risk"

    elif risk_score < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ==============================
# Prediction Function
# ==============================

def predict_risk(
    city,
    road_type,
    weather,
    visibility,
    traffic_density,
    date,
    time,
    day_of_week,
    is_weekend,
    is_peak_hour
):

    date_obj = pd.to_datetime(date)

    hour = int(time.split(":")[0])
    minute = int(time.split(":")[1])

    input_data = pd.DataFrame([{

        "city": city,
        "road_type": road_type,
        "weather": weather,
        "visibility": visibility,
        "traffic_density": traffic_density,
        "day_of_week": day_of_week,

        "hour": hour,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,

        "year": date_obj.year,
        "month": date_obj.month,
        "date_day": date_obj.day,

        "time_hour": hour,
        "time_minute": minute
    }])


    # Encode categorical features
    categorical_data = encoder.transform(
        input_data[categorical_features]
    )


    # Scale numerical features
    numerical_data = scaler.transform(
        input_data[numerical_features]
    )


    # Combine features
    processed_input = np.hstack([
        categorical_data,
        numerical_data
    ])


    # Predict
    risk_score = float(
        model.predict(
            processed_input,
            verbose=0
        )[0][0]
    )


    risk_category = get_risk_category(
        risk_score
    )


    return (
        f"Predicted Risk Score: {risk_score:.4f}",
        f"Risk Category: {risk_category}"
    )


# ==============================
# Gradio Interface
# ==============================

demo = gr.Interface(

    fn=predict_risk,

    inputs=[

        gr.Dropdown(
            choices=[
                "Bangalore",
                "Chandigarh",
                "Chennai",
                "Delhi",
                "Hyderabad",
                "Kolkata",
                "Mumbai",
                "Pune"
            ],
            label="City",
            value="Delhi",
            allow_custom_value=True
        ),

        gr.Dropdown(
            choices=[
                "highway",
                "rural",
                "urban"
            ],
            label="Road Type",
            value="urban"
        ),

        gr.Dropdown(
            choices=[
                "clear",
                "fog",
                "rain"
            ],
            label="Weather",
            value="clear"
        ),

        gr.Dropdown(
            choices=[
                "high",
                "low",
                "medium"
            ],
            label="Visibility",
            value="high"
        ),

        gr.Dropdown(
            choices=[
                "high",
                "low",
                "medium"
            ],
            label="Traffic Density",
            value="low"
        ),

        gr.Textbox(
            label="Date",
            value="2026-09-11",
            placeholder="YYYY-MM-DD"
        ),

        gr.Textbox(
            label="Time",
            value="10:00",
            placeholder="HH:MM"
        ),

        gr.Dropdown(
            choices=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ],
            label="Day of Week",
            value="Friday"
        ),

        gr.Radio(
            choices=[0, 1],
            label="Is Weekend? (0 = No, 1 = Yes)",
            value=0
        ),

        gr.Radio(
            choices=[0, 1],
            label="Is Peak Hour? (0 = No, 1 = Yes)",
            value=0
        )
    ],

    outputs=[

        gr.Textbox(
            label="Predicted Risk Score"
        ),

        gr.Textbox(
            label="Risk Category"
        )
    ],

    title="🚗 DL-Based Road Accident Risk Prediction",

    description=(
        "Enter road and environmental conditions "
        "to estimate accident risk."
    )
)


# ==============================
# Launch
# ==============================

if __name__ == "__main__":
    demo.launch()