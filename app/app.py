import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import spaces


# ============================================================
# Load Model and Preprocessing
# ============================================================

model = tf.keras.models.load_model(
    "improved_model.keras"
)

encoder = joblib.load(
    "encoder.pkl"
)

scaler = joblib.load(
    "scaler.pkl"
)


# ============================================================
# Feature Configuration
# ============================================================

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


# ============================================================
# Processed Feature Names
# ============================================================

feature_names = (
    list(
        encoder.get_feature_names_out(
            categorical_features
        )
    )
    + numerical_features
)


# ============================================================
# SHAP Explainability
# ============================================================

background = np.zeros(
    (1, len(feature_names)),
    dtype=np.float32
)

explainer = shap.DeepExplainer(
    model,
    background
)


# ============================================================
# ZeroGPU Startup Probe
# ============================================================

@spaces.GPU
def _zerogpu_startup_probe():
    return None


# ============================================================
# Risk Category
# ============================================================

def get_risk_category(risk_score):

    if risk_score < 0.25:
        return "Low Risk"

    elif risk_score < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# SHAP Explanation
# ============================================================

def create_shap_explanation(processed_input):

    # Calculate SHAP values
    shap_values = explainer.shap_values(
        processed_input
    )

    shap_array = np.squeeze(
        np.array(shap_values)
    )

    # Create feature contribution table
    contributions = pd.DataFrame({

        "Feature": feature_names,

        "SHAP": shap_array.flatten()
    })

    # Absolute SHAP value tells feature importance
    contributions["Abs_SHAP"] = (
        contributions["SHAP"].abs()
    )

    # Select top 10 most influential features
    contributions = (
        contributions
        .sort_values(
            "Abs_SHAP",
            ascending=False
        )
        .head(10)
    )

    # Sort for horizontal bar chart
    contributions = contributions.sort_values(
        "SHAP"
    )


    # ========================================================
    # Create Chart
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        contributions["Feature"],
        contributions["SHAP"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "SHAP Contribution"
    )

    ax.set_ylabel(
        "Features"
    )

    ax.set_title(
        "SHAP Explanation - Feature Contributions"
    )

    plt.tight_layout()


    return fig


# ============================================================
# Prediction Function
# ============================================================

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

    # ========================================================
    # STEP 1: Read User Input
    # ========================================================

    date_obj = pd.to_datetime(
        date
    )

    hour = int(
        time.split(":")[0]
    )

    minute = int(
        time.split(":")[1]
    )


    # ========================================================
    # STEP 2: Create Input DataFrame
    # ========================================================

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


    # ========================================================
    # STEP 3: One-Hot Encode Categorical Features
    # ========================================================

    categorical_data = encoder.transform(
        input_data[categorical_features]
    )


    # ========================================================
    # STEP 4: Scale Numerical Features
    # ========================================================

    numerical_data = scaler.transform(
        input_data[numerical_features]
    )


    # ========================================================
    # STEP 5: Combine Processed Features
    # ========================================================

    processed_input = np.hstack([

        categorical_data,

        numerical_data

    ])


    # Convert to TensorFlow compatible format
    processed_input = processed_input.astype(
        np.float32
    )


    # ========================================================
    # STEP 6: Deep Learning Model Prediction
    # ========================================================

    risk_score = float(

        model.predict(
            processed_input,
            verbose=0
        )[0][0]

    )


    # ========================================================
    # STEP 7: Convert Score to Risk Category
    # ========================================================

    risk_category = get_risk_category(
        risk_score
    )


    # ========================================================
    # STEP 8: SHAP Explainability
    # ========================================================

    shap_plot = create_shap_explanation(
        processed_input
    )


    # ========================================================
    # STEP 9: Generate Current Prediction Explanation
    # ========================================================

    shap_values = explainer.shap_values(
        processed_input
    )

    shap_array = np.squeeze(
        np.array(shap_values)
    )

    contributions = pd.DataFrame({

        "Feature": feature_names,

        "SHAP": shap_array.flatten()

    })


    # Factors that increased risk
    higher_risk = (
        contributions[
            contributions["SHAP"] > 0
        ]
        .sort_values(
            "SHAP",
            ascending=False
        )
        .head(3)
    )


    # Factors that decreased risk
    lower_risk = (
        contributions[
            contributions["SHAP"] < 0
        ]
        .sort_values(
            "SHAP",
            ascending=True
        )
        .head(3)
    )


    # Convert feature names into readable names
    def readable_feature(feature):

        feature = feature.replace(
            "_",
            " "
        )

        return feature.title()


    high_factors = ", ".join(
        readable_feature(feature)
        for feature in higher_risk["Feature"]
    )


    low_factors = ", ".join(
        readable_feature(feature)
        for feature in lower_risk["Feature"]
    )


    explanation = (
        f"Factors increasing risk: "
        f"{high_factors if high_factors else 'None'}\n\n"
        f"Factors decreasing risk: "
        f"{low_factors if low_factors else 'None'}"
    )


    # ========================================================
    # STEP 10: Return Results
    # ========================================================

    return (

        f"Predicted Risk Score: "
        f"{risk_score:.4f}",

        f"Risk Category: "
        f"{risk_category}",

        shap_plot,

        explanation
    )


# ============================================================
# Gradio Interface
# ============================================================

demo = gr.Interface(

    fn=predict_risk,


    # ========================================================
    # USER INPUTS
    # ========================================================

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


    # ========================================================
    # OUTPUTS
    # ========================================================

    outputs=[

        gr.Textbox(
            label="Predicted Risk Score"
        ),


        gr.Textbox(
            label="Risk Category"
        ),


        gr.Plot(
            label="SHAP Explanation"
        ),


        gr.Textbox(
            label="How to Read the Explanation"
        )

    ],


    # ========================================================
    # TITLE AND DESCRIPTION
    # ========================================================

    title="🚗 DL-Based Road Accident Risk Prediction",


    description=(
        "Enter road and environmental conditions "
        "to estimate accident risk and understand "
        "the model's prediction using SHAP."
    )

)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )