import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import spaces


# ============================================================
# ZERO GPU STARTUP REQUIREMENT
# ============================================================

@spaces.GPU
def _zerogpu_startup_probe():
    return None


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

model = tf.keras.models.load_model("improved_model.keras")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")


# ============================================================
# FEATURE CONFIGURATION
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
# RISK CATEGORY
# ============================================================

def get_risk_category(risk_score):

    if risk_score < 0.25:
        return "Low Risk"

    elif risk_score < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# RISK CATEGORY HTML
# ============================================================

def get_category_html(category):

    if category == "Low Risk":

        return """
        <div class="risk-card low-risk">
            <div class="risk-label">Risk Category</div>
            <div class="risk-value">Low Risk</div>
            <div class="risk-description">
                The estimated risk level is relatively low.
            </div>
        </div>
        """

    elif category == "Medium Risk":

        return """
        <div class="risk-card medium-risk">
            <div class="risk-label">Risk Category</div>
            <div class="risk-value">Medium Risk</div>
            <div class="risk-description">
                The estimated risk level is moderate.
            </div>
        </div>
        """

    else:

        return """
        <div class="risk-card high-risk">
            <div class="risk-label">Risk Category</div>
            <div class="risk-value">High Risk</div>
            <div class="risk-description">
                The estimated risk level is relatively high.
            </div>
        </div>
        """


# ============================================================
# PREDICTION FUNCTION
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

    try:

        # ----------------------------------------------------
        # Convert date
        # ----------------------------------------------------

        date_obj = pd.to_datetime(
            date,
            errors="raise"
        )

        # ----------------------------------------------------
        # Convert time
        # ----------------------------------------------------

        hour = int(time.split(":")[0])
        minute = int(time.split(":")[1])

        # ----------------------------------------------------
        # Validate time
        # ----------------------------------------------------

        if hour < 0 or hour > 23:
            raise ValueError("Hour must be between 00 and 23.")

        if minute < 0 or minute > 59:
            raise ValueError("Minute must be between 00 and 59.")

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "city": city,
            "road_type": road_type,
            "weather": weather,
            "visibility": visibility,
            "traffic_density": traffic_density,
            "day_of_week": day_of_week,

            "hour": hour,
            "is_weekend": int(is_weekend),
            "is_peak_hour": int(is_peak_hour),

            "year": date_obj.year,
            "month": date_obj.month,
            "date_day": date_obj.day,

            "time_hour": hour,
            "time_minute": minute
        }])

        # ----------------------------------------------------
        # One-hot encoding
        # ----------------------------------------------------

        categorical_data = encoder.transform(
            input_data[categorical_features]
        )

        # ----------------------------------------------------
        # Numerical scaling
        # ----------------------------------------------------

        numerical_data = scaler.transform(
            input_data[numerical_features]
        )

        # ----------------------------------------------------
        # Combine processed features
        # ----------------------------------------------------

        processed_input = np.hstack([
            categorical_data,
            numerical_data
        ])

        # ----------------------------------------------------
        # Deep Learning prediction
        # ----------------------------------------------------

        risk_score = float(
            model.predict(
                processed_input,
                verbose=0
            )[0][0]
        )

        # ----------------------------------------------------
        # Keep score within expected range
        # ----------------------------------------------------

        risk_score = np.clip(
            risk_score,
            0.10,
            1.00
        )

        # ----------------------------------------------------
        # Risk category
        # ----------------------------------------------------

        risk_category = get_risk_category(
            risk_score
        )

        # ----------------------------------------------------
        # Score HTML
        # ----------------------------------------------------

        score_html = f"""
        <div class="score-card">
            <div class="score-label">
                Predicted Risk Score
            </div>

            <div class="score-value">
                {risk_score:.4f}
            </div>

            <div class="score-scale">
                Scale: 0.10 — 1.00
            </div>
        </div>
        """

        # ----------------------------------------------------
        # Category HTML
        # ----------------------------------------------------

        category_html = get_category_html(
            risk_category
        )

        return score_html, category_html

    except Exception as e:

        error_html = f"""
        <div class="error-card">
            <div class="error-title">
                Unable to generate prediction
            </div>

            <div class="error-message">
                Please check the date and time format.
            </div>
        </div>
        """

        return error_html, ""


# ============================================================
# CUSTOM CSS
# ============================================================

css = """

/* ---------------------------------------------------------
   MAIN PAGE
--------------------------------------------------------- */

body {
    background: #f6efe3 !important;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: #f6efe3 !important;
    font-family: Arial, Helvetica, sans-serif !important;
}


/* ---------------------------------------------------------
   HEADER
--------------------------------------------------------- */

.main-header {
    text-align: center;
    padding: 28px 20px 12px 20px;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #173f6f;
    margin-bottom: 8px;
}

.main-subtitle {
    font-size: 16px;
    color: #5f7185;
    margin-bottom: 10px;
}


/* ---------------------------------------------------------
   INPUT CARD
--------------------------------------------------------- */

.input-card {
    background: #fffaf2;
    border: 1px solid #e4d7c5;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(60, 45, 25, 0.08);
}


/* ---------------------------------------------------------
   LABELS
--------------------------------------------------------- */

label,
span {
    color: #24496d !important;
}

label {
    font-weight: 600 !important;
}


/* ---------------------------------------------------------
   INPUT BOXES
--------------------------------------------------------- */

.gradio-container input,
.gradio-container textarea,
.gradio-container button,
.gradio-container select {
    border-radius: 10px !important;
}

.gradio-container input,
.gradio-container textarea {
    background: #fffdf8 !important;
    border: 1px solid #d8cbb9 !important;
}


/* ---------------------------------------------------------
   DROPDOWN
--------------------------------------------------------- */

.gradio-container .wrap {
    border-radius: 10px !important;
}


/* ---------------------------------------------------------
   PREDICT BUTTON
--------------------------------------------------------- */

.predict-btn {
    background: linear-gradient(
        135deg,
        #2469ad 0%,
        #174b80 55%,
        #123c68 100%
    ) !important;

    color: white !important;

    border: 2px solid #8db8df !important;

    border-radius: 12px !important;

    font-size: 18px !important;

    font-weight: 700 !important;

    padding: 14px 28px !important;

    box-shadow:
        0 4px 0 #0d3155,
        0 8px 18px rgba(20, 65, 105, 0.25) !important;

    transition: all 0.2s ease !important;
}

.predict-btn:hover {
    background: linear-gradient(
        135deg,
        #2f7bc5 0%,
        #1c5b97 55%,
        #154775 100%
    ) !important;

    transform: translateY(-1px) !important;

    box-shadow:
        0 5px 0 #0d3155,
        0 10px 22px rgba(20, 65, 105, 0.30) !important;
}

.predict-btn:active {
    transform: translateY(3px) !important;

    box-shadow:
        0 2px 0 #0d3155,
        0 4px 10px rgba(20, 65, 105, 0.20) !important;
}


/* ---------------------------------------------------------
   RESULT SECTION
--------------------------------------------------------- */

.result-heading {
    color: #173f6f;
    font-size: 25px;
    font-weight: 800;
    margin: 25px 0 15px 0;
}


/* ---------------------------------------------------------
   SCORE CARD
--------------------------------------------------------- */

.score-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #fff8ec
    );

    border: 1px solid #e0d1bd;

    border-radius: 16px;

    padding: 24px;

    text-align: center;

    box-shadow: 0 8px 20px rgba(60, 45, 25, 0.08);
}

.score-label {
    font-size: 17px;
    font-weight: 600;
    color: #5f7185;
}

.score-value {
    font-size: 38px;
    font-weight: 800;
    color: #1b527f;
    margin-top: 8px;
}

.score-scale {
    font-size: 13px;
    color: #8793a0;
    margin-top: 6px;
}


/* ---------------------------------------------------------
   RISK CARDS
--------------------------------------------------------- */

.risk-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-top: 10px;
}

.low-risk {
    background: #edf7ef;
    border: 1px solid #b8dcbc;
}

.medium-risk {
    background: #fff6df;
    border: 1px solid #e7ce91;
}

.high-risk {
    background: #fff0ed;
    border: 1px solid #e5b8ad;
}

.risk-label {
    font-size: 15px;
    color: #5f7185;
    font-weight: 600;
}

.risk-value {
    font-size: 30px;
    font-weight: 800;
    margin-top: 6px;
    color: #24496d;
}

.risk-description {
    font-size: 14px;
    color: #68798a;
    margin-top: 8px;
}


/* ---------------------------------------------------------
   ERROR CARD
--------------------------------------------------------- */

.error-card {
    background: #fff0ed;
    border: 1px solid #e5b8ad;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}

.error-title {
    color: #8a3f32;
    font-size: 18px;
    font-weight: 700;
}

.error-message {
    color: #735b56;
    margin-top: 6px;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer-note {
    text-align: center;
    color: #7a8793;
    font-size: 13px;
    padding: 20px;
}

"""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(
    css=css,
    title="DL-Based Road Accident Risk Prediction"
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="main-header">

            <div class="main-title">
                🚗 DL-Based Road Accident Risk Prediction
            </div>

            <div class="main-subtitle">
                Estimate accident risk using road, traffic,
                weather, visibility and time-related conditions.
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    with gr.Column(elem_classes="input-card"):

        gr.Markdown(
            "### Enter Road & Environmental Conditions"
        )

        with gr.Row():

            with gr.Column():

                city = gr.Dropdown(
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
                )

                road_type = gr.Dropdown(
                    choices=[
                        "highway",
                        "rural",
                        "urban"
                    ],
                    label="Road Type",
                    value="urban"
                )

                weather = gr.Dropdown(
                    choices=[
                        "clear",
                        "fog",
                        "rain"
                    ],
                    label="Weather",
                    value="clear"
                )

                visibility = gr.Dropdown(
                    choices=[
                        "high",
                        "low",
                        "medium"
                    ],
                    label="Visibility",
                    value="high"
                )

                traffic_density = gr.Dropdown(
                    choices=[
                        "high",
                        "low",
                        "medium"
                    ],
                    label="Traffic Density",
                    value="low"
                )

            with gr.Column():

                date = gr.Textbox(
                    label="Date",
                    value="2026-09-11",
                    placeholder="YYYY-MM-DD"
                )

                time = gr.Textbox(
                    label="Time",
                    value="10:00",
                    placeholder="HH:MM"
                )

                day_of_week = gr.Dropdown(
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
                )

                is_weekend = gr.Radio(
                    choices=[0, 1],
                    label="Is Weekend? (0 = No, 1 = Yes)",
                    value=0
                )

                is_peak_hour = gr.Radio(
                    choices=[0, 1],
                    label="Is Peak Hour? (0 = No, 1 = Yes)",
                    value=0
                )


        # ----------------------------------------------------
        # BUTTON
        # ----------------------------------------------------

        predict_button = gr.Button(
            "Predict Accident Risk",
            variant="primary",
            elem_classes="predict-btn"
        )


    # --------------------------------------------------------
    # RESULT SECTION
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="result-heading">
            Prediction Result
        </div>
        """
    )


    with gr.Row():

        score_output = gr.HTML(
            label="Predicted Risk Score"
        )

        category_output = gr.HTML(
            label="Risk Category"
        )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="footer-note">
            This system provides an estimated risk score based
            on the conditions entered by the user.
        </div>
        """
    )


    # --------------------------------------------------------
    # BUTTON ACTION
    # --------------------------------------------------------

    predict_button.click(
        fn=predict_risk,
        inputs=[
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
        ],
        outputs=[
            score_output,
            category_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False
    )
