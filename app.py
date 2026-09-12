import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DL-Based Road Accident Risk Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       MAIN BACKGROUND
    ===================================================== */

    .stApp {
        background: #f6efe3;
    }

    .main .block-container {
        max-width: 1100px;
        padding-top: 35px;
        padding-bottom: 40px;
    }


    /* =====================================================
       TITLE
    ===================================================== */

    .main-title {
        text-align: center;
        color: #173f6f;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .main-subtitle {
        text-align: center;
        color: #5f7185;
        font-size: 17px;
        margin-bottom: 35px;
    }


    /* =====================================================
       INPUT SECTION
    ===================================================== */

    .input-heading {
        color: #173f6f;
        font-size: 24px;
        font-weight: 750;
        margin-bottom: 20px;
    }


    /* =====================================================
       LABELS
    ===================================================== */

    label {
        color: #24496d !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       INPUT BOXES
    ===================================================== */

    div[data-baseweb="select"] > div {
        background-color: #fffdf8;
        border: 1px solid #d8cbb9;
        border-radius: 10px;
    }

    .stTextInput input {
        background-color: #fffdf8 !important;
        border: 1px solid #d8cbb9 !important;
        border-radius: 10px !important;
        color: #24496d !important;
    }


    /* =====================================================
       RADIO BUTTONS
    ===================================================== */

    div[role="radiogroup"] label {
        color: #24496d !important;
    }


    /* =====================================================
       PREDICT BUTTON
    ===================================================== */

    div.stButton > button {

        width: 100%;

        background: linear-gradient(
            135deg,
            #2d78bd 0%,
            #1d5a96 52%,
            #123e6b 100%
        );

        color: white !important;

        border: 2px solid #8db8df;

        border-radius: 12px;

        font-size: 18px;
        font-weight: 700;

        padding: 14px 25px;

        box-shadow:
            0 4px 0 #0d3155,
            0 8px 18px rgba(20, 65, 105, 0.25);

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {

        background: linear-gradient(
            135deg,
            #3a8bd0 0%,
            #246baa 52%,
            #174b80 100%
        );

        border-color: #a9cceb;

        transform: translateY(-1px);

        box-shadow:
            0 5px 0 #0d3155,
            0 10px 22px rgba(20, 65, 105, 0.30);
    }

    div.stButton > button:active {

        transform: translateY(3px);

        box-shadow:
            0 2px 0 #0d3155,
            0 4px 10px rgba(20, 65, 105, 0.20);
    }


    /* =====================================================
       RESULT HEADING
    ===================================================== */

    .result-heading {
        color: #173f6f;
        font-size: 27px;
        font-weight: 800;
        margin-top: 35px;
        margin-bottom: 18px;
    }


    /* =====================================================
       SCORE CARD
    ===================================================== */

    .score-card {

        background: linear-gradient(
            135deg,
            #ffffff,
            #fff8ec
        );

        border: 1px solid #e0d1bd;

        border-radius: 17px;

        padding: 25px;

        text-align: center;

        box-shadow:
            0 8px 20px rgba(60, 45, 25, 0.08);
    }

    .score-label {
        color: #5f7185;
        font-size: 17px;
        font-weight: 600;
    }

    .score-value {
        color: #1b527f;
        font-size: 40px;
        font-weight: 800;
        margin-top: 8px;
    }

    .score-scale {
        color: #8793a0;
        font-size: 13px;
        margin-top: 6px;
    }


    /* =====================================================
       RISK CARDS
    ===================================================== */

    .risk-card {

        border-radius: 17px;

        padding: 25px;

        text-align: center;

        min-height: 150px;
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
        color: #5f7185;
        font-size: 15px;
        font-weight: 600;
    }

    .risk-value {
        color: #24496d;
        font-size: 30px;
        font-weight: 800;
        margin-top: 7px;
    }

    .risk-description {
        color: #68798a;
        font-size: 14px;
        margin-top: 8px;
    }


    /* =====================================================
       INFORMATION CARD
    ===================================================== */

    .info-card {

        background: #fffaf2;

        border: 1px solid #e4d7c5;

        border-radius: 15px;

        padding: 18px 22px;

        margin-top: 25px;

        color: #5f7185;

        font-size: 14px;

        line-height: 1.6;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {

        text-align: center;

        color: #7a8793;

        font-size: 13px;

        margin-top: 35px;

        padding-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL AND PREPROCESSING
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():

    model = tf.keras.models.load_model(
        "improved_model.keras",
        compile=False
    )

    encoder = joblib.load(
        "encoder.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, encoder, scaler


model, encoder, scaler = load_model_and_preprocessing()


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
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🚦 DL-Based Road Accident Risk Prediction
    </div>

    <div class="main-subtitle">
        Estimate accident risk using road, traffic,
        weather, visibility and time-related conditions.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="input-heading">Enter Road & Environmental Conditions</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    city = st.selectbox(
        "City",
        [
            "Bangalore",
            "Chandigarh",
            "Chennai",
            "Delhi",
            "Hyderabad",
            "Kolkata",
            "Mumbai",
            "Pune",
            "Other"
        ],
        index=3
    )

    custom_city = ""

    if city == "Other":

        custom_city = st.text_input(
            "Enter City",
            placeholder="Enter city name"
        )

        if custom_city.strip():

            city = custom_city.strip()


    road_type = st.selectbox(
        "Road Type",
        [
            "highway",
            "rural",
            "urban"
        ],
        index=2
    )


    weather = st.selectbox(
        "Weather",
        [
            "clear",
            "fog",
            "rain"
        ],
        index=0
    )


    visibility = st.selectbox(
        "Visibility",
        [
            "high",
            "low",
            "medium"
        ],
        index=0
    )


    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "high",
            "low",
            "medium"
        ],
        index=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    date = st.text_input(
        "Date",
        value="2026-09-11",
        placeholder="YYYY-MM-DD"
    )


    time = st.text_input(
        "Time",
        value="10:00",
        placeholder="HH:MM"
    )


    day_of_week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ],
        index=4
    )


    is_weekend = st.radio(
        "Is Weekend?",
        [0, 1],
        index=0,
        horizontal=True
    )


    is_peak_hour = st.radio(
        "Is Peak Hour?",
        [0, 1],
        index=0,
        horizontal=True
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict = st.button(
    "Predict Accident Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    try:

        # ----------------------------------------------------
        # Validate City
        # ----------------------------------------------------

        if not city or city == "Other":

            st.error(
                "Please enter a valid city."
            )

            st.stop()


        # ----------------------------------------------------
        # Validate Date
        # ----------------------------------------------------

        date_obj = pd.to_datetime(
            date,
            errors="raise"
        )


        # ----------------------------------------------------
        # Validate Time
        # ----------------------------------------------------

        time_parts = time.split(":")

        if len(time_parts) != 2:

            raise ValueError(
                "Invalid time format."
            )


        hour = int(time_parts[0])
        minute = int(time_parts[1])


        if hour < 0 or hour > 23:

            raise ValueError(
                "Hour must be between 00 and 23."
            )


        if minute < 0 or minute > 59:

            raise ValueError(
                "Minute must be between 00 and 59."
            )


        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [{
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
            }]
        )


        # ----------------------------------------------------
        # Encode categorical features
        # ----------------------------------------------------

        categorical_data = encoder.transform(
            input_data[categorical_features]
        )


        # ----------------------------------------------------
        # Scale numerical features
        # ----------------------------------------------------

        numerical_data = scaler.transform(
            input_data[numerical_features]
        )


        # ----------------------------------------------------
        # Combine features
        # ----------------------------------------------------

        processed_input = np.hstack(
            [
                categorical_data,
                numerical_data
            ]
        )


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

        risk_score = float(
            np.clip(
                risk_score,
                0.10,
                1.00
            )
        )


        # ----------------------------------------------------
        # Risk category
        # ----------------------------------------------------

        risk_category = get_risk_category(
            risk_score
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="result-heading">Prediction Result</div>',
            unsafe_allow_html=True
        )


        result_col1, result_col2 = st.columns(2)


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                f"""
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
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        with result_col2:

            if risk_category == "Low Risk":

                card_class = "low-risk"
                description = (
                    "The estimated risk level is relatively low."
                )

            elif risk_category == "Medium Risk":

                card_class = "medium-risk"
                description = (
                    "The estimated risk level is moderate."
                )

            else:

                card_class = "high-risk"
                description = (
                    "The estimated risk level is relatively high."
                )


            st.markdown(
                f"""
                <div class="risk-card {card_class}">

                    <div class="risk-label">
                        Risk Category
                    </div>

                    <div class="risk-value">
                        {risk_category}
                    </div>

                    <div class="risk-description">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # INFORMATION
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="info-card">

                <b>Note:</b>
                The predicted risk score is an estimate generated
                by the trained Deep Learning model based on the
                conditions entered by the user. It should not be
                interpreted as a guaranteed prediction of an accident.

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception:

        st.error(
            "Unable to generate prediction. "
            "Please check the Date format (YYYY-MM-DD) "
            "and Time format (HH:MM)."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        DL-Based Road Accident Risk Prediction Using Explainable AI

    </div>
    """,
    unsafe_allow_html=True
)
