import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "improved_model.keras"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------
       MAIN BACKGROUND
    ------------------------------ */

    .stApp {
        background: #f8f1e3;
        color: #17365d;
    }

    .main {
        background: #f8f1e3;
    }

    [data-testid="stAppViewContainer"] {
        background: #f8f1e3;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 241, 227, 0.95);
    }


    /* ------------------------------
       SIDEBAR
    ------------------------------ */

    [data-testid="stSidebar"] {
        background: #eef4f8;
        border-right: 1px solid #d7e1e8;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17365d;
    }


    /* ------------------------------
       HEADINGS
    ------------------------------ */

    h1 {
        color: #17365d !important;
        font-weight: 750 !important;
        letter-spacing: -0.5px;
    }

    h2 {
        color: #17365d !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #214d78 !important;
    }

    p, label, span {
        color: #294d70;
    }


    /* ------------------------------
       INPUT LABELS
    ------------------------------ */

    [data-testid="stWidgetLabel"] p {
        color: #17365d !important;
        font-weight: 650 !important;
    }


    /* ------------------------------
       INPUT BOXES
    ------------------------------ */

    div[data-baseweb="select"] > div {
        background: #fffdf8 !important;
        border: 1px solid #c8d8e5 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] > div {
        background: #fffdf8 !important;
        border: 1px solid #c8d8e5 !important;
        border-radius: 10px !important;
    }

    input {
        color: #17365d !important;
        background: #fffdf8 !important;
    }


    /* ------------------------------
       RADIO BUTTONS
    ------------------------------ */

    div[role="radiogroup"] label {
        color: #294d70 !important;
    }


    /* ------------------------------
       MAIN CARD
    ------------------------------ */

    .input-card {
        background: #fffdf8;
        border: 1px solid #dce5eb;
        border-radius: 18px;
        padding: 25px 28px 22px 28px;
        box-shadow: 0 6px 22px rgba(39, 70, 96, 0.08);
        margin-top: 10px;
        margin-bottom: 24px;
    }


    /* ------------------------------
       INFORMATION BAR
    ------------------------------ */

    .info-bar {
        background: #f2f8fc;
        border: 1px solid #cbdfee;
        border-radius: 12px;
        padding: 13px 16px;
        margin-top: 15px;
        margin-bottom: 18px;
        color: #214d78;
        font-size: 14px;
        font-weight: 550;
    }


    /* ------------------------------
       PREDICT BUTTON
       DOUBLE SHADE BLUE
    ------------------------------ */

    div.stButton > button {
        width: 100%;
        min-height: 52px;

        background: linear-gradient(
            90deg,
            #4c9be8 0%,
            #317dcc 48%,
            #205b98 100%
        ) !important;

        color: white !important;
        border: none !important;
        border-radius: 11px !important;

        font-size: 16px !important;
        font-weight: 700 !important;

        box-shadow:
            0 5px 12px rgba(36, 103, 164, 0.25);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(
            90deg,
            #3e8fdf 0%,
            #286fb9 48%,
            #174c83 100%
        ) !important;

        color: white !important;

        transform: translateY(-1px);

        box-shadow:
            0 7px 16px rgba(36, 103, 164, 0.32);
    }


    /* ------------------------------
       RESULT CARD
    ------------------------------ */

    .result-card {
        background: #f5faff;
        border: 1px solid #d5e3ed;
        border-radius: 16px;
        padding: 27px;
        text-align: center;
        box-shadow: 0 5px 18px rgba(39, 70, 96, 0.07);
        margin-top: 10px;
        margin-bottom: 25px;
    }

    .result-label {
        font-size: 16px;
        color: #214d78;
        font-weight: 650;
        margin-bottom: 7px;
    }

    .prediction-score {
        font-size: 42px;
        font-weight: 800;
        color: #17365d;
        margin: 3px 0 10px 0;
    }

    .prediction-category {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 750;
    }


    /* ------------------------------
       SUMMARY CARD
    ------------------------------ */

    .summary-card {
        background: #fffdf8;
        border: 1px solid #dce5eb;
        border-radius: 15px;
        padding: 20px 24px;
        box-shadow: 0 4px 15px rgba(39, 70, 96, 0.06);
    }


    /* ------------------------------
       DIVIDER
    ------------------------------ */

    hr {
        border: none;
        border-top: 1px solid #d5e0e7;
        margin: 22px 0;
    }


    /* ------------------------------
       ALERTS
    ------------------------------ */

    .stAlert {
        border-radius: 12px;
    }


    /* ------------------------------
       FOOTER
    ------------------------------ */

    .footer {
        text-align: center;
        color: #66809a;
        font-size: 13px;
        margin-top: 30px;
        padding: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL + PREPROCESSING
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, encoder, scaler


# ============================================================
# FEATURE NAMES
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


expected_categorical_features = [
    "city_Bangalore",
    "city_Chandigarh",
    "city_Chennai",
    "city_Delhi",
    "city_Hyderabad",
    "city_Kolkata",
    "city_Mumbai",
    "city_Pune",
    "road_type_highway",
    "road_type_rural",
    "road_type_urban",
    "weather_clear",
    "weather_fog",
    "weather_rain",
    "visibility_high",
    "visibility_low",
    "visibility_medium",
    "traffic_density_high",
    "traffic_density_low",
    "traffic_density_medium",
    "day_of_week_Friday",
    "day_of_week_Monday",
    "day_of_week_Saturday",
    "day_of_week_Sunday",
    "day_of_week_Thursday",
    "day_of_week_Tuesday",
    "day_of_week_Wednesday"
]


feature_names = expected_categorical_features + numerical_features


# ============================================================
# RISK CATEGORY
# ============================================================

def get_risk_category(score):

    if score < 0.25:
        return "Low Risk"

    elif score < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# PEAK HOUR CALCULATION
# ============================================================

def calculate_peak_hour(hour):

    # Morning peak: 7–10
    # Evening peak: 17–20

    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 1

    return 0


# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(
    city,
    road_type,
    weather,
    visibility,
    traffic_density,
    day_of_week,
    selected_date,
    selected_time,
    is_weekend,
    is_peak_hour,
    encoder,
    scaler
):

    date_obj = pd.to_datetime(
        selected_date,
        format="%Y-%m-%d"
    )

    time_obj = datetime.strptime(
        selected_time,
        "%H:%M"
    )

    input_df = pd.DataFrame({
        "city": [city],
        "road_type": [road_type],
        "weather": [weather],
        "visibility": [visibility],
        "traffic_density": [traffic_density],
        "day_of_week": [day_of_week]
    })

    encoded = encoder.transform(input_df)

    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    numerical_df = pd.DataFrame({
        "hour": [time_obj.hour],
        "is_weekend": [is_weekend],
        "is_peak_hour": [is_peak_hour],
        "year": [date_obj.year],
        "month": [date_obj.month],
        "date_day": [date_obj.day],
        "time_hour": [time_obj.hour],
        "time_minute": [time_obj.minute]
    })

    scaled = scaler.transform(
        numerical_df[numerical_features]
    )

    final_input = np.hstack([
        encoded,
        scaled
    ])

    final_input = final_input.astype(
        np.float32
    )

    if final_input.shape[1] != 35:
        raise ValueError(
            f"Expected 35 features but received "
            f"{final_input.shape[1]} features."
        )

    return final_input


# ============================================================
# SAFE MODEL PREDICTION
# ============================================================

def safe_predict(model, X):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    # Direct model call avoids the Keras
    # input-structure warning produced by model.predict()

    prediction = model(
        tf.convert_to_tensor(X),
        training=False
    )

    return prediction.numpy().reshape(-1)


# ============================================================
# SHAP
#
# IMPORTANT:
# We use a prediction wrapper instead of passing the Keras
# model directly to SHAP DeepExplainer.
#
# This prevents the:
# "Expected input_layer_3 / Received inputs Tensor"
# warning from appearing repeatedly.
# ============================================================

@st.cache_resource
def create_shap_explainer(model):

    background = np.zeros(
        (1, 35),
        dtype=np.float32
    )

    def prediction_function(X):

        return safe_predict(
            model,
            X
        )

    explainer = shap.Explainer(
        prediction_function,
        background,
        feature_names=feature_names
    )

    return explainer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 8px 0 18px 0;
        ">

        <div style="
            font-size: 25px;
            font-weight: 800;
            color: #17365d;
        ">
        🚗 Road Accident Risk
        </div>

        <div style="
            font-size: 15px;
            color: #52708d;
            margin-top: 4px;
        ">
        Deep Learning + Explainable AI
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size: 14px;
            line-height: 1.7;
            color: #3d5d78;
        ">
        This application estimates road accident risk
        based on the road and environmental conditions
        entered by the user.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <h3 style="
            color:#17365d;
            margin-bottom:20px;
        ">
        Risk Levels
        </h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="margin-bottom:22px;">

            <div style="
                display:flex;
                align-items:center;
                gap:12px;
            ">
                <div style="
                    width:28px;
                    height:28px;
                    border-radius:50%;
                    background:#35b879;
                "></div>

                <div>
                    <b style="color:#17365d;">
                    Low Risk
                    </b>

                    <br>

                    <span style="
                        color:#5c7891;
                        font-size:13px;
                    ">
                    Score &lt; 0.25
                    </span>
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="margin-bottom:22px;">

            <div style="
                display:flex;
                align-items:center;
                gap:12px;
            ">
                <div style="
                    width:28px;
                    height:28px;
                    border-radius:50%;
                    background:#f5b82e;
                "></div>

                <div>
                    <b style="color:#17365d;">
                    Medium Risk
                    </b>

                    <br>

                    <span style="
                        color:#5c7891;
                        font-size:13px;
                    ">
                    Score 0.25 – &lt; 0.60
                    </span>
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="margin-bottom:22px;">

            <div style="
                display:flex;
                align-items:center;
                gap:12px;
            ">
                <div style="
                    width:28px;
                    height:28px;
                    border-radius:50%;
                    background:#ef4444;
                "></div>

                <div>
                    <b style="color:#17365d;">
                    High Risk
                    </b>

                    <br>

                    <span style="
                        color:#5c7891;
                        font-size:13px;
                    ">
                    Score ≥ 0.60
                    </span>
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            color:#365b7d;
            font-size:13px;
            line-height:1.5;
        ">
        <b>🧠 DL-Based Road Accident<br>
        Risk Prediction</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:5px;
        margin-bottom:20px;
    ">

        <h1 style="
            font-size:35px;
            margin-bottom:4px;
        ">
        🚗 Road Accident Risk Prediction
        </h1>

        <div style="
            font-size:17px;
            color:#52708d;
        ">
        Deep Learning based risk estimation with
        Explainable AI
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, encoder, scaler = load_model_and_preprocessing()

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessing files."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# CURRENT DATE / TIME
# ============================================================

now = datetime.now()


# ============================================================
# SESSION STATE
# ============================================================

if "is_weekend" not in st.session_state:
    st.session_state.is_weekend = (
        1 if now.weekday() >= 5 else 0
    )

if "is_peak_hour" not in st.session_state:
    st.session_state.is_peak_hour = calculate_peak_hour(
        now.hour
    )

if "last_selected_date" not in st.session_state:
    st.session_state.last_selected_date = now.date()

if "last_selected_time" not in st.session_state:
    st.session_state.last_selected_time = now.time().replace(
        second=0,
        microsecond=0
    )


# ============================================================
# INPUT CARD
# ============================================================

st.markdown(
    """
    <div class="input-card">
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h2 style="
        font-size:21px;
        margin-top:0;
        margin-bottom:20px;
    ">
    Enter Road Conditions
    </h2>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


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
            "Pune"
        ]
    )


with col2:

    road_type = st.selectbox(
        "Road Type",
        [
            "highway",
            "rural",
            "urban"
        ]
    )


with col3:

    weather = st.selectbox(
        "Weather",
        [
            "clear",
            "fog",
            "rain"
        ]
    )


# ============================================================
# ROW 2
# ============================================================

col4, col5, col6 = st.columns(3)


with col4:

    visibility = st.selectbox(
        "Visibility",
        [
            "high",
            "low",
            "medium"
        ]
    )


with col5:

    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "high",
            "low",
            "medium"
        ]
    )


with col6:

    selected_date_obj = st.date_input(
        "Date",
        value=now.date()
    )


# ============================================================
# ROW 3
# ============================================================

col7, col8, col9 = st.columns(3)


with col7:

    selected_time_obj = st.time_input(
        "Time",
        value=now.time().replace(
            second=0,
            microsecond=0
        )
    )


# ============================================================
# DERIVED VALUES
# ============================================================

automatic_weekend = (
    1
    if selected_date_obj.weekday() >= 5
    else 0
)

automatic_peak_hour = calculate_peak_hour(
    selected_time_obj.hour
)


# ============================================================
# DETECT DATE / TIME CHANGES
#
# If the user changes Date:
# automatically update Weekend.
#
# If the user changes Time:
# automatically update Peak Hour.
#
# But the radio buttons remain editable.
# ============================================================

date_changed = (
    selected_date_obj
    != st.session_state.last_selected_date
)

time_changed = (
    selected_time_obj
    != st.session_state.last_selected_time
)


if date_changed:

    st.session_state.is_weekend = automatic_weekend

    st.session_state.last_selected_date = (
        selected_date_obj
    )


if time_changed:

    st.session_state.is_peak_hour = automatic_peak_hour

    st.session_state.last_selected_time = (
        selected_time_obj
    )


with col8:

    is_weekend = st.radio(
        "Is Weekend",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No",
        horizontal=True,
        key="is_weekend"
    )


with col9:

    is_peak_hour = st.radio(
        "Is Peak Hour",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No",
        horizontal=True,
        key="is_peak_hour"
    )


# ============================================================
# DAY OF WEEK
# ============================================================

day_of_week = selected_date_obj.strftime(
    "%A"
)


selected_date = selected_date_obj.strftime(
    "%Y-%m-%d"
)

selected_time = selected_time_obj.strftime(
    "%H:%M"
)


# ============================================================
# INFORMATION BAR
# ============================================================

weekend_text = (
    "Yes" if is_weekend == 1 else "No"
)

peak_text = (
    "Yes" if is_peak_hour == 1 else "No"
)

st.markdown(
    f"""
    <div class="info-bar">

        📅 &nbsp;
        <b>Selected Date:</b>
        {selected_date}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        📆 &nbsp;
        <b>Day:</b>
        {day_of_week}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        🕐 &nbsp;
        <b>Selected Time:</b>
        {selected_time}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        ☀️ &nbsp;
        <b>Weekend:</b>
        {weekend_text}

        &nbsp;&nbsp; | &nbsp;&nbsp;

        📈 &nbsp;
        <b>Peak Hour:</b>
        {peak_text}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_clicked = st.button(
    "📈  Predict Accident Risk",
    use_container_width=True
)


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    try:

        # -----------------------------------------
        # PREPROCESS
        # -----------------------------------------

        final_input = preprocess_input(
            city=city,
            road_type=road_type,
            weather=weather,
            visibility=visibility,
            traffic_density=traffic_density,
            day_of_week=day_of_week,
            selected_date=selected_date,
            selected_time=selected_time,
            is_weekend=is_weekend,
            is_peak_hour=is_peak_hour,
            encoder=encoder,
            scaler=scaler
        )


        # -----------------------------------------
        # PREDICT
        # -----------------------------------------

        prediction = safe_predict(
            model,
            final_input
        )

        risk_score = float(
            prediction[0]
        )


        # Keep score in valid probability range
        risk_score = float(
            np.clip(
                risk_score,
                0.0,
                1.0
            )
        )


        risk_category = get_risk_category(
            risk_score
        )


        # -----------------------------------------
        # CATEGORY STYLE
        # -----------------------------------------

        if risk_category == "Low Risk":

            category_background = "#d9f6e6"
            category_text = "#16834a"

        elif risk_category == "Medium Risk":

            category_background = "#fff0c7"
            category_text = "#a76b00"

        else:

            category_background = "#ffe0e0"
            category_text = "#c62828"


        # -----------------------------------------
        # RESULT HEADING
        # -----------------------------------------

        st.markdown(
            """
            <h2 style="
                margin-top:25px;
                margin-bottom:12px;
                font-size:23px;
            ">
            Prediction Result
            </h2>
            """,
            unsafe_allow_html=True
        )


        # -----------------------------------------
        # RESULT CARD
        # -----------------------------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Predicted Risk Score
                </div>

                <div class="prediction-score">
                    {risk_score:.4f}
                </div>

                <div
                    class="prediction-category"
                    style="
                        background:{category_background};
                        color:{category_text};
                    "
                >
                    {risk_category}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.markdown(
            """
            <h2 style="
                font-size:21px;
                margin-top:25px;
            ">
            Input Summary
            </h2>
            """,
            unsafe_allow_html=True
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.markdown(
                f"""
                <div class="summary-card">

                <b>Location & Road</b>

                <br><br>

                📍 City:
                <b>{city}</b>

                <br>

                🛣️ Road Type:
                <b>{road_type}</b>

                <br>

                🚗 Traffic Density:
                <b>{traffic_density}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        with summary_col2:

            st.markdown(
                f"""
                <div class="summary-card">

                <b>Environment & Time</b>

                <br><br>

                🌦️ Weather:
                <b>{weather}</b>

                <br>

                👁️ Visibility:
                <b>{visibility}</b>

                <br>

                📅 {day_of_week},
                {selected_date}

                <br>

                🕐 Time:
                <b>{selected_time}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            """
            <h2 style="
                font-size:21px;
                margin-top:30px;
            ">
            Explainable AI — SHAP
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "SHAP shows which input features contributed "
            "most to the model's prediction."
        )


        with st.spinner(
            "Generating explanation..."
        ):

            try:

                explainer = create_shap_explainer(
                    model
                )

                shap_result = explainer(
                    final_input
                )

                shap_values = shap_result.values

                if shap_values.ndim == 3:

                    shap_values = np.squeeze(
                        shap_values,
                        axis=-1
                    )

                shap_values = shap_values.reshape(
                    -1,
                    35
                )


                # -----------------------------------------
                # TOP 10 FEATURES
                # -----------------------------------------

                absolute_values = np.abs(
                    shap_values[0]
                )

                top_indices = np.argsort(
                    absolute_values
                )[-10:][::-1]


                top_names = [
                    feature_names[i]
                    for i in top_indices
                ]

                top_values = [
                    shap_values[0][i]
                    for i in top_indices
                ]


                # -----------------------------------------
                # SHAP CHART
                # -----------------------------------------

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.barh(
                    top_names[::-1],
                    top_values[::-1]
                )

                ax.axvline(
                    0,
                    linewidth=1
                )

                ax.set_xlabel(
                    "SHAP Value"
                )

                ax.set_title(
                    "Top 10 Features Affecting Prediction"
                )

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)


                # -----------------------------------------
                # SHAP TABLE
                # -----------------------------------------

                shap_table = pd.DataFrame({
                    "Feature": top_names,
                    "SHAP Value": [
                        round(float(v), 5)
                        for v in top_values
                    ],
                    "Impact": [
                        "Increases Risk"
                        if v > 0
                        else "Decreases Risk"
                        for v in top_values
                    ]
                })

                st.dataframe(
                    shap_table,
                    use_container_width=True,
                    hide_index=True
                )


            except Exception as shap_error:

                st.warning(
                    "Prediction was successful, but the "
                    "SHAP explanation could not be generated."
                )

                st.caption(
                    str(shap_error)
                )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div style="
                background:#fff8e8;
                border:1px solid #ead9ae;
                border-radius:12px;
                padding:15px 18px;
                margin-top:25px;
                color:#66552f;
                font-size:13px;
                line-height:1.6;
            ">

            <b>Note:</b>
            The predicted score is an estimated risk based
            on the conditions entered into the model.
            It does not guarantee that an accident will
            occur.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        DL-Based Road Accident Risk Prediction
        &nbsp; • &nbsp;
        Deep Learning + Explainable AI

    </div>
    """,
    unsafe_allow_html=True
)
