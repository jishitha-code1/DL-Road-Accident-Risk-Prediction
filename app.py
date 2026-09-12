import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime, date, time


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚦",
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

    /* ======================================================
       MAIN PAGE
       ====================================================== */

    .stApp {
        background-color: #ADD8E6;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ADD8E6;
    }

    [data-testid="stMain"] {
        background-color: #ADD8E6;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    [data-testid="stSidebar"] {
        background-color: #FFE4C4 !important;
    }

    [data-testid="stSidebar"] > div {
        background-color: #FFE4C4 !important;
    }

    [data-testid="stSidebarContent"] {
        background-color: #FFE4C4 !important;
    }

    [data-testid="stSidebarUserContent"] {
        background-color: #FFE4C4 !important;
    }


    /* ======================================================
       GENERAL TEXT
       ====================================================== */

    html,
    body,
    [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    h1,
    h2,
    h3,
    h4 {
        color: #173F68 !important;
    }

    p,
    label,
    span,
    div {
        color: #173F68;
    }


    /* ======================================================
       TITLE
       ====================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #173F68;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 8px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #315878;
        text-align: center;
        margin-bottom: 35px;
    }


    /* ======================================================
       SIDEBAR TITLE
       ====================================================== */

    .sidebar-title {
        font-size: 27px;
        font-weight: 800;
        color: #173F68;
        margin-bottom: 8px;
    }

    .sidebar-subtitle {
        font-size: 16px;
        color: #173F68;
        margin-bottom: 35px;
    }

    .sidebar-description {
        font-size: 16px;
        line-height: 1.8;
        color: #173F68;
        margin-bottom: 30px;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    .info-card {
        background-color: #F7FBFD;
        border: 1px solid #D5E2E9;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 25px;
        box-shadow: 0px 7px 20px rgba(30, 70, 100, 0.10);
    }

    .result-card {
        background-color: #F7FBFD;
        border: 1px solid #D5E2E9;
        border-radius: 22px;
        padding: 45px 30px;
        margin-top: 15px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0px 10px 28px rgba(30, 70, 100, 0.12);
    }


    /* ======================================================
       SECTION TITLES
       ====================================================== */

    .section-title {
        font-size: 27px;
        font-weight: 800;
        color: #173F68;
        margin-top: 25px;
        margin-bottom: 25px;
    }


    /* ======================================================
       PREDICTION BUTTON
       ====================================================== */

    div.stButton > button {
        width: 100%;
        height: 58px;

        background-color: #FFE4C4 !important;

        color: #173F68 !important;

        border: 1px solid #F19AAA !important;
        border-radius: 15px !important;

        font-size: 17px !important;
        font-weight: 700 !important;

        box-shadow: 0px 7px 15px rgba(90, 50, 60, 0.15);

        transition: all 0.2s ease-in-out;
    }

    div.stButton > button:hover {
        background-color: #F5AFC0 !important;
        color: #173F68 !important;
        border-color: #E99AAA !important;
        transform: translateY(-1px);
    }


    /* ======================================================
       INPUT BOXES
       ====================================================== */

    div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] > div {
        background-color: #F8FAFC !important;
        border-radius: 10px !important;
    }

    input {
        color: #173F68 !important;
    }


    /* ======================================================
       DATE/TIME INPUT
       ====================================================== */

    [data-testid="stDateInput"] > div,
    [data-testid="stTimeInput"] > div {
        background-color: #F8FAFC !important;
        border-radius: 10px;
    }


    /* ======================================================
       SUMMARY BAR
       ====================================================== */

    .summary-bar {
        background-color: #F7FBFD;
        border: 1px solid #D5E2E9;
        border-radius: 14px;
        padding: 15px 20px;
        margin-top: 18px;
        margin-bottom: 25px;
        font-size: 16px;
        color: #173F68;
    }


    /* ======================================================
       SCORE
       ====================================================== */

    .result-title {
        font-size: 20px;
        font-weight: 700;
        color: #315878;
        margin-bottom: 10px;
    }

    .score {
        font-size: 48px;
        font-weight: 800;
        color: #173F68;
        margin: 5px 0 18px 0;
    }

    .low-risk {
        display: inline-block;
        background-color: #D8F5E5;
        color: #16824A;
        padding: 10px 28px;
        border-radius: 25px;
        font-size: 17px;
        font-weight: 750;
    }

    .medium-risk {
        display: inline-block;
        background-color: #FFF1C9;
        color: #B77900;
        padding: 10px 28px;
        border-radius: 25px;
        font-size: 17px;
        font-weight: 750;
    }

    .high-risk {
        display: inline-block;
        background-color: #FFE0E0;
        color: #C62828;
        padding: 10px 28px;
        border-radius: 25px;
        font-size: 17px;
        font-weight: 750;
    }


    /* ======================================================
       SIDEBAR RISK LEVELS
       ====================================================== */

    .risk-row {
        display: flex;
        align-items: center;
        margin-top: 22px;
        margin-bottom: 7px;
    }

    .risk-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 12px;
    }

    .low-dot {
        background-color: #43D17F;
    }

    .medium-dot {
        background-color: #F4C542;
    }

    .high-dot {
        background-color: #EF5B67;
    }

    .risk-name {
        font-size: 17px;
        font-weight: 700;
        color: #173F68;
    }

    .risk-score-text {
        font-size: 15px;
        color: #315878;
        margin-bottom: 18px;
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    .custom-divider {
        height: 1px;
        background-color: rgba(23, 63, 104, 0.20);
        margin: 30px 0;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer-text {
        color: #315878;
        font-size: 14px;
        text-align: center;
        margin-top: 35px;
    }


    /* ======================================================
       EXPANDER
       ====================================================== */

    [data-testid="stExpander"] {
        background-color: #F7FBFD;
        border: 1px solid #D5E2E9;
        border-radius: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FEATURE DEFINITIONS
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
# LOAD MODEL + PREPROCESSING
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Encoder file not found: {ENCODER_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    encoder = joblib.load(
        ENCODER_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, encoder, scaler


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
# PEAK HOUR CALCULATION
# ============================================================

def calculate_peak_hour(hour):

    # Morning peak: 7–10
    # Evening peak: 17–20

    if 7 <= hour <= 10:
        return 1

    if 17 <= hour <= 20:
        return 1

    return 0


# ============================================================
# PREPROCESS INPUT
#
# THIS MATCHES THE NOTEBOOK PREDICTION FUNCTION
# ============================================================

def preprocess_input(
    city,
    road_type,
    weather,
    visibility,
    traffic_density,
    date_value,
    time_value,
    day_of_week,
    is_weekend,
    is_peak_hour,
    encoder,
    scaler
):

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    date_obj = pd.to_datetime(
        date_value
    )

    # --------------------------------------------------------
    # Convert time
    # --------------------------------------------------------

    time_string = time_value.strftime("%H:%M")

    hour = int(
        time_string.split(":")[0]
    )

    minute = int(
        time_string.split(":")[1]
    )

    # --------------------------------------------------------
    # Create input DataFrame
    #
    # SAME FEATURES AS NOTEBOOK
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CATEGORICAL ENCODING
    # --------------------------------------------------------

    categorical_data = encoder.transform(
        input_data[categorical_features]
    )

    # --------------------------------------------------------
    # Handle sparse encoder if required
    # --------------------------------------------------------

    if hasattr(
        categorical_data,
        "toarray"
    ):
        categorical_data = (
            categorical_data.toarray()
        )

    # --------------------------------------------------------
    # NUMERICAL SCALING
    #
    # IMPORTANT:
    # Uses the ORIGINAL saved scaler.
    # Do NOT fit the scaler here.
    # --------------------------------------------------------

    numerical_data = scaler.transform(
        input_data[numerical_features]
    )

    # --------------------------------------------------------
    # Combine categorical + numerical
    # --------------------------------------------------------

    processed_input = np.hstack([
        categorical_data,
        numerical_data
    ])

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    processed_input = processed_input.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Validate 35 features
    # --------------------------------------------------------

    if processed_input.shape[1] != 35:

        raise ValueError(
            f"Expected 35 features, "
            f"but received {processed_input.shape[1]}."
        )

    return processed_input


# ============================================================
# EXACT MODEL PREDICTION
#
# Matches notebook:
#
# improved_model.predict(
#     processed_input,
#     verbose=0
# )[0][0]
# ============================================================

def predict_risk(
    model,
    processed_input
):

    prediction = model.predict(
        processed_input,
        verbose=0
    )[0][0]

    return float(prediction)


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(model):

    background = np.zeros(
        (1, 35),
        dtype=np.float32
    )

    try:

        explainer = shap.DeepExplainer(
            model,
            background
        )

        return explainer

    except Exception:

        return None


# ============================================================
# SHAP FEATURE NAMES
# ============================================================

def get_feature_names(
    encoder
):

    encoded_names = list(
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    return (
        encoded_names
        + numerical_features
    )


# ============================================================
# SHAP VALUES
# ============================================================

def calculate_shap_values(
    explainer,
    processed_input
):

    if explainer is None:
        return None

    try:

        shap_values = explainer.shap_values(
            processed_input
        )

        # SHAP may return a list
        if isinstance(
            shap_values,
            list
        ):
            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        )

        # Shape may be:
        # (1, 35, 1)
        # or (1, 35)

        if shap_values.ndim == 3:
            shap_values = np.squeeze(
                shap_values,
                axis=-1
            )

        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(
                1,
                -1
            )

        return shap_values

    except Exception:

        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
            🚦 Road Accident Risk
        </div>

        <div class="sidebar-subtitle">
            Deep Learning + Explainable AI
        </div>

        <div class="sidebar-description">
            This application estimates road accident
            risk based on the conditions entered by
            the user.
        </div>

        <div class="custom-divider"></div>

        <h3>Risk Levels</h3>
        """,
        unsafe_allow_html=True
    )

    # LOW

    st.markdown(
        """
        <div class="risk-row">
            <div class="risk-dot low-dot"></div>
            <div class="risk-name">
                Low Risk
            </div>
        </div>

        <div class="risk-score-text">
            Score &lt; 0.25
        </div>
        """,
        unsafe_allow_html=True
    )

    # MEDIUM

    st.markdown(
        """
        <div class="risk-row">
            <div class="risk-dot medium-dot"></div>
            <div class="risk-name">
                Medium Risk
            </div>
        </div>

        <div class="risk-score-text">
            Score 0.25 – &lt; 0.60
        </div>
        """,
        unsafe_allow_html=True
    )

    # HIGH

    st.markdown(
        """
        <div class="risk-row">
            <div class="risk-dot high-dot"></div>
            <div class="risk-name">
                High Risk
            </div>
        </div>

        <div class="risk-score-text">
            Score ≥ 0.60
        </div>

        <div class="custom-divider"></div>

        <div style="
            font-size:14px;
            color:#315878;
            margin-top:20px;
        ">
            DL-Based Road Accident Risk Prediction
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🚦 Road Accident Risk Prediction
    </div>

    <div class="main-subtitle">
        Deep Learning based risk estimation with Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, encoder, scaler = (
        load_model_and_preprocessing()
    )

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessing files."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Enter Road Conditions</div>',
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
        ],
        index=0
    )


with col2:

    road_type = st.selectbox(
        "Road Type",
        [
            "highway",
            "rural",
            "urban"
        ],
        index=0
    )


with col3:

    weather = st.selectbox(
        "Weather",
        [
            "clear",
            "fog",
            "rain"
        ],
        index=0
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
        ],
        index=0
    )


with col5:

    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "high",
            "low",
            "medium"
        ],
        index=0
    )


with col6:

    selected_date = st.date_input(
        "Date",
        value=date(2026, 9, 11)
    )


# ============================================================
# AUTOMATIC DAY OF WEEK
# ============================================================

selected_date_timestamp = pd.to_datetime(
    selected_date
)

day_of_week = (
    selected_date_timestamp.strftime("%A")
)


# ============================================================
# AUTOMATIC WEEKEND VALUE
# ============================================================

automatic_weekend = (
    1
    if selected_date_timestamp.weekday() >= 5
    else 0
)


# ============================================================
# TIME
# ============================================================

col7, col8, col9 = st.columns(3)


with col7:

    selected_time = st.time_input(
        "Time",
        value=time(
            datetime.now().hour,
            datetime.now().minute
        )
    )


# ============================================================
# AUTOMATIC PEAK HOUR
# ============================================================

automatic_peak_hour = calculate_peak_hour(
    selected_time.hour
)


# ============================================================
# SESSION STATE FOR WEEKEND
# ============================================================

if "is_weekend" not in st.session_state:

    st.session_state.is_weekend = (
        automatic_weekend
    )


if "previous_date" not in st.session_state:

    st.session_state.previous_date = (
        selected_date
    )


# If date changed, update automatic weekend

if (
    selected_date
    != st.session_state.previous_date
):

    st.session_state.is_weekend = (
        automatic_weekend
    )

    st.session_state.previous_date = (
        selected_date
    )


# ============================================================
# SESSION STATE FOR PEAK HOUR
# ============================================================

if "is_peak_hour" not in st.session_state:

    st.session_state.is_peak_hour = (
        automatic_peak_hour
    )


if "previous_time" not in st.session_state:

    st.session_state.previous_time = (
        selected_time
    )


# If time changed, update automatic peak hour

if (
    selected_time
    != st.session_state.previous_time
):

    st.session_state.is_peak_hour = (
        automatic_peak_hour
    )

    st.session_state.previous_time = (
        selected_time
    )


# ============================================================
# WEEKEND
# ============================================================

with col8:

    is_weekend = st.radio(
        "Is Weekend",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No",
        horizontal=True,
        key="is_weekend"
    )


# ============================================================
# PEAK HOUR
# ============================================================

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
# INPUT SUMMARY
# ============================================================

weekend_text = (
    "Yes"
    if is_weekend == 1
    else "No"
)

peak_text = (
    "Yes"
    if is_peak_hour == 1
    else "No"
)

st.markdown(
    f"""
    <div class="summary-bar">

        📅 {selected_date.strftime("%Y-%m-%d")}
        &nbsp; | &nbsp;

        📅 {day_of_week}
        &nbsp; | &nbsp;

        🕐 {selected_time.strftime("%H:%M")}
        &nbsp; | &nbsp;

        Weekend: <b>{weekend_text}</b>
        &nbsp; | &nbsp;

        Peak Hour: <b>{peak_text}</b>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "📈 Predict Accident Risk",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_input = preprocess_input(

            city=city,

            road_type=road_type,

            weather=weather,

            visibility=visibility,

            traffic_density=traffic_density,

            date_value=selected_date,

            time_value=selected_time,

            day_of_week=day_of_week,

            is_weekend=is_weekend,

            is_peak_hour=is_peak_hour,

            encoder=encoder,

            scaler=scaler
        )

        # ----------------------------------------------------
        # PREDICT
        #
        # EXACT NOTEBOOK METHOD
        # ----------------------------------------------------

        risk_score = predict_risk(
            model,
            processed_input
        )

        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        risk_category = get_risk_category(
            risk_score
        )

        # ----------------------------------------------------
        # RESULT TITLE
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # CATEGORY CSS
        # ----------------------------------------------------

        if risk_category == "Low Risk":

            category_class = "low-risk"

        elif risk_category == "Medium Risk":

            category_class = "medium-risk"

        else:

            category_class = "high-risk"


        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-title">
                    Predicted Risk Score
                </div>

                <div class="score">
                    {risk_score:.4f}
                </div>

                <div class="{category_class}">
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
            '<div class="section-title">Input Summary</div>',
            unsafe_allow_html=True
        )

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.markdown(
                f"""
                <div class="info-card">

                    <b>City:</b> {city}<br><br>

                    <b>Road Type:</b> {road_type}<br><br>

                    <b>Weather:</b> {weather}<br><br>

                    <b>Visibility:</b> {visibility}<br><br>

                    <b>Traffic Density:</b>
                    {traffic_density}

                </div>
                """,
                unsafe_allow_html=True
            )


        with summary_col2:

            st.markdown(
                f"""
                <div class="info-card">

                    <b>Date:</b>
                    {selected_date.strftime("%Y-%m-%d")}<br><br>

                    <b>Day:</b>
                    {day_of_week}<br><br>

                    <b>Time:</b>
                    {selected_time.strftime("%H:%M")}<br><br>

                    <b>Weekend:</b>
                    {weekend_text}<br><br>

                    <b>Peak Hour:</b>
                    {peak_text}

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-title">Explainable AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">

            SHAP helps explain how the input features
            contributed to the predicted risk score.

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        with st.spinner(
            "Generating explanation..."
        ):

            explainer = create_shap_explainer(
                model
            )

            shap_values = calculate_shap_values(
                explainer,
                processed_input
            )


        if shap_values is not None:

            feature_names = get_feature_names(
                encoder
            )

            values = shap_values[0]

            if len(feature_names) == len(values):

                shap_df = pd.DataFrame({

                    "Feature": feature_names,

                    "SHAP Value": values,

                    "Absolute Impact":
                        np.abs(values)

                })

                shap_df = shap_df.sort_values(
                    "Absolute Impact",
                    ascending=False
                )

                top_features = shap_df.head(
                    10
                ).sort_values(
                    "Absolute Impact",
                    ascending=True
                )


                # ------------------------------------------------
                # SHAP BAR CHART
                # ------------------------------------------------

                fig, ax = plt.subplots(
                    figsize=(9, 5)
                )

                ax.barh(
                    top_features["Feature"],
                    top_features["SHAP Value"]
                )

                ax.set_xlabel(
                    "SHAP Value"
                )

                ax.set_ylabel(
                    "Feature"
                )

                ax.set_title(
                    "Top Features Contributing to Risk Prediction"
                )

                plt.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True
                )

                plt.close(fig)


                # ------------------------------------------------
                # SHAP TABLE
                # ------------------------------------------------

                st.markdown(
                    """
                    <div class="info-card">
                    <b>Feature Contribution Details</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                display_df = shap_df.head(
                    10
                )[[
                    "Feature",
                    "SHAP Value"
                ]].copy()

                display_df["SHAP Value"] = (
                    display_df["SHAP Value"]
                    .round(4)
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.warning(
                    "SHAP feature names do not match "
                    "the model input size."
                )

        else:

            st.info(
                "The prediction was generated successfully, "
                "but SHAP explanation could not be generated "
                "for this session."
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander(
            "Model and preprocessing information"
        ):

            st.write(
                "Processed feature count:",
                processed_input.shape[1]
            )

            st.write(
                "Categorical features:",
                len(
                    encoder.get_feature_names_out(
                        categorical_features
                    )
                )
            )

            st.write(
                "Numerical features:",
                len(numerical_features)
            )

            st.write(
                "Total features:",
                processed_input.shape[1]
            )

            st.write(
                "Prediction method:",
                "Deep Learning model.predict()"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div class="info-card">

            <b>Note:</b>
            This application provides an estimated road
            accident risk score based on the conditions
            entered by the user. It does not guarantee that
            an accident will or will not occur.

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer-text">
        DL-Based Road Accident Risk Prediction
        &nbsp; | &nbsp;
        Deep Learning + Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)
