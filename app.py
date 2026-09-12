import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI STYLE
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f4f8ff 0%,
            #eef6f7 50%,
            #f8f5ff 100%
        );
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #17324d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #52677a;
        margin-bottom: 25px;
    }

    /* Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #dce7f0;
        box-shadow: 0 6px 20px rgba(40, 70, 100, 0.08);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #24445e;
        margin-bottom: 12px;
    }

    /* Risk cards */
    .risk-low {
        background: #e8f7ef;
        border-left: 7px solid #28a66a;
        padding: 20px;
        border-radius: 14px;
    }

    .risk-medium {
        background: #fff6df;
        border-left: 7px solid #e4a72c;
        padding: 20px;
        border-radius: 14px;
    }

    .risk-high {
        background: #fdebec;
        border-left: 7px solid #df5360;
        padding: 20px;
        border-radius: 14px;
    }

    .risk-score {
        font-size: 34px;
        font-weight: 800;
        color: #17324d;
    }

    .risk-label {
        font-size: 22px;
        font-weight: 700;
        margin-top: 5px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        font-size: 17px;
        font-weight: 700;
        background: #2d6cdf;
        color: white;
        border: none;
    }

    .stButton > button:hover {
        background: #1f58bf;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f7fbff;
        border-right: 1px solid #dbe7f0;
    }

    /* Input labels */
    label {
        color: #29465d !important;
        font-weight: 600 !important;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #d9e4ed;
        margin: 25px 0;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "improved_model.keras"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ============================================================
# LOAD MODEL AND PREPROCESSING
# ============================================================

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, encoder, scaler


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
# PREPROCESS INPUT
# ============================================================

def preprocess_input(
    city,
    road_type,
    weather,
    visibility,
    traffic_density,
    date,
    time,
    day_of_week,
    is_weekend,
    is_peak_hour,
    encoder,
    scaler
):

    # Validate date
    date_obj = pd.to_datetime(
        date,
        format="%Y-%m-%d",
        errors="raise"
    )

    # Validate time
    time_parts = str(time).strip().split(":")

    if len(time_parts) != 2:
        raise ValueError(
            "Time must be in HH:MM format."
        )

    hour = int(time_parts[0])
    minute = int(time_parts[1])

    if not (0 <= hour <= 23):
        raise ValueError(
            "Hour must be between 00 and 23."
        )

    if not (0 <= minute <= 59):
        raise ValueError(
            "Minute must be between 00 and 59."
        )

    # Create input dataframe
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

    # Categorical encoding
    categorical_data = encoder.transform(
        input_data[categorical_features]
    )

    # Convert sparse matrix to dense array if necessary
    if hasattr(categorical_data, "toarray"):
        categorical_data = categorical_data.toarray()

    # Numerical scaling
    numerical_data = scaler.transform(
        input_data[numerical_features]
    )

    # Combine exactly 35 features
    processed_input = np.hstack([
        categorical_data,
        numerical_data
    ])

    if processed_input.shape[1] != 35:
        raise ValueError(
            f"Expected 35 features but got "
            f"{processed_input.shape[1]}."
        )

    return processed_input.astype(np.float32)


# ============================================================
# SHAP EXPLANATION
# ============================================================

@st.cache_resource
def create_explainer(_model):

    background = np.zeros(
        (1, 35),
        dtype=np.float32
    )

    return shap.DeepExplainer(
        _model,
        background
    )


# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(encoder):

    encoded_names = list(
        encoder.get_feature_names_out(
            categorical_features
        )
    )

    return encoded_names + numerical_features


# ============================================================
# SHAP PLOT
# ============================================================

def create_shap_plot(
    shap_values,
    feature_names
):

    shap_array = np.squeeze(
        np.array(shap_values)
    )

    if shap_array.ndim != 1:
        raise ValueError(
            "Unexpected SHAP output shape."
        )

    if len(shap_array) != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match model features."
        )

    contributions = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": shap_array
    })

    contributions["Abs_SHAP"] = (
        contributions["SHAP"].abs()
    )

    # Top 10 features
    contributions = (
        contributions
        .sort_values(
            "Abs_SHAP",
            ascending=False
        )
        .head(10)
        .sort_values("SHAP")
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
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
        "Feature"
    )

    ax.set_title(
        "Top Factors Affecting the Prediction"
    )

    plt.tight_layout()

    return fig, contributions


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    model, encoder, scaler = load_artifacts()

    explainer = create_explainer(model)

    feature_names = get_feature_names(
        encoder
    )

except Exception as e:

    st.error(
        "Unable to load the trained model or preprocessing files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚗 Road Accident Risk Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning-based prediction with Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🚦 Prediction System"
    )

    st.write(
        "Enter the current road and environmental "
        "conditions to estimate accident risk."
    )

    st.markdown("---")

    st.markdown("### Risk Categories")

    st.write(
        "🟢 **Low Risk**"
    )

    st.write(
        "🟡 **Medium Risk**"
    )

    st.write(
        "🔴 **High Risk**"
    )

    st.markdown("---")

    st.caption(
        "The prediction is an estimated risk score "
        "from the trained Deep Learning model."
    )


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '📍 Road & Environmental Conditions'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


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
        index=3
    )

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


with col2:

    traffic_density = st.selectbox(
        "Traffic Density",
        [
            "high",
            "low",
            "medium"
        ],
        index=1
    )

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


col3, col4 = st.columns(2)


with col3:

    is_weekend = st.radio(
        "Is Weekend? (0 = No, 1 = Yes)",
        [0, 1],
        index=0,
        horizontal=True
    )


with col4:

    is_peak_hour = st.radio(
        "Is Peak Hour? (0 = No, 1 = Yes)",
        [0, 1],
        index=0,
        horizontal=True
    )


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("")

predict_button = st.button(
    "🔍 Predict Accident Risk"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        processed_input = preprocess_input(
            city,
            road_type,
            weather,
            visibility,
            traffic_density,
            date,
            time,
            day_of_week,
            is_weekend,
            is_peak_hour,
            encoder,
            scaler
        )

        # Model prediction
        prediction = model.predict(
            processed_input,
            verbose=0
        )

        risk_score = float(
            prediction[0][0]
        )

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

        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Result'
            '</div>',
            unsafe_allow_html=True
        )

        if risk_category == "Low Risk":

            css_class = "risk-low"
            emoji = "🟢"

        elif risk_category == "Medium Risk":

            css_class = "risk-medium"
            emoji = "🟡"

        else:

            css_class = "risk-high"
            emoji = "🔴"


        st.markdown(
            f"""
            <div class="{css_class}">
                <div class="risk-score">
                    Predicted Risk Score: {risk_score:.4f}
                </div>
                <div class="risk-label">
                    {emoji} {risk_category}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # SCORE INTERPRETATION
        # ====================================================

        st.markdown("")

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Risk Score",
                f"{risk_score:.4f}"
            )


        with result_col2:

            st.metric(
                "Risk Category",
                risk_category
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown("---")

        st.markdown(
            '<div class="section-title">'
            '🧠 Explainable AI — SHAP'
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "SHAP shows which model features contributed "
            "most to this prediction. Positive values "
            "increase the predicted risk, while negative "
            "values decrease it."
        )


        # Calculate SHAP values
        shap_values = explainer.shap_values(
            processed_input
        )


        fig, contributions = create_shap_plot(
            shap_values,
            feature_names
        )


        st.pyplot(
            fig,
            use_container_width=True
        )


        # ====================================================
        # TOP CONTRIBUTING FEATURES
        # ====================================================

        st.markdown(
            "### Top Contributing Features"
        )

        display_table = contributions[
            ["Feature", "SHAP"]
        ].copy()

        display_table["Effect"] = display_table[
            "SHAP"
        ].apply(
            lambda x:
                "Increases Risk"
                if x > 0
                else "Decreases Risk"
                if x < 0
                else "Neutral"
        )

        display_table["SHAP"] = display_table[
            "SHAP"
        ].round(4)

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.markdown("---")

        st.markdown(
            "### Input Summary"
        )

        input_summary = pd.DataFrame({
            "Parameter": [
                "City",
                "Road Type",
                "Weather",
                "Visibility",
                "Traffic Density",
                "Date",
                "Time",
                "Day of Week",
                "Is Weekend",
                "Is Peak Hour"
            ],

            "Selected Value": [
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
            ]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown("---")

        st.caption(
            "Note: This system provides an estimated accident "
            "risk based on the conditions entered by the user. "
            "It does not guarantee that an accident will or "
            "will not occur."
        )


    except Exception as e:

        st.error(
            "Prediction failed. Please check your inputs."
        )

        st.exception(e)
