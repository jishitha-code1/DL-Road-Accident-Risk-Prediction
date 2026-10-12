import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image 

from pathlib import Path
from datetime import datetime, time 


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

# NEW BEST MODEL FROM COLAB
MODEL_PATH = MODEL_DIR / "improved_model (3).keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
DAMAGE_MODEL_PATH = MODEL_DIR / "best.pt"


@st.cache_resource
def load_damage_model():
    """Load the YOLO road-damage detector once per Streamlit session."""
    if not DAMAGE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Road-damage model not found: {DAMAGE_MODEL_PATH}. "
            "Place best.pt inside the project's models folder."
        )
    return YOLO(str(DAMAGE_MODEL_PATH))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* =========================================================
   MAIN PAGE
   ========================================================= */

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
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background-color: #EED9C4 !important;
}

[data-testid="stSidebar"] > div {
    background-color: #EED9C4 !important;
}

[data-testid="stSidebarContent"] {
    background-color: #EED9C4 !important;
}


/* =========================================================
   GENERAL TEXT
   ========================================================= */

h1,
h2,
h3,
h4 {
    color: #173F68 !important;
}

p,
label {
    color: #173F68;
}


/* =========================================================
   MAIN TITLE
   ========================================================= */

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #173F68;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 8px;
}

.main-subtitle {
    font-size: 18px;
    color: #315878;
    text-align: center;
    margin-bottom: 32px;
}


/* =========================================================
   SIDEBAR TITLE
   ========================================================= */

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
    line-height: 1.7;
    color: #173F68;
    margin-bottom: 25px;
}


/* =========================================================
   DIVIDER
   ========================================================= */

.custom-divider {
    height: 1px;
    background-color: rgba(23, 63, 104, 0.22);
    margin: 30px 0;
}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #173F68;
    margin-top: 28px;
    margin-bottom: 24px;
}


/* =========================================================
   CARDS
   ========================================================= */

.info-card {
    background-color: #F7FBFD;
    border: 1px solid #D3E2EA;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 7px 20px rgba(30, 70, 100, 0.10);
    margin-bottom: 25px;
}

.summary-card {
    background-color: #F7FBFD;
    border: 1px solid #D3E2EA;
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 18px;
    margin-bottom: 25px;
    box-shadow: 0 5px 16px rgba(30, 70, 100, 0.08);
}


/* =========================================================
   PREDICT BUTTON
   ========================================================= */

div.stButton > button {
    width: 100%;
    height: 58px;

    background-color: #EED9C4 !important;
    color: #173F68 !important;

    border: 1px solid #F09AAA !important;
    border-radius: 15px !important;

    font-size: 17px !important;
    font-weight: 700 !important;

    box-shadow: 0 7px 16px rgba(80, 50, 60, 0.16);

    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: #F4B0C0 !important;
    color: #173F68 !important;

    border-color: #E996A8 !important;

    transform: translateY(-1px);
}


/* =========================================================
   INPUTS
   ========================================================= */

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

[data-testid="stDateInput"] > div {
    background-color: #F8FAFC !important;
    border-radius: 10px !important;
}

[data-testid="stTimeInput"] > div {
    background-color: #F8FAFC !important;
    border-radius: 10px !important;
}


/* =========================================================
   RESULT CARD
   ========================================================= */

.result-card {
    background-color: #F7FBFD;
    border: 1px solid #D3E2EA;
    border-radius: 22px;
    padding: 45px 30px;
    text-align: center;
    box-shadow: 0 10px 28px rgba(30, 70, 100, 0.12);
    margin-bottom: 30px;
}

.result-title {
    font-size: 20px;
    font-weight: 700;
    color: #315878;
    margin-bottom: 10px;
}

.score {
    font-size: 50px;
    font-weight: 800;
    color: #173F68;
    margin: 5px 0 18px 0;
}


/* =========================================================
   RISK LABELS
   ========================================================= */

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


/* =========================================================
   SIDEBAR RISK ITEMS
   ========================================================= */

.risk-item {
    margin-top: 24px;
    margin-bottom: 22px;
}

.risk-name {
    font-size: 17px;
    font-weight: 700;
    color: #173F68;
    margin-bottom: 5px;
}

.risk-score {
    font-size: 15px;
    color: #315878;
}

.low-dot {
    color: #43D17F;
    font-size: 22px;
}

.medium-dot {
    color: #F4C542;
    font-size: 22px;
}

.high-dot {
    color: #EF5B67;
    font-size: 22px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer-text {
    text-align: center;
    color: #315878;
    font-size: 14px;
    margin-top: 35px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# These must match the exact feature order used during Colab training.
MODEL_FEATURES = [
    "visibility",
    "traffic_density",
    "is_peak_hour",
    "weather",
    "hour"
]


# ============================================================
# MODEL INFORMATION
# ============================================================

BEST_MODEL_NAME = "Deep Learning DNN (5 input features)"
BEST_MODEL_PARAMETERS = 1057
BEST_MODEL_MAE = 0.0504
BEST_MODEL_RMSE = 0.0720
BEST_MODEL_R2 = 0.8884




# ============================================================
# ============================================================
# LOAD MODEL AND PREPROCESSING
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor file not found:\n{PREPROCESSOR_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    expected_features = int(model.input_shape[-1])
    if expected_features != 11:
        raise ValueError(
            f"The loaded model expects {expected_features} features, but this "
            "app's saved preprocessor should produce 11. Check that the model "
            "and preprocessor are from the same Colab training run."
        )

    return model, preprocessor


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

    if 7 <= hour <= 10:
        return 1

    if 17 <= hour <= 20:
        return 1

    return 0


# ============================================================
# ============================================================
# PREPROCESS INPUT
# ============================================================

def preprocess_input(weather, visibility, traffic_density, selected_time,
                     is_peak_hour, preprocessor):
    """Transform the five features in the exact training order."""
    input_data = pd.DataFrame([{
        "visibility": str(visibility),
        "traffic_density": str(traffic_density),
        "is_peak_hour": int(is_peak_hour),
        "weather": str(weather),
        "hour": int(selected_time.hour)
    }], columns=MODEL_FEATURES)

    processed_input = preprocessor.transform(input_data)
    if hasattr(processed_input, "toarray"):
        processed_input = processed_input.toarray()
    processed_input = np.asarray(processed_input, dtype=np.float32)

    expected_features = 11
    if processed_input.shape[1] != expected_features:
        raise ValueError(
            f"Preprocessor produced {processed_input.shape[1]} features; "
            f"the saved model expects {expected_features}. Ensure the model "
            "and preprocessor files are a matching pair."
        )
    return processed_input


# MODEL PREDICTION
# ============================================================

def predict_risk(
    model,
    processed_input
):

    prediction = model.predict(
        processed_input,
        verbose=0
    )

    raw_score = float(
        prediction[0][0]
    )

    # The model was trained with a sigmoid output, which returns 0-1.

    risk_score = float(
        np.clip(
            raw_score,
            0.0,
            1.0
        )
    )

    return risk_score


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(model):

    try:

        background = np.zeros(
            (1, int(model.input_shape[-1])),
            dtype=np.float32
        )

        explainer = shap.DeepExplainer(
            model,
            background
        )

        return explainer

    except Exception:

        return None


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

        if isinstance(
            shap_values,
            list
        ):
            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        )

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
# ============================================================
# FEATURE NAMES
# ============================================================

def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback names for the known 11-column transformed pipeline.
        return [f"processed_feature_{i + 1}" for i in range(11)]


# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🚦 Road Accident Risk</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Deep Learning + Explainable AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-description">'
        'This application estimates road accident risk based on '
        'the conditions entered by the user.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Risk Levels"
    )

    st.markdown(
        '<div class="risk-item">'
        '<div class="risk-name">'
        '<span class="low-dot">●</span>&nbsp; Low Risk'
        '</div>'
        '<div class="risk-score">Score &lt; 0.25</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="risk-item">'
        '<div class="risk-name">'
        '<span class="medium-dot">●</span>&nbsp; Medium Risk'
        '</div>'
        '<div class="risk-score">Score 0.25 – &lt; 0.60</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="risk-item">'
        '<div class="risk-name">'
        '<span class="high-dot">●</span>&nbsp; High Risk'
        '</div>'
        '<div class="risk-score">Score ≥ 0.60</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="custom-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="font-size:14px;color:#315878;margin-top:20px;">'
        'DL-Based Road Accident Risk Prediction'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🚦 Road Accident Risk Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Deep Learning based risk estimation with Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, preprocessor = load_model_and_preprocessing()

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
        "City (context only; not used by model)",
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
        "Road Type (context only; not used by model)",
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

    selected_date = st.date_input(
        "Date",
        value=datetime.now().date()
    )


# ============================================================
# DAY OF WEEK
# ============================================================

selected_date_timestamp = pd.to_datetime(
    selected_date
)

day_of_week = (
    selected_date_timestamp.strftime("%A")
)


# ============================================================
# AUTOMATIC WEEKEND
# ============================================================

is_weekend = (
    1
    if selected_date_timestamp.weekday() >= 5
    else 0
)


# ============================================================
# ROW 3
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

is_peak_hour = calculate_peak_hour(
    selected_time.hour
)


# ============================================================
# SUMMARY TEXT
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

summary_text = (
    f"📅 {selected_date.strftime('%Y-%m-%d')} "
    f"&nbsp; | &nbsp; "
    f"📅 {day_of_week} "
    f"&nbsp; | &nbsp; "
    f"🕐 {selected_time.strftime('%H:%M')} "
    f"&nbsp; | &nbsp; "
    f"Weekend: <b>{weekend_text}</b> "
    f"&nbsp; | &nbsp; "
    f"Peak Hour: <b>{peak_text}</b>"
)

st.markdown(
    f'<div class="summary-card">{summary_text}</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🚗 Predict Accident Risk",
    width="stretch"
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
            weather=weather,
            visibility=visibility,
            traffic_density=traffic_density,
            selected_time=selected_time,
            is_peak_hour=is_peak_hour,
            preprocessor=preprocessor
        )


        # ----------------------------------------------------
        # PREDICT
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
        # RESULT
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )


        if risk_category == "Low Risk":

            category_class = "low-risk"

        elif risk_category == "Medium Risk":

            category_class = "medium-risk"

        else:

            category_class = "high-risk"


        result_html = (
            f'<div class="result-card">'
            f'<div class="result-title">'
            f'Predicted Risk Score'
            f'</div>'
            f'<div class="score">'
            f'{risk_score:.4f}'
            f'</div>'
            f'<div class="{category_class}">'
            f'{risk_category}'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            result_html,
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

            left_summary = (
                f'<div class="info-card">'
                f'<b>City:</b> {city}<br><br>'
                f'<b>Road Type:</b> {road_type}<br><br>'
                f'<b>Weather:</b> {weather}<br><br>'
                f'<b>Visibility:</b> {visibility}<br><br>'
                f'<b>Traffic Density:</b> {traffic_density}'
                f'</div>'
            )

            st.markdown(
                left_summary,
                unsafe_allow_html=True
            )


        with summary_col2:

            right_summary = (
                f'<div class="info-card">'
                f'<b>Date:</b> '
                f'{selected_date.strftime("%Y-%m-%d")}<br><br>'
                f'<b>Day:</b> {day_of_week}<br><br>'
                f'<b>Time:</b> '
                f'{selected_time.strftime("%H:%M")}<br><br>'
                f'<b>Weekend:</b> {weekend_text}<br><br>'
                f'<b>Peak Hour:</b> {peak_text}'
                f'</div>'
            )

            st.markdown(
                right_summary,
                unsafe_allow_html=True
            )


        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.markdown(
            '<div class="section-title">Explainable AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="info-card">'
            'SHAP explains how the input features contributed '
            'to the predicted risk score.'
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

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

            feature_names = get_feature_names(preprocessor)

            values = shap_values[0]

            if len(feature_names) == len(values):

                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "SHAP Value": values,
                        "Absolute Impact": np.abs(values)
                    }
                )

                shap_df = shap_df.sort_values(
                    "Absolute Impact",
                    ascending=False
                )


                top_features = (
                    shap_df
                    .head(10)
                    .sort_values(
                        "Absolute Impact",
                        ascending=True
                    )
                )


                # ------------------------------------------------
                # SHAP CHART
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
                    width="stretch"
                )

                plt.close(fig)


                # ------------------------------------------------
                # SHAP TABLE
                # ------------------------------------------------

                st.markdown(
                    '<div class="info-card">'
                    '<b>Feature Contribution Details</b>'
                    '</div>',
                    unsafe_allow_html=True
                )

                display_df = (
                    shap_df
                    .head(10)
                    [["Feature", "SHAP Value"]]
                    .copy()
                )

                display_df["SHAP Value"] = (
                    display_df["SHAP Value"]
                    .round(4)
                )

                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.warning(
                    "SHAP feature names do not match "
                    "the model input size."
                )

        else:

            st.info(
                "Prediction was generated successfully, "
                "but SHAP explanation could not be generated."
            )


        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        st.markdown(
            '<div class="section-title">Model Performance</div>',
            unsafe_allow_html=True
        )

        performance_col1, performance_col2, performance_col3, performance_col4 = st.columns(4)


        with performance_col1:

            st.metric(
                "MAE",
                f"{BEST_MODEL_MAE:.4f}"
            )


        with performance_col2:

            st.metric(
                "RMSE",
                f"{BEST_MODEL_RMSE:.4f}"
            )


        with performance_col3:

            st.metric(
                "R² Score",
                f"{BEST_MODEL_R2:.4f}"
            )


        with performance_col4:

            st.metric(
                "Parameters",
                f"{BEST_MODEL_PARAMETERS:,}"
            )


        st.markdown(
            f'<div class="info-card">'
            f'<b>Selected Model:</b> {BEST_MODEL_NAME}<br><br>'
            f'<b>Architecture:</b> '
            f'11 → 32 → 16 → 8 → 1<br><br>'
            f'<b>Optimizer:</b> Adam<br>'
            f'<b>Loss:</b> Mean Squared Error (MSE)<br>'
            f'<b>Output activation:</b> Sigmoid'
            f'</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        with st.expander(
            "Model and preprocessing information"
        ):

            st.write(
                "Selected model:",
                BEST_MODEL_NAME
            )

            st.write(
                "Model parameters:",
                f"{BEST_MODEL_PARAMETERS:,}"
            )

            st.write(
                "Processed feature count:",
                processed_input.shape[1]
            )

            st.write(
                "Raw model input features:",
                ", ".join(MODEL_FEATURES)
            )

            st.write(
                "Preprocessing pipeline:",
                "One-hot encoding + standard scaling"
            )

            st.write(
                "Total model input features:",
                processed_input.shape[1]
            )

            st.write(
                "Prediction method:",
                "Deep Learning model.predict()"
            )

            st.write(
                "Explainability method:",
                "SHAP DeepExplainer"
            )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            '<div class="info-card">'
            '<b>Note:</b> This application estimates accident '
            'risk based on the provided road, traffic, weather, '
            'visibility and time-related conditions. '
            'The prediction is an estimate and does not guarantee '
            'that an accident will or will not occur.'
            '</div>',
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
# ROAD DAMAGE DETECTION (IMAGE UPLOAD)
# ============================================================

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Road Damage Detection from Image</div>',
    unsafe_allow_html=True
)
st.write(
    "Upload a road image to detect visible road-damage patterns. "
    "This is a separate feature from the parameter-based accident-risk prediction above."
)

uploaded_road_image = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
    key="road_damage_image_uploader"
)

if uploaded_road_image is not None:
    try:
        road_image = Image.open(uploaded_road_image).convert("RGB")
        st.image(
            road_image,
            caption="Uploaded road image",
            use_container_width=True
        )

        damage_confidence = st.slider(
            "Detection confidence threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help=(
                "Lower thresholds show more candidate detections but may add false positives. "
                "This setting does not guarantee a particular precision."
            ),
            key="road_damage_confidence"
        )

        if st.button("Detect Road Damage", key="detect_road_damage_button"):
            with st.spinner("Analyzing the road image..."):
                damage_model = load_damage_model()
                detection_result = damage_model.predict(
                    source=road_image,
                    conf=damage_confidence,
                    verbose=False
                )[0]

            # Ultralytics plot() returns BGR; convert it to RGB for Streamlit.
            annotated_bgr = detection_result.plot()
            annotated_rgb = annotated_bgr[:, :, ::-1]

            st.subheader("Detection Results")
            st.image(
                annotated_rgb,
                caption="Road-damage detections",
                use_container_width=True
            )

            if len(detection_result.boxes) == 0:
                st.info(
                    "No damage was detected at this confidence threshold. "
                    "This does not prove that the road has no damage. "
                    "You can try a lower threshold to inspect candidate detections."
                )
            else:
                st.success(
                    f"Found {len(detection_result.boxes)} possible road-damage instance(s)."
                )

                detection_rows = []
                for box in detection_result.boxes:
                    class_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    detection_rows.append({
                        "Detected class": damage_model.names[class_id],
                        "Confidence": f"{score:.1%}"
                    })

                st.dataframe(
                    pd.DataFrame(detection_rows),
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Model confidence is not the probability that an accident will occur. "
                "Image-based detections do not directly calculate the accident-risk score."
            )

    except Exception as e:
        st.error("Road-damage detection could not be completed.")
        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-text">'
    'DL-Based Road Accident Risk Prediction '
    '&nbsp; | &nbsp; '
    'Deep Learning + Explainable AI'
    '</div>',
    unsafe_allow_html=True
) 
