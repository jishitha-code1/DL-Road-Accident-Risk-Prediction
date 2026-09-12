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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #eef6ff 0%, #f8fbff 45%, #eef9f5 100%);
}

.main {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #12355b;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #52708d;
    margin-bottom: 25px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 750;
    color: #164e63;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Cards */
.info-card {
    background: rgba(255, 255, 255, 0.92);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #d9e7f3;
    box-shadow: 0 5px 18px rgba(38, 78, 112, 0.08);
    margin-bottom: 18px;
}

/* Prediction card */
.prediction-card {
    background: linear-gradient(135deg, #ffffff, #f1f8ff);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #cfe1f1;
    box-shadow: 0 8px 25px rgba(28, 78, 121, 0.12);
    text-align: center;
    margin-top: 20px;
}

.risk-score {
    font-size: 42px;
    font-weight: 800;
    color: #155e75;
}

.risk-label {
    font-size: 28px;
    font-weight: 750;
    color: #256d5a;
    margin-top: 5px;
}

/* Button */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 12px;
    font-size: 17px;
    font-weight: 700;
    background: linear-gradient(90deg, #1677c8, #159a8c);
    color: white;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1265aa, #128477);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eaf4ff, #eefaf7);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #164e63;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "improved_model.keras"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, encoder, scaler


try:

    model, encoder, scaler = load_artifacts()

except Exception as e:

    st.error("Unable to load the trained model or preprocessing files.")

    st.code(str(e))

    st.stop()


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
# FEATURE NAMES
# ============================================================

try:

    encoded_feature_names = list(
        encoder.get_feature_names_out(categorical_features)
    )

except Exception:

    encoded_feature_names = [
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


feature_names = encoded_feature_names + numerical_features


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
# AUTOMATIC PEAK-HOUR CALCULATION
# ============================================================

def calculate_peak_hour(hour):

    # Morning peak: 07:00 - 10:00
    # Evening peak: 17:00 - 20:00

    if 7 <= hour < 10 or 17 <= hour < 20:
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
    selected_date,
    selected_time,
    day_of_week,
    is_weekend,
    is_peak_hour
):

    # --------------------------------------------------------
    # Parse date
    # --------------------------------------------------------

    date_value = pd.to_datetime(
        selected_date,
        format="%Y-%m-%d",
        errors="coerce"
    )

    if pd.isna(date_value):

        raise ValueError(
            "Invalid date. Please use YYYY-MM-DD format."
        )

    # --------------------------------------------------------
    # Parse time
    # --------------------------------------------------------

    time_value = pd.to_datetime(
        selected_time,
        format="%H:%M",
        errors="coerce"
    )

    if pd.isna(time_value):

        raise ValueError(
            "Invalid time. Please use HH:MM format."
        )

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_df = pd.DataFrame({

        "city": [city],

        "road_type": [road_type],

        "weather": [weather],

        "visibility": [visibility],

        "traffic_density": [traffic_density],

        "day_of_week": [day_of_week],

        "hour": [time_value.hour],

        "is_weekend": [is_weekend],

        "is_peak_hour": [is_peak_hour],

        "year": [date_value.year],

        "month": [date_value.month],

        "date_day": [date_value.day],

        "time_hour": [time_value.hour],

        "time_minute": [time_value.minute]
    })


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    encoded = encoder.transform(
        input_df[categorical_features]
    )

    if hasattr(encoded, "toarray"):

        encoded = encoded.toarray()


    # --------------------------------------------------------
    # Scale numerical features
    # --------------------------------------------------------

    scaled = scaler.transform(
        input_df[numerical_features]
    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    processed = np.hstack([
        encoded,
        scaled
    ])


    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    if processed.shape[1] != 35:

        raise ValueError(
            f"Expected 35 features but received "
            f"{processed.shape[1]} features."
        )


    return processed.astype(np.float32)


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(_model):

    background = np.zeros(
        (1, 35),
        dtype=np.float32
    )

    explainer = shap.DeepExplainer(
        _model,
        background
    )

    return explainer


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🚦 Risk Prediction"
    )

    st.markdown(
        "Enter the road and environmental conditions "
        "to estimate accident risk."
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.write("**Model:** Deep Neural Network")

    st.write("**Task:** Risk Score Regression")

    st.write("**Input Features:** 35")

    st.write("**Explainability:** SHAP")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 Road Accident Risk Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning based risk estimation with Explainable AI'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CURRENT DATE AND TIME
# ============================================================

now = datetime.now()

current_date = now.strftime("%Y-%m-%d")

current_time = now.strftime("%H:%M")

current_day = now.strftime("%A")


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Road & Environmental Conditions</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)


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


# ============================================================
# CHANGED ONLY: EDITABLE DATE
# ============================================================

with col6:

    selected_date_obj = st.date_input(
        "Date",
        value=now.date()
    )

    selected_date = selected_date_obj.strftime("%Y-%m-%d")


# ============================================================
# CHANGED ONLY: EDITABLE TIME
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

    selected_time = selected_time_obj.strftime("%H:%M")


# ============================================================
# CHANGED ONLY: DAY OF WEEK FROM SELECTED DATE
# ============================================================

with col8:

    selected_date_obj = pd.to_datetime(
        selected_date
    )

    day_of_week = selected_date_obj.strftime("%A")

    st.markdown("**Day of Week**")

    st.info(
        f"📆 {day_of_week}",
        icon="📆"
    )


# ============================================================
# CHANGED ONLY: WEEKEND FROM SELECTED DATE
# ============================================================

with col9:

    automatic_weekend = (
        1 if selected_date_obj.weekday() >= 5 else 0
    )

    weekend_text = (
        "Yes"
        if automatic_weekend == 1
        else "No"
    )

    st.markdown("**Weekend**")

    st.info(
        f"Weekend: {weekend_text}",
        icon="🗓️"
    )


# ============================================================
# AUTOMATIC PEAK HOUR
# ============================================================

automatic_peak_hour = calculate_peak_hour(
    selected_time_obj.hour
)

peak_text = (
    "Yes"
    if automatic_peak_hour == 1
    else "No"
)


st.markdown(
    f"""
    <div style="
        background: #f0f8ff;
        padding: 12px 16px;
        border-radius: 12px;
        margin-top: 10px;
        border: 1px solid #d5e8f5;
        color: #164e63;
        font-size: 15px;
    ">
        🕒 <b>Peak Hour:</b> {peak_text}
        &nbsp;&nbsp; | &nbsp;&nbsp;
        Automatic based on the selected/current time
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# CURRENT INPUT SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">Current Input</div>',
    unsafe_allow_html=True
)

summary_col1, summary_col2 = st.columns(2)


with summary_col1:

    st.write(f"**City:** {city}")

    st.write(f"**Road Type:** {road_type}")

    st.write(f"**Weather:** {weather}")

    st.write(f"**Visibility:** {visibility}")


with summary_col2:

    st.write(f"**Traffic Density:** {traffic_density}")

    st.write(f"**Date:** {selected_date}")

    st.write(f"**Time:** {selected_time}")

    st.write(f"**Day:** {day_of_week}")


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Accident Risk"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # CHANGED ONLY: USE SELECTED DATE/TIME VALUES
        # ----------------------------------------------------

        is_weekend = (
            1 if selected_date_obj.weekday() >= 5
            else 0
        )

        is_peak_hour = calculate_peak_hour(
            selected_time_obj.hour
        )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed_input = preprocess_input(

            city,

            road_type,

            weather,

            visibility,

            traffic_density,

            selected_date,

            selected_time,

            day_of_week,

            is_weekend,

            is_peak_hour
        )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            processed_input,
            verbose=0
        )


        score = float(
            np.asarray(prediction).reshape(-1)[0]
        )


        # Keep score inside 0-1
        score = float(
            np.clip(score, 0.0, 1.0)
        )


        risk_category = get_risk_category(score)


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### Estimated Risk Score"
        )

        st.markdown(
            f'<div class="risk-score">{score:.4f}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="risk-label">{risk_category}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Risk explanation
        # ----------------------------------------------------

        if risk_category == "Low Risk":

            st.success(
                "The entered conditions result in a relatively low estimated risk."
            )

        elif risk_category == "Medium Risk":

            st.warning(
                "The entered conditions result in a moderate estimated risk."
            )

        else:

            st.error(
                "The entered conditions result in a relatively high estimated risk."
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-title">Explainable AI — SHAP</div>',
            unsafe_allow_html=True
        )

        try:

            explainer = create_shap_explainer(model)

            shap_values = explainer.shap_values(
                processed_input
            )


            # Handle SHAP output shape
            if isinstance(shap_values, list):

                shap_array = np.asarray(
                    shap_values[0]
                )

            else:

                shap_array = np.asarray(
                    shap_values
                )


            shap_array = np.squeeze(
                shap_array
            )


            if shap_array.ndim == 1:

                shap_array = shap_array.reshape(1, -1)


            shap_row = shap_array[0]


            # ------------------------------------------------
            # Top 10 features
            # ------------------------------------------------

            importance = np.abs(shap_row)

            top_indices = np.argsort(
                importance
            )[::-1][:10]


            top_features = [
                feature_names[i]
                for i in top_indices
            ]

            top_values = [
                shap_row[i]
                for i in top_indices
            ]


            shap_df = pd.DataFrame({

                "Feature": top_features,

                "SHAP Value": top_values,

                "Impact": [
                    "Increases risk"
                    if value > 0
                    else "Decreases risk"
                    for value in top_values
                ]
            })


            # ------------------------------------------------
            # SHAP chart
            # ------------------------------------------------

            fig, ax = plt.subplots(
                figsize=(9, 5)
            )

            y_positions = np.arange(
                len(top_features)
            )

            ax.barh(
                y_positions,
                top_values
            )

            ax.set_yticks(
                y_positions
            )

            ax.set_yticklabels(
                top_features
            )

            ax.invert_yaxis()

            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_title(
                "Top Factors Influencing the Prediction"
            )

            ax.axvline(
                0,
                linewidth=1
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)


            # ------------------------------------------------
            # SHAP table
            # ------------------------------------------------

            st.dataframe(
                shap_df,
                use_container_width=True,
                hide_index=True
            )


        except Exception as shap_error:

            st.warning(
                "Prediction was successful, but the SHAP explanation "
                "could not be generated."
            )

            st.caption(
                str(shap_error)
            )


        # ====================================================
        # INPUT SUMMARY AFTER PREDICTION
        # ====================================================

        st.markdown(
            '<div class="section-title">Prediction Input Summary</div>',
            unsafe_allow_html=True
        )

        final_summary = pd.DataFrame({

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

            "Value": [

                city,
                road_type,
                weather,
                visibility,
                traffic_density,
                selected_date,
                selected_time,
                day_of_week,
                "Yes" if is_weekend == 1 else "No",
                "Yes" if is_peak_hour == 1 else "No"
            ]
        })


        st.dataframe(
            final_summary,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.info(
            "⚠️ This system provides an estimated accident risk "
            "based on the entered road, traffic, weather and "
            "time-related conditions. It does not guarantee that "
            "an accident will or will not occur."
        )


    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.code(
            str(e)
        )
