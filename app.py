import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f4f7fb;
    }

    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Main title */
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #17365d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #5f7185;
        margin-bottom: 25px;
    }

    /* Section headings */
    .section-title {
        font-size: 23px;
        font-weight: 650;
        color: #17365d;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Cards */
    .info-card {
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #dce5ef;
        box-shadow: 0 3px 12px rgba(30, 60, 90, 0.07);
        margin-bottom: 18px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        border-radius: 16px;
        padding: 25px;
        border: 1px solid #dce5ef;
        box-shadow: 0 4px 15px rgba(30, 60, 90, 0.08);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .prediction-score {
        font-size: 42px;
        font-weight: 750;
        color: #17365d;
    }

    .prediction-category {
        font-size: 25px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 48px;
        font-size: 17px;
        font-weight: 600;
        border: none;
        background: #2f75b5;
        color: white;
    }

    .stButton > button:hover {
        background: #245d91;
        color: white;
    }

    /* Input labels */
    label {
        font-weight: 600 !important;
        color: #29445f !important;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        color: #29445f !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #edf3f9;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "improved_model.keras"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = load_model(MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    scaler = joblib.load(SCALER_PATH)

    return model, encoder, scaler


model, encoder, scaler = load_artifacts()


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

    # Morning peak: 07:00 - 09:59
    # Evening peak: 17:00 - 19:59

    if (7 <= hour < 10) or (17 <= hour < 20):
        return 1

    return 0


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource
def create_shap_explainer(model):

    background = np.zeros(
        (1, 35),
        dtype=np.float32
    )

    explainer = shap.DeepExplainer(
        model,
        background
    )

    return explainer


# ============================================================
# GET FEATURE NAMES
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
# PREPROCESS INPUT
# ============================================================

def preprocess_input(
    city,
    road_type,
    weather,
    visibility,
    traffic_density,
    selected_date_obj,
    selected_time_obj,
    is_weekend,
    is_peak_hour
):

    # --------------------------------------------------------
    # Date / Time values
    # --------------------------------------------------------

    year = selected_date_obj.year
    month = selected_date_obj.month
    date_day = selected_date_obj.day

    hour = selected_time_obj.hour
    minute = selected_time_obj.minute

    day_of_week = selected_date_obj.strftime("%A")


    # --------------------------------------------------------
    # Categorical data
    # --------------------------------------------------------

    categorical_data = pd.DataFrame([{

        "city": city,

        "road_type": road_type,

        "weather": weather,

        "visibility": visibility,

        "traffic_density": traffic_density,

        "day_of_week": day_of_week

    }])


    # --------------------------------------------------------
    # Encode categorical features
    # --------------------------------------------------------

    encoded_data = encoder.transform(
        categorical_data
    )

    if hasattr(encoded_data, "toarray"):

        encoded_data = encoded_data.toarray()


    encoded_data = np.asarray(
        encoded_data,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Numerical data
    # --------------------------------------------------------

    numerical_data = pd.DataFrame([{

        "hour": hour,

        # IMPORTANT:
        # These are the values selected by the user.
        "is_weekend": is_weekend,

        "is_peak_hour": is_peak_hour,

        "year": year,

        "month": month,

        "date_day": date_day,

        "time_hour": hour,

        "time_minute": minute

    }])


    # --------------------------------------------------------
    # Scale numerical features
    # --------------------------------------------------------

    scaled_data = scaler.transform(
        numerical_data
    )

    scaled_data = np.asarray(
        scaled_data,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    final_data = np.hstack([
        encoded_data,
        scaled_data
    ])


    final_data = np.asarray(
        final_data,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Check feature count
    # --------------------------------------------------------

    if final_data.shape[1] != 35:

        raise ValueError(
            f"Expected 35 features but got "
            f"{final_data.shape[1]}"
        )


    return final_data


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🚦 Road Accident Risk"
    )

    st.markdown(
        """
        **Deep Learning + Explainable AI**

        This application estimates road
        accident risk based on the conditions
        entered by the user.
        """
    )

    st.markdown("---")

    st.markdown("### Risk Levels")

    st.markdown(
        """
        🟢 **Low Risk**  
        Score < 0.25

        🟡 **Medium Risk**  
        Score 0.25 – < 0.60

        🔴 **High Risk**  
        Score ≥ 0.60
        """
    )

    st.markdown("---")

    st.caption(
        "DL-Based Road Accident Risk Prediction"
    )


# ============================================================
# HEADER
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
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Enter Road Conditions</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# INPUT ROW 1
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# INPUT ROW 2
# ------------------------------------------------------------

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
# CURRENT DATE AND TIME
# ============================================================

now = datetime.now()


with col6:

    selected_date_obj = st.date_input(
        "Date",
        value=now.date()
    )


# ------------------------------------------------------------
# TIME
# ------------------------------------------------------------

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
# AUTOMATIC VALUES
# ============================================================

selected_date = selected_date_obj.strftime(
    "%Y-%m-%d"
)

selected_time = selected_time_obj.strftime(
    "%H:%M"
)


# ------------------------------------------------------------
# Day of week automatically from selected date
# ------------------------------------------------------------

day_of_week = selected_date_obj.strftime(
    "%A"
)


# ------------------------------------------------------------
# Automatic defaults
# ------------------------------------------------------------

automatic_weekend = (
    1
    if selected_date_obj.weekday() >= 5
    else 0
)


automatic_peak_hour = calculate_peak_hour(
    selected_time_obj.hour
)


# ============================================================
# USER-CONTROLLABLE WEEKEND / PEAK HOUR
# ============================================================

with col8:

    is_weekend = st.radio(
        "Is Weekend",
        options=[0, 1],

        index=automatic_weekend,

        format_func=lambda x:
            "Yes" if x == 1 else "No",

        horizontal=True
    )


with col9:

    is_peak_hour = st.radio(
        "Is Peak Hour",
        options=[0, 1],

        index=automatic_peak_hour,

        format_func=lambda x:
            "Yes" if x == 1 else "No",

        horizontal=True
    )


# ============================================================
# AUTOMATIC INFORMATION
# ============================================================

st.markdown(
    f"""
    <div class="info-card">

    <b>Selected Date:</b> {selected_date}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>Day:</b> {day_of_week}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>Selected Time:</b> {selected_time}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>Weekend:</b> {"Yes" if is_weekend == 1 else "No"}
    &nbsp;&nbsp; | &nbsp;&nbsp;

    <b>Peak Hour:</b> {"Yes" if is_peak_hour == 1 else "No"}

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")


predict_button = st.button(
    "Predict Accident Risk"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # Prepare model input
        # ----------------------------------------------------

        X_input = preprocess_input(

            city=city,

            road_type=road_type,

            weather=weather,

            visibility=visibility,

            traffic_density=traffic_density,

            selected_date_obj=selected_date_obj,

            selected_time_obj=selected_time_obj,

            is_weekend=is_weekend,

            is_peak_hour=is_peak_hour

        )


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model.predict(
            X_input,
            verbose=0
        )


        score = float(
            np.asarray(prediction).flatten()[0]
        )


        # Keep score inside 0-1
        score = float(
            np.clip(score, 0, 1)
        )


        category = get_risk_category(
            score
        )


        # ----------------------------------------------------
        # Prediction display
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="prediction-card">

                <div style="
                    font-size:18px;
                    color:#5f7185;
                ">
                    Predicted Risk Score
                </div>

                <div class="prediction-score">
                    {score:.4f}
                </div>

                <div class="prediction-category">
                    {category}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # Progress indicator
        # ----------------------------------------------------

        st.progress(
            score
        )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Why did the model make this prediction?'
            '</div>',
            unsafe_allow_html=True
        )


        try:

            explainer = create_shap_explainer(
                model
            )


            shap_values = explainer.shap_values(
                X_input
            )


            # Handle SHAP output shapes
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


            shap_values = shap_values.reshape(
                X_input.shape[1]
            )


            # ------------------------------------------------
            # Top 10 SHAP features
            # ------------------------------------------------

            shap_df = pd.DataFrame({

                "Feature": feature_names,

                "SHAP Value": shap_values

            })


            shap_df["Absolute Impact"] = (
                shap_df["SHAP Value"].abs()
            )


            shap_df = shap_df.sort_values(
                "Absolute Impact",
                ascending=False
            ).head(10)


            # ------------------------------------------------
            # SHAP chart
            # ------------------------------------------------

            chart_df = shap_df.sort_values(
                "SHAP Value"
            )


            fig, ax = plt.subplots(
                figsize=(9, 5)
            )


            ax.barh(
                chart_df["Feature"],
                chart_df["SHAP Value"]
            )


            ax.set_xlabel(
                "SHAP Value"
            )

            ax.set_ylabel(
                "Feature"
            )

            ax.set_title(
                "Top Factors Influencing the Prediction"
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

            st.markdown(
                "**Top influencing features**"
            )


            display_df = shap_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ].copy()


            display_df["SHAP Value"] = (
                display_df["SHAP Value"]
                .round(5)
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        except Exception as shap_error:

            st.warning(
                "SHAP explanation could not be generated "
                "for this prediction."
            )

            st.caption(
                f"SHAP message: {shap_error}"
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Current Input Summary'
            '</div>',
            unsafe_allow_html=True
        )


        summary_df = pd.DataFrame({

            "Feature": [

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

                selected_date,

                selected_time,

                day_of_week,

                "Yes" if is_weekend == 1 else "No",

                "Yes" if is_peak_hour == 1 else "No"

            ]

        })


        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.markdown(
            """
            <div class="info-card">

            <b>Note:</b> This system provides an estimated
            accident risk score based on the conditions entered
            by the user. It does not guarantee that an accident
            will or will not occur.

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

        st.exception(e)
