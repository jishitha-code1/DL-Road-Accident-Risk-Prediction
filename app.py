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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
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
        background-color: #ADD8E6 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #ADD8E6 !important;
    }

    [data-testid="stHeader"] {
        background-color: #ADD8E6 !important;
    }

    .main {
        background-color: #ADD8E6 !important;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    /* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #FFC0CB !important;
}

[data-testid="stSidebar"] > div {
    background-color: #FFC0CB !important;
}

[data-testid="stSidebarContent"] {
    background-color: #FFC0CB !important;
}


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1, h2, h3, h4 {
        color: #173F68 !important;
    }


    /* ======================================================
       NORMAL TEXT
       ====================================================== */

    p {
        color: #315878;
    }


    /* ======================================================
       INPUT CARD
       ====================================================== */

    .input-card {
        background-color: #FFFDF9;
        border: 1px solid #D5E1E8;
        border-radius: 18px;
        padding: 28px;
        margin-top: 15px;
        margin-bottom: 25px;
        box-shadow: 0 7px 22px rgba(35, 75, 100, 0.10);
    }


    /* ======================================================
       SELECT BOXES
       ====================================================== */

    div[data-baseweb="select"] > div {
        background-color: #FFFDF9 !important;
        border: 1px solid #C8D9E4 !important;
        border-radius: 10px !important;
    }


    /* ======================================================
       DATE AND TIME
       ====================================================== */

    div[data-baseweb="input"] > div {
        background-color: #FFFDF9 !important;
        border: 1px solid #C8D9E4 !important;
        border-radius: 10px !important;
    }

    input {
        color: #173F68 !important;
        background-color: #FFFDF9 !important;
    }


    /* ======================================================
       RADIO BUTTON
       ====================================================== */

    div[data-testid="stRadio"] label {
        color: #315878 !important;
    }


    /* ======================================================
       PREDICT BUTTON
       ====================================================== */

    div.stButton > button {

        width: 100%;
        min-height: 54px;

        background-color: #FFC0CB !important;
        color: #173F68 !important;

        border: 1px solid #E39AA9 !important;
        border-radius: 12px !important;

        font-size: 17px !important;
        font-weight: 750 !important;

        box-shadow:
            0 6px 15px rgba(145, 80, 95, 0.18);

        transition: all 0.2s ease;
    }


    div.stButton > button:hover {

        background-color: #F5A9B8 !important;
        color: #173F68 !important;

        border: 1px solid #D88C9B !important;

        transform: translateY(-1px);

        box-shadow:
            0 8px 20px rgba(145, 80, 95, 0.25);
    }


    div.stButton > button:active {

        background-color: #EFA0B0 !important;
        color: #173F68 !important;
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border-color: #9CC9D9 !important;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():

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
# FEATURE LISTS
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
# 35 FEATURE NAMES
# ============================================================

feature_names = [

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
    "day_of_week_Wednesday",

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
# PEAK HOUR
# ============================================================

def calculate_peak_hour(hour):

    if 7 <= hour <= 10:
        return 1

    if 17 <= hour <= 20:
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

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_obj = pd.to_datetime(
        selected_date,
        format="%Y-%m-%d"
    )


    # --------------------------------------------------------
    # TIME
    # --------------------------------------------------------

    time_obj = datetime.strptime(
        selected_time,
        "%H:%M"
    )


    # --------------------------------------------------------
    # CATEGORICAL FEATURES
    # --------------------------------------------------------

    categorical_df = pd.DataFrame({

        "city": [city],

        "road_type": [road_type],

        "weather": [weather],

        "visibility": [visibility],

        "traffic_density": [traffic_density],

        "day_of_week": [day_of_week]

    })


    encoded = encoder.transform(
        categorical_df
    )


    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()


    # --------------------------------------------------------
    # NUMERICAL FEATURES
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    final_input = np.hstack(
        [encoded, scaled]
    )


    final_input = final_input.astype(
        np.float32
    )


    # --------------------------------------------------------
    # CHECK FEATURE COUNT
    # --------------------------------------------------------

    if final_input.shape[1] != 35:

        raise ValueError(
            f"Expected 35 features but received "
            f"{final_input.shape[1]}"
        )


    return final_input


# ============================================================
# MODEL PREDICTION
# ============================================================

def safe_predict(model, X):

    X = np.asarray(
        X,
        dtype=np.float32
    )

    tensor_input = tf.convert_to_tensor(
        X,
        dtype=tf.float32
    )

    output = model(
        tensor_input,
        training=False
    )

    return output.numpy().reshape(-1)


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
# LOAD TRAINED MODEL
# ============================================================

try:

    model, encoder, scaler = (
        load_model_and_preprocessing()
    )

except Exception as e:

    st.error(
        "Unable to load the trained model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:#173F68;
            margin-bottom:8px;
        ">
            🚦 Road Accident Risk
        </div>

        <div style="
            font-size:15px;
            font-weight:500;
            color:#315878;
            margin-bottom:20px;
        ">
            Deep Learning + Explainable AI
        </div>
        """
    )


    st.html(
        """
        <div style="
            font-size:15px;
            line-height:1.65;
            color:#315878;
            margin-bottom:22px;
        ">
            This application estimates road accident
            risk based on the conditions entered by
            the user.
        </div>
        """
    )


    st.divider()


    # --------------------------------------------------------
    # RISK LEVELS TITLE
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            font-size:20px;
            font-weight:750;
            color:#173F68;
            margin-top:10px;
            margin-bottom:22px;
        ">
            Risk Levels
        </div>
        """
    )


    # --------------------------------------------------------
    # LOW RISK
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            margin-bottom:22px;
        ">

            <div style="
                font-size:16px;
                font-weight:700;
                color:#315878;
            ">

                <span style="
                    display:inline-block;
                    width:17px;
                    height:17px;
                    border-radius:50%;
                    background:#4FD18B;
                    margin-right:9px;
                    vertical-align:-2px;
                "></span>

                Low Risk

            </div>

            <div style="
                margin-top:9px;
                font-size:14px;
                color:#315878;
            ">
                Score &lt; 0.25
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # MEDIUM RISK
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            margin-bottom:22px;
        ">

            <div style="
                font-size:16px;
                font-weight:700;
                color:#315878;
            ">

                <span style="
                    display:inline-block;
                    width:17px;
                    height:17px;
                    border-radius:50%;
                    background:#F4C542;
                    margin-right:9px;
                    vertical-align:-2px;
                "></span>

                Medium Risk

            </div>

            <div style="
                margin-top:9px;
                font-size:14px;
                color:#315878;
            ">
                Score 0.25 – &lt; 0.60
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    st.html(
        """
        <div style="
            margin-bottom:22px;
        ">

            <div style="
                font-size:16px;
                font-weight:700;
                color:#315878;
            ">

                <span style="
                    display:inline-block;
                    width:17px;
                    height:17px;
                    border-radius:50%;
                    background:#F05A68;
                    margin-right:9px;
                    vertical-align:-2px;
                "></span>

                High Risk

            </div>

            <div style="
                margin-top:9px;
                font-size:14px;
                color:#315878;
            ">
                Score ≥ 0.60
            </div>

        </div>
        """
    )


    st.divider()


    st.html(
        """
        <div style="
            font-size:13px;
            color:#52708D;
            margin-top:20px;
        ">
            DL-Based Road Accident Risk Prediction
        </div>
        """
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.html(
    """
    <div style="
        padding:20px 0 5px 0;
    ">

        <div style="
            font-size:42px;
            font-weight:800;
            color:#173F68;
            line-height:1.15;
        ">
            🚦 Road Accident Risk Prediction
        </div>

        <div style="
            font-size:17px;
            color:#315878;
            margin-top:12px;
        ">
            Deep Learning based risk estimation
            with Explainable AI
        </div>

    </div>
    """
)


# ============================================================
# SPACER
# ============================================================

st.html(
    """
    <div style="
        height:30px;
        background:#FFFDF9;
        border:1px solid #D5E1E8;
        border-radius:18px;
        margin:34px 0 28px 0;
        box-shadow:
            0 5px 18px rgba(35,75,100,0.08);
    ">
    </div>
    """
)


# ============================================================
# INPUT SECTION TITLE
# ============================================================

st.html(
    """
    <div style="
        font-size:28px;
        font-weight:800;
        color:#173F68;
        margin-bottom:18px;
    ">
        Enter Road Conditions
    </div>
    """
)


# ============================================================
# INPUT CARD START
# ============================================================

st.markdown(
    '<div class="input-card">',
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
        value=datetime.now().date()
    )


# ============================================================
# ROW 3
# ============================================================

col7, col8, col9 = st.columns(3)


with col7:

    selected_time_obj = st.time_input(
        "Time",
        value=datetime.now().time().replace(
            second=0,
            microsecond=0
        )
    )


# ============================================================
# AUTOMATIC WEEKEND
# ============================================================

automatic_weekend = (
    1
    if selected_date_obj.weekday() >= 5
    else 0
)


# ============================================================
# AUTOMATIC PEAK HOUR
# ============================================================

automatic_peak_hour = calculate_peak_hour(
    selected_time_obj.hour
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_selected_date" not in st.session_state:

    st.session_state.last_selected_date = (
        selected_date_obj
    )


if "last_selected_time" not in st.session_state:

    st.session_state.last_selected_time = (
        selected_time_obj
    )


if "is_weekend" not in st.session_state:

    st.session_state.is_weekend = (
        automatic_weekend
    )


if "is_peak_hour" not in st.session_state:

    st.session_state.is_peak_hour = (
        automatic_peak_hour
    )


# ============================================================
# UPDATE WEEKEND WHEN DATE CHANGES
# ============================================================

if (
    selected_date_obj
    != st.session_state.last_selected_date
):

    st.session_state.is_weekend = (
        automatic_weekend
    )

    st.session_state.last_selected_date = (
        selected_date_obj
    )


# ============================================================
# UPDATE PEAK HOUR WHEN TIME CHANGES
# ============================================================

if (
    selected_time_obj
    != st.session_state.last_selected_time
):

    st.session_state.is_peak_hour = (
        automatic_peak_hour
    )

    st.session_state.last_selected_time = (
        selected_time_obj
    )


# ============================================================
# WEEKEND
# ============================================================

with col8:

    is_weekend = st.radio(
        "Is Weekend",
        [0, 1],
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
        [0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No",
        horizontal=True,
        key="is_peak_hour"
    )


# ============================================================
# DERIVED VALUES
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


# ============================================================
# DATE/TIME SUMMARY
# ============================================================

st.html(
    f"""
    <div style="
        background:#EAF5F9;
        border:1px solid #D0E2E9;
        border-radius:10px;
        padding:13px 17px;
        margin-top:12px;
        margin-bottom:22px;
        font-size:15px;
        color:#315878;
    ">

        📅 {selected_date}

        &nbsp; | &nbsp;

        📆 {day_of_week}

        &nbsp; | &nbsp;

        🕐 {selected_time}

        &nbsp; | &nbsp;

        Weekend: <b>{weekend_text}</b>

        &nbsp; | &nbsp;

        Peak Hour: <b>{peak_text}</b>

    </div>
    """
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_clicked = st.button(
    "🚗  Predict Accident Risk",
    use_container_width=True
)


# ============================================================
# INPUT CARD END
# ============================================================

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    try:

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = safe_predict(
            model,
            final_input
        )


        risk_score = float(
            prediction[0]
        )


        risk_score = float(
            np.clip(
                risk_score,
                0.0,
                1.0
            )
        )


        # ----------------------------------------------------
        # CATEGORY
        # ----------------------------------------------------

        risk_category = get_risk_category(
            risk_score
        )


        # ====================================================
        # RESULT TITLE
        # ====================================================

        st.html(
            """
            <div style="
                font-size:28px;
                font-weight:800;
                color:#173F68;
                margin-top:35px;
                margin-bottom:18px;
            ">
                Prediction Result
            </div>
            """
        )


        # ====================================================
        # RESULT COLORS
        # ====================================================

        if risk_category == "Low Risk":

            result_bg = "#E2F4E9"
            result_text = "#18794E"

        elif risk_category == "Medium Risk":

            result_bg = "#FFF2CC"
            result_text = "#9A6A00"

        else:

            result_bg = "#FFE2E2"
            result_text = "#C62828"


        # ====================================================
        # RESULT CARD
        # ====================================================

        st.html(
            f"""
            <div style="
                background:#F4F9FD;

                border:1px solid #D1E0E9;

                border-radius:18px;

                padding:32px;

                text-align:center;

                margin-top:5px;
                margin-bottom:28px;

                box-shadow:
                    0 7px 22px
                    rgba(35,75,100,0.10);
            ">

                <div style="
                    font-size:17px;
                    font-weight:650;
                    color:#315878;
                    margin-bottom:8px;
                ">
                    Predicted Risk Score
                </div>


                <div style="
                    font-size:46px;
                    font-weight:800;
                    color:#173F68;
                    margin:4px 0 16px 0;
                ">
                    {risk_score:.4f}
                </div>


                <div style="
                    display:inline-block;

                    background:{result_bg};

                    color:{result_text};

                    padding:10px 25px;

                    border-radius:25px;

                    font-size:17px;

                    font-weight:750;
                ">
                    {risk_category}
                </div>

            </div>
            """
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.html(
            """
            <div style="
                font-size:25px;
                font-weight:800;
                color:#173F68;
                margin-top:25px;
                margin-bottom:18px;
            ">
                Input Summary
            </div>
            """
        )


        summary1, summary2 = st.columns(2)


        # ----------------------------------------------------
        # LOCATION & ROAD
        # ----------------------------------------------------

        with summary1:

            st.html(
                f"""
                <div style="
                    background:#FFFDF9;

                    border:1px solid #D8E3E9;

                    border-radius:14px;

                    padding:21px;

                    color:#315878;

                    line-height:1.9;

                    min-height:175px;

                    box-shadow:
                        0 4px 14px
                        rgba(35,75,100,0.06);
                ">

                    <div style="
                        font-size:17px;
                        font-weight:750;
                        color:#173F68;
                        margin-bottom:10px;
                    ">
                        Location & Road
                    </div>

                    🌆 City:
                    <b>{city}</b>

                    <br>

                    🛣️ Road Type:
                    <b>{road_type}</b>

                    <br>

                    🚗 Traffic Density:
                    <b>{traffic_density}</b>

                </div>
                """
            )


        # ----------------------------------------------------
        # ENVIRONMENT & TIME
        # ----------------------------------------------------

        with summary2:

            st.html(
                f"""
                <div style="
                    background:#FFFDF9;

                    border:1px solid #D8E3E9;

                    border-radius:14px;

                    padding:21px;

                    color:#315878;

                    line-height:1.9;

                    min-height:175px;

                    box-shadow:
                        0 4px 14px
                        rgba(35,75,100,0.06);
                ">

                    <div style="
                        font-size:17px;
                        font-weight:750;
                        color:#173F68;
                        margin-bottom:10px;
                    ">
                        Environment & Time
                    </div>

                    🌦️ Weather:
                    <b>{weather}</b>

                    <br>

                    👁️ Visibility:
                    <b>{visibility}</b>

                    <br>

                    📆 Day:
                    <b>{day_of_week}</b>

                    <br>

                    🕐 Time:
                    <b>{selected_time}</b>

                </div>
                """
            )


        # ====================================================
        # SHAP TITLE
        # ====================================================

        st.html(
            """
            <div style="
                font-size:25px;
                font-weight:800;
                color:#173F68;
                margin-top:35px;
                margin-bottom:8px;
            ">
                Explainable AI — SHAP
            </div>

            <div style="
                font-size:15px;
                color:#315878;
                margin-bottom:18px;
            ">
                SHAP explains which features contributed
                most to the predicted risk.
            </div>
            """
        )


        # ====================================================
        # SHAP
        # ====================================================

        with st.spinner(
            "Generating explanation..."
        ):

            try:

                explainer = create_shap_explainer(
                    model
                )


                shap_values = explainer.shap_values(
                    final_input
                )


                # ------------------------------------------------
                # HANDLE LIST OUTPUT
                # ------------------------------------------------

                if isinstance(
                    shap_values,
                    list
                ):

                    shap_values = (
                        shap_values[0]
                    )


                shap_values = np.asarray(
                    shap_values
                )


                # ------------------------------------------------
                # REMOVE OUTPUT DIMENSION
                # ------------------------------------------------

                if shap_values.ndim == 3:

                    shap_values = np.squeeze(
                        shap_values,
                        axis=-1
                    )


                shap_values = shap_values.reshape(
                    -1,
                    35
                )


                values = shap_values[0]


                # ------------------------------------------------
                # TOP 10 FEATURES
                # ------------------------------------------------

                top_indices = np.argsort(
                    np.abs(values)
                )[-10:][::-1]


                top_names = [

                    feature_names[i]

                    for i in top_indices

                ]


                top_values = [

                    float(values[i])

                    for i in top_indices

                ]


                # =================================================
                # SHAP CHART
                # =================================================

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


                # =================================================
                # SHAP TABLE
                # =================================================

                shap_table = pd.DataFrame({

                    "Feature": top_names,

                    "SHAP Value": [

                        round(
                            value,
                            5
                        )

                        for value in top_values

                    ],

                    "Impact": [

                        "Increases Risk"

                        if value > 0

                        else "Decreases Risk"

                        for value in top_values

                    ]

                })


                st.dataframe(
                    shap_table,
                    use_container_width=True,
                    hide_index=True
                )


            except Exception as shap_error:

                st.warning(
                    "Prediction succeeded, but the "
                    "SHAP explanation could not be generated."
                )

                st.caption(
                    str(shap_error)
                )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.html(
            """
            <div style="
                background:#FFF8E7;

                border:1px solid #EBD9A8;

                border-radius:12px;

                padding:15px 18px;

                margin-top:25px;

                color:#6E5A25;

                font-size:14px;

                line-height:1.6;
            ">
                ⚠️ The predicted score is an estimated risk
                based on the conditions entered into the model.
                It does not guarantee that an accident will occur.
            </div>
            """
        )


    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "An error occurred while making the prediction."
        )

        st.code(
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;

        color:#52708D;

        font-size:13px;

        padding:30px 10px 15px 10px;

        margin-top:25px;
    ">
        DL-Based Road Accident Risk Prediction
        &nbsp; • &nbsp;
        Deep Learning + Explainable AI
    </div>
    """
)
