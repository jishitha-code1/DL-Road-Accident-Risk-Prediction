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

    /* ---------- MAIN BACKGROUND ---------- */

    .stApp {
        background: #f8f1e3;
    }

    [data-testid="stAppViewContainer"] {
        background: #f8f1e3;
    }

    [data-testid="stHeader"] {
        background: #f8f1e3;
    }


    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background: #eef5fa;
        border-right: 1px solid #d5e1e9;
    }


    /* ---------- TEXT ---------- */

    h1, h2, h3 {
        color: #173f68 !important;
    }

    p, label {
        color: #315878 !important;
    }


    /* ---------- INPUTS ---------- */

    div[data-baseweb="select"] > div {
        background: #fffdf9 !important;
        border: 1px solid #cbd9e4 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] > div {
        background: #fffdf9 !important;
        border: 1px solid #cbd9e4 !important;
        border-radius: 10px !important;
    }

    input {
        background: #fffdf9 !important;
        color: #173f68 !important;
    }


    /* ---------- INPUT CARD ---------- */

    .input-card {
        background: #fffdf9;
        border: 1px solid #dce5eb;
        border-radius: 18px;
        padding: 25px 28px;
        margin-top: 12px;
        margin-bottom: 25px;
        box-shadow: 0 6px 20px rgba(40, 70, 95, 0.08);
    }


    /* ---------- INFO BAR ---------- */

    .info-bar {
        background: #f1f7fb;
        border: 1px solid #cfdeea;
        border-radius: 11px;
        padding: 13px 16px;
        margin-top: 18px;
        margin-bottom: 20px;
        color: #315878;
        font-size: 14px;
    }


    /* ---------- PREDICT BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        min-height: 54px;

        background:
            linear-gradient(
                100deg,
                #58a6eb 0%,
                #337fc8 50%,
                #205b96 100%
            ) !important;

        color: white !important;

        border: none !important;
        border-radius: 11px !important;

        font-size: 16px !important;
        font-weight: 700 !important;

        box-shadow:
            0 6px 15px rgba(36, 100, 160, 0.25);

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background:
            linear-gradient(
                100deg,
                #4397df 0%,
                #286fb7 50%,
                #174d80 100%
            ) !important;

        color: white !important;

        transform: translateY(-1px);

        box-shadow:
            0 8px 18px rgba(36, 100, 160, 0.32);
    }


    /* ---------- RESULT CARD ---------- */

    .result-card {
        background: #f4f9fd;
        border: 1px solid #d5e2eb;
        border-radius: 17px;
        padding: 28px;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 25px;
        box-shadow: 0 5px 18px rgba(40, 70, 95, 0.07);
    }

    .result-title {
        font-size: 17px;
        font-weight: 650;
        color: #315878;
        margin-bottom: 7px;
    }

    .score {
        font-size: 43px;
        font-weight: 800;
        color: #173f68;
        margin: 5px 0 12px 0;
    }


    /* ---------- SUMMARY ---------- */

    .summary-card {
        background: #fffdf9;
        border: 1px solid #dce5eb;
        border-radius: 14px;
        padding: 20px;
        line-height: 1.9;
        color: #315878;
        box-shadow: 0 4px 14px rgba(40, 70, 95, 0.06);
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #6b8398;
        font-size: 13px;
        padding: 25px 10px 10px 10px;
        margin-top: 25px;
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
# FEATURES
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
# 35 FEATURES
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

    date_obj = pd.to_datetime(
        selected_date,
        format="%Y-%m-%d"
    )

    time_obj = datetime.strptime(
        selected_time,
        "%H:%M"
    )

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

    final_input = np.hstack(
        [encoded, scaled]
    )

    final_input = final_input.astype(
        np.float32
    )

    if final_input.shape[1] != 35:

        raise ValueError(
            f"Expected 35 features but got "
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
# SHAP
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

    return shap.Explainer(
        prediction_function,
        background,
        feature_names=feature_names
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
        "Unable to load the trained model."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:25px;
            font-weight:800;
            color:#173f68;
            margin-bottom:5px;
        ">
            🚦 Road Accident Risk
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            font-size:15px;
            color:#52708d;
            margin-bottom:20px;
        ">
            Deep Learning + Explainable AI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "This application estimates road accident "
        "risk based on the conditions entered by "
        "the user."
    )

    st.divider()

    st.subheader("Risk Levels")

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

    st.divider()

    st.caption(
        "DL-Based Road Accident Risk Prediction"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🚗 Road Accident Risk Prediction"
)

st.write(
    "Deep Learning based risk estimation "
    "with Explainable AI"
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="input-card">',
    unsafe_allow_html=True
)

st.subheader(
    "Enter Road Conditions"
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
# AUTOMATIC VALUES
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
# SESSION STATE
# ============================================================

if "last_date" not in st.session_state:

    st.session_state.last_date = (
        selected_date_obj
    )

if "last_time" not in st.session_state:

    st.session_state.last_time = (
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
# DATE CHANGE → UPDATE WEEKEND
# ============================================================

if (
    selected_date_obj
    != st.session_state.last_date
):

    st.session_state.is_weekend = (
        automatic_weekend
    )

    st.session_state.last_date = (
        selected_date_obj
    )


# ============================================================
# TIME CHANGE → UPDATE PEAK HOUR
# ============================================================

if (
    selected_time_obj
    != st.session_state.last_time
):

    st.session_state.is_peak_hour = (
        automatic_peak_hour
    )

    st.session_state.last_time = (
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
# INFO BAR
# ============================================================

st.info(
    f"📅 {selected_date}  |  "
    f"📆 {day_of_week}  |  "
    f"🕐 {selected_time}  |  "
    f"Weekend: {weekend_text}  |  "
    f"Peak Hour: {peak_text}"
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
# PREDICTION SECTION
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
        # RESULT
        # ====================================================

        st.subheader(
            "Prediction Result"
        )


        if risk_category == "Low Risk":

            bg = "#dff5e8"
            text = "#16834a"

        elif risk_category == "Medium Risk":

            bg = "#fff1c9"
            text = "#9a6900"

        else:

            bg = "#ffe1e1"
            text = "#c62828"


        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-title">
                    Predicted Risk Score
                </div>

                <div class="score">
                    {risk_score:.4f}
                </div>

                <div style="
                    display:inline-block;
                    background:{bg};
                    color:{text};
                    padding:9px 22px;
                    border-radius:25px;
                    font-size:16px;
                    font-weight:750;
                ">
                    {risk_category}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader(
            "Input Summary"
        )


        summary1, summary2 = st.columns(2)


        with summary1:

            st.markdown(
                f"""
                <div class="summary-card">

                <b>Location & Road</b><br><br>

                🌆 City:
                <b>{city}</b><br>

                🛣️ Road Type:
                <b>{road_type}</b><br>

                🚗 Traffic Density:
                <b>{traffic_density}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        with summary2:

            st.markdown(
                f"""
                <div class="summary-card">

                <b>Environment & Time</b><br><br>

                🌦️ Weather:
                <b>{weather}</b><br>

                👁️ Visibility:
                <b>{visibility}</b><br>

                📅 Day:
                <b>{day_of_week}</b><br>

                🕐 Time:
                <b>{selected_time}</b>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # SHAP
        # ====================================================

        st.subheader(
            "Explainable AI — SHAP"
        )

        st.write(
            "SHAP explains which features contributed "
            "most to the predicted risk."
        )


        with st.spinner(
            "Generating explanation..."
        ):

            try:

                explainer = (
                    create_shap_explainer(
                        model
                    )
                )


                shap_result = explainer(
                    final_input
                )


                shap_values = (
                    shap_result.values
                )


                # ------------------------------------------------
                # SHAP DIMENSION
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
                # TOP FEATURES
                # ------------------------------------------------

                top_indices = np.argsort(
                    np.abs(values)
                )[-10:][::-1]


                top_names = [
                    feature_names[i]
                    for i in top_indices
                ]


                top_values = [
                    values[i]
                    for i in top_indices
                ]


                # =================================================
                # CHART
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
                # TABLE
                # =================================================

                shap_table = pd.DataFrame({

                    "Feature": top_names,

                    "SHAP Value": [
                        round(
                            float(v),
                            5
                        )
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
                    "Prediction succeeded, but "
                    "SHAP explanation could not "
                    "be generated."
                )

                st.caption(
                    str(shap_error)
                )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            "The predicted score is an estimated risk "
            "based on the conditions entered into the "
            "model. It does not guarantee that an accident "
            "will occur."
        )


    # ========================================================
    # OUTER PREDICTION ERROR
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
