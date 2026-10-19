import os

# Set limits BEFORE importing TensorFlow, NumPy, or other numerical libraries.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from datetime import datetime, time
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning, module="keras")


# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Road Accident Risk Prediction",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "improved_model (3).keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# These are the five raw columns in the exact order used for preprocessing.
MODEL_FEATURES = [
    "visibility",
    "traffic_density",
    "is_peak_hour",
    "weather",
    "hour",
]

BEST_MODEL_NAME = "Deep Learning DNN (5 input features)"
BEST_MODEL_PARAMETERS = 1057
BEST_MODEL_MAE = 0.0504
BEST_MODEL_RMSE = 0.0720
BEST_MODEL_R2 = 0.8884


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #ADD8E6;
    }
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        background-color: #EED9C4 !important;
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, h4, p, label { color: #173F68; }
    .main-title {
        font-size: 38px; font-weight: 800; color: #173F68;
        text-align: center; margin: 10px 0 8px;
    }
    .main-subtitle {
        font-size: 17px; color: #315878; text-align: center;
        margin-bottom: 28px;
    }
    .section-title {
        font-size: 27px; font-weight: 800; color: #173F68;
        margin-top: 25px; margin-bottom: 18px;
    }
    .info-card, .summary-card, .result-card {
        background: #F7FBFD; border: 1px solid #D3E2EA;
        border-radius: 16px; padding: 20px; margin-bottom: 20px;
        box-shadow: 0 5px 16px rgba(30,70,100,.08);
        color: #173F68;
    }
    .result-card { text-align: center; padding: 34px 20px; }
    .result-title { font-size: 19px; color: #315878; font-weight: 700; }
    .score { font-size: 46px; font-weight: 800; color: #173F68; margin: 8px 0 16px; }
    .low-risk, .medium-risk, .high-risk {
        display: inline-block; padding: 9px 24px; border-radius: 24px;
        font-size: 16px; font-weight: 750;
    }
    .low-risk { background: #D8F5E5; color: #16824A; }
    .medium-risk { background: #FFF1C9; color: #946200; }
    .high-risk { background: #FFE0E0; color: #C62828; }
    div.stButton > button {
        width: 100%; min-height: 50px; background: #EED9C4;
        color: #173F68; border: 1px solid #F09AAA;
        border-radius: 12px; font-weight: 700;
    }
    div.stButton > button:hover { background: #F4B0C0; color: #173F68; }
    [data-baseweb="select"] > div, [data-baseweb="input"] > div {
        background-color: #F8FAFC;
    }
    .footer-text { text-align: center; color: #315878; margin-top: 32px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Model loading: imports are delayed until required, and resources cached.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the trained risk model...")
def load_risk_model():
    import tensorflow as tf

    # Limit TensorFlow thread pools to reduce resource pressure in hosted apps.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # TensorFlow may already have initialized its runtime.
        pass

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Risk model not found: {MODEL_PATH}. "
            "Check that the filename and models folder match your repository."
        )
    if not PREPROCESSOR_PATH.is_file():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}. "
            "Check that preprocessor.pkl is inside models/."
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model_shape = model.input_shape
    if isinstance(model_shape, list):
        if len(model_shape) != 1:
            raise ValueError(f"Multiple model inputs are not supported: {model_shape}")
        model_shape = model_shape[0]

    expected = int(model_shape[-1])
    if not hasattr(preprocessor, "transform"):
        raise TypeError("preprocessor.pkl does not contain a transformer with transform().")

    # Validate compatibility when the preprocessor exposes feature names.
    try:
        transformed_count = len(preprocessor.get_feature_names_out())
    except Exception:
        transformed_count = None

    if transformed_count is not None and transformed_count != expected:
        raise ValueError(
            f"Model expects {expected} transformed features, but the preprocessor "
            f"reports {transformed_count}. Use the model and preprocessor saved "
            "from the same training run."
        )

    return model, preprocessor, expected


def get_risk_category(score):
    if score < 0.25:
        return "Low Risk"
    if score < 0.60:
        return "Medium Risk"
    return "High Risk"


def calculate_peak_hour(hour):
    return int(7 <= hour <= 10 or 17 <= hour <= 20)


def preprocess_input(
    weather,
    visibility,
    traffic_density,
    selected_time,
    is_peak_hour,
    preprocessor,
    expected_features,
):
    raw = pd.DataFrame(
        [
            {
                "visibility": str(visibility),
                "traffic_density": str(traffic_density),
                "is_peak_hour": int(is_peak_hour),
                "weather": str(weather),
                "hour": int(selected_time.hour),
            }
        ],
        columns=MODEL_FEATURES,
    )

    transformed = preprocessor.transform(raw)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=np.float32)

    if transformed.ndim != 2 or transformed.shape[0] != 1:
        raise ValueError(f"Unexpected preprocessor output shape: {transformed.shape}")
    if transformed.shape[1] != expected_features:
        raise ValueError(
            f"Preprocessor produced {transformed.shape[1]} features, "
            f"but the model expects {expected_features}. Ensure the model and "
            "preprocessor belong to the same training run."
        )
    if not np.isfinite(transformed).all():
        raise ValueError("Preprocessing produced NaN or infinite values.")
    return transformed


def predict_risk(model, processed_input):
    # Calling the model directly avoids Keras predict() data-adapter warnings.
    output = model(np.asarray(processed_input, dtype=np.float32), training=False)
    raw_score = float(np.asarray(output).reshape(-1)[0])
    if not np.isfinite(raw_score):
        raise ValueError("The model returned a non-finite prediction.")
    # This bounds the display score; it does not calibrate it as a probability.
    return float(np.clip(raw_score, 0.0, 1.0))


def get_feature_names(preprocessor, expected_features):
    try:
        names = list(preprocessor.get_feature_names_out())
        if len(names) == expected_features:
            return names
    except Exception:
        pass
    return [f"Processed feature {i + 1}" for i in range(expected_features)]


def make_shap_explanation(model, processed_input, preprocessor, expected_features):
    """
    Optional, model-agnostic SHAP explanation.
    A small background and limited sample count are used to reduce CPU load.
    If SHAP/native dependencies fail, the prediction remains available.
    """
    try:
        import shap

        background = np.zeros((1, expected_features), dtype=np.float32)

        def predict_batch(batch):
            batch = np.asarray(batch, dtype=np.float32)
            result = model(batch, training=False)
            return np.asarray(result).reshape(-1)

        explainer = shap.KernelExplainer(predict_batch, background)
        values = explainer.shap_values(
            np.asarray(processed_input, dtype=np.float32),
            nsamples=40,
            silent=True,
        )

        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        values = np.squeeze(values)

        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim > 2:
            values = values.reshape(values.shape[0], -1)

        names = get_feature_names(preprocessor, expected_features)
        if values.shape[-1] != len(names):
            return None, None
        return names, values[0]
    except Exception:
        return None, None


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🚦 Road Accident Risk")
    st.caption("Deep Learning + Explainable AI")
    st.write(
        "Estimate road accident risk using selected visibility, traffic, "
        "weather, and time conditions."
    )
    st.divider()
    st.markdown("### Risk Levels")
    st.markdown("🟢 **Low Risk** — score below 0.25")
    st.markdown("🟡 **Medium Risk** — score from 0.25 to below 0.60")
    st.markdown("🔴 **High Risk** — score 0.60 or above")
    st.divider()
    st.caption("DL-Based Road Accident Risk Prediction")


# ---------------------------------------------------------------------
# Main heading and model load
# ---------------------------------------------------------------------
st.markdown(
    '<div class="main-title">🚦 Road Accident Risk Prediction</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="main-subtitle">Deep Learning based risk estimation with Explainable AI</div>',
    unsafe_allow_html=True,
)

try:
    model, preprocessor, EXPECTED_FEATURES = load_risk_model()
except Exception as exc:
    st.error("Unable to load the accident-risk model or preprocessor.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Verify that models/improved_model (3).keras and "
        "models/preprocessor.pkl exist and come from the same training run."
    )
    st.stop()


# ---------------------------------------------------------------------
# Input form: only run prediction when submitted
# ---------------------------------------------------------------------
st.markdown(
    '<div class="section-title">Enter Road Conditions</div>',
    unsafe_allow_html=True,
)

with st.form("risk_prediction_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
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
                "Pune",
            ],
        )

    with c2:
        road_type = st.selectbox(
            "Road Type (context only; not used by model)",
            ["highway", "rural", "urban"],
        )

    with c3:
        weather = st.selectbox("Weather", ["clear", "fog", "rain"])

    c4, c5, c6 = st.columns(3)

    with c4:
        visibility = st.selectbox("Visibility", ["high", "low", "medium"])

    with c5:
        traffic_density = st.selectbox(
            "Traffic Density", ["high", "low", "medium"]
        )

    with c6:
        selected_date = st.date_input("Date", value=datetime.now().date())

    c7, c8, c9 = st.columns(3)

    with c7:
        now = datetime.now()
        selected_time = st.time_input("Time", value=time(now.hour, now.minute))

    is_peak_hour = calculate_peak_hour(selected_time.hour)
    is_weekend = int(pd.Timestamp(selected_date).weekday() >= 5)
    day_of_week = pd.Timestamp(selected_date).strftime("%A")
    weekend_text = "Yes" if is_weekend else "No"
    peak_text = "Yes" if is_peak_hour else "No"

    st.caption(
        f"Date: {selected_date:%Y-%m-%d} | {day_of_week} | "
        f"Time: {selected_time:%H:%M} | Weekend: {weekend_text} | "
        f"Peak hour: {peak_text}"
    )

    submitted = st.form_submit_button(
        "🚗 Predict Accident Risk",
        width="stretch",
    )


if submitted:
    try:
        processed_input = preprocess_input(
            weather=weather,
            visibility=visibility,
            traffic_density=traffic_density,
            selected_time=selected_time,
            is_peak_hour=is_peak_hour,
            preprocessor=preprocessor,
            expected_features=EXPECTED_FEATURES,
        )

        risk_score = predict_risk(model, processed_input)
        risk_category = get_risk_category(risk_score)

        category_class = {
            "Low Risk": "low-risk",
            "Medium Risk": "medium-risk",
            "High Risk": "high-risk",
        }[risk_category]

        st.markdown(
            '<div class="section-title">Prediction Result</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="result-card"><div class="result-title">Predicted Risk Score</div>'
            f'<div class="score">{risk_score:.4f}</div>'
            f'<div class="{category_class}">{risk_category}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Input Summary</div>',
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)

        with left:
            st.markdown(
                f'<div class="info-card"><b>City:</b> {city}<br><br>'
                f'<b>Road type:</b> {road_type}<br><br>'
                f'<b>Weather:</b> {weather}<br><br>'
                f'<b>Visibility:</b> {visibility}<br><br>'
                f'<b>Traffic density:</b> {traffic_density}</div>',
                unsafe_allow_html=True,
            )

        with right:
            st.markdown(
                f'<div class="info-card"><b>Date:</b> {selected_date:%Y-%m-%d}<br><br>'
                f'<b>Day:</b> {day_of_week}<br><br>'
                f'<b>Time:</b> {selected_time:%H:%M}<br><br>'
                f'<b>Weekend:</b> {weekend_text}<br><br>'
                f'<b>Peak hour:</b> {peak_text}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-title">Explainable AI</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "SHAP estimates how transformed input features contribute to this model's output. "
            "These are model explanations, not proof of causation."
        )

        with st.spinner("Attempting to generate a SHAP explanation..."):
            feature_names, shap_values = make_shap_explanation(
                model, processed_input, preprocessor, EXPECTED_FEATURES
            )

        if feature_names is not None and shap_values is not None:
            shap_df = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "SHAP Value": np.asarray(shap_values, dtype=float),
                }
            )
            shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
            shap_df = shap_df.sort_values("Absolute Impact", ascending=False)
            top = shap_df.head(10).sort_values("Absolute Impact", ascending=True)

            fig, ax = plt.subplots(figsize=(9, max(3.5, 0.42 * len(top))))
            ax.barh(top["Feature"], top["SHAP Value"])
            ax.set_xlabel("SHAP value (positive/negative contribution)")
            ax.set_title("Features Contributing to the Model Output")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

            display_df = shap_df.head(10)[["Feature", "SHAP Value"]].copy()
            display_df["SHAP Value"] = display_df["SHAP Value"].round(5)
            st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.info(
                "The risk prediction succeeded, but SHAP could not be generated in this environment. "
                "This does not affect the prediction."
            )

        st.markdown(
            '<div class="section-title">Model Performance</div>',
            unsafe_allow_html=True,
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("MAE", f"{BEST_MODEL_MAE:.4f}")
        p2.metric("RMSE", f"{BEST_MODEL_RMSE:.4f}")
        p3.metric("R² Score", f"{BEST_MODEL_R2:.4f}")
        p4.metric("Parameters", f"{BEST_MODEL_PARAMETERS:,}")

        st.markdown(
            f'<div class="info-card"><b>Selected model:</b> {BEST_MODEL_NAME}<br><br>'
            f'<b>Architecture reported for this model:</b> 11 → 32 → 16 → 8 → 1<br><br>'
            f'<b>Optimizer:</b> Adam<br><b>Loss:</b> Mean Squared Error (MSE)<br>'
            f'<b>Output activation:</b> Sigmoid</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Model and preprocessing information"):
            st.write("Model input shape:", model.input_shape)
            st.write("Model output shape:", model.output_shape)
            st.write("Raw feature order:", ", ".join(MODEL_FEATURES))
            st.write("Transformed feature count:", processed_input.shape[1])
            st.write("Preprocessing: saved transformer loaded from preprocessor.pkl")
            st.write("Prediction: direct Keras inference on CPU")
            st.write("Explainability: optional SHAP KernelExplainer")

        st.warning(
            "This is an experimental estimate based on entered conditions. "
            "It is not a calibrated probability and cannot guarantee whether an accident will occur."
        )

    except Exception as exc:
        st.error("Prediction could not be completed.")
        st.code(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------
# Image-based integrated risk analysis: traffic + weather only
# ---------------------------------------------------------------------
st.divider()
st.markdown(
    '<div class="section-title">Image-Based Integrated Accident Risk</div>',
    unsafe_allow_html=True,
)
st.write(
    "Upload one road image. The app estimates visible weather using an image "
    "classifier and detects visible vehicles to estimate traffic density. "
    "These inferred conditions are then passed to the existing risk model."
)
st.info(
    "This workflow uses separate pretrained models for vehicle detection and "
    "weather classification. First-time use may require internet access to "
    "download their weights. The result is an experimental estimate, not a "
    "calibrated accident probability."
)

integrated_image_file = st.file_uploader(
    "Upload one road image for traffic and weather analysis",
    type=["jpg", "jpeg", "png"],
    key="integrated_road_image_uploader",
)

if integrated_image_file is not None:
    try:
        integrated_image = Image.open(integrated_image_file).convert("RGB")
        st.image(integrated_image, caption="Image to analyze", width="stretch")
    except Exception as exc:
        integrated_image = None
        st.error("The uploaded image could not be opened.")
        st.code(f"{type(exc).__name__}: {exc}")
else:
    integrated_image = None

if st.button("Analyze Traffic and Weather", key="integrated_image_risk_button"):
    if integrated_image is None:
        st.error("Please upload a valid road image first.")
    else:
        try:
            with tempfile.TemporaryDirectory(prefix="integrated_road_risk_") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "input.png"
                vehicle_output_path = temp_path / "vehicles.png"
                metadata_path = temp_path / "analysis.json"
                integrated_image.save(input_path, format="PNG")

                integrated_worker = r"""
import json, os, sys, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

input_path, vehicle_output, metadata_path = sys.argv[1:4]

try:
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from PIL import Image
    from ultralytics import YOLO
    from transformers import pipeline

    image = Image.open(input_path).convert("RGB")

    # Detect visible vehicles using the pretrained COCO YOLO model.
    vehicle_model = YOLO("yolov8n.pt")
    vehicle_result = vehicle_model.predict(
        source=input_path, conf=0.25, device="cpu", imgsz=640,
        batch=1, half=False, workers=0, verbose=False
    )[0]
    Image.fromarray(vehicle_result.plot()[..., ::-1].copy()).save(vehicle_output)

    vehicle_classes = {"car", "motorcycle", "bus", "truck"}
    vehicle_counts = {}
    if vehicle_result.boxes is not None:
        names = vehicle_model.names
        for box in vehicle_result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = (names.get(cls_id, str(cls_id))
                        if isinstance(names, dict) else names[cls_id])
            cls_name = str(cls_name).lower()
            if cls_name in vehicle_classes:
                vehicle_counts[cls_name] = vehicle_counts.get(cls_name, 0) + 1

    # Estimate visible weather appearance using zero-shot CLIP classification.
    weather_classifier = pipeline(
        task="zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        framework="pt",
        device=-1
    )
    weather_candidates = [
        "a road scene in rainy weather with rain and wet road",
        "a road scene in foggy weather with poor visibility",
        "a road scene in clear sunny weather",
        "a road scene in cloudy overcast weather"
    ]
    weather_result = weather_classifier(image, candidate_labels=weather_candidates)
    best_weather = weather_result[0]
    label = str(best_weather["label"]).lower()
    if "rainy" in label:
        weather = "rain"
    elif "foggy" in label:
        weather = "fog"
    else:
        # The DNN's available weather categories do not include cloudy.
        weather = "clear"

    vehicle_total = int(sum(vehicle_counts.values()))
    # A rough single-image vehicle-count proxy, not a traffic-flow measurement.
    if vehicle_total <= 2:
        traffic_density = "low"
    elif vehicle_total <= 6:
        traffic_density = "medium"
    else:
        traffic_density = "high"

    # Approximate visibility from the estimated visual weather cues.
    visibility = "low" if weather == "fog" else ("medium" if weather == "rain" else "high")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "vehicle_counts": vehicle_counts,
            "vehicle_total": vehicle_total,
            "traffic_density": traffic_density,
            "weather": weather,
            "weather_label": str(best_weather["label"]),
            "weather_confidence": float(best_weather["score"]),
            "visibility": visibility
        }, f)
except Exception:
    traceback.print_exc()
    sys.exit(2)
"""
                st.info("Analyzing visible weather and vehicles in an isolated CPU process...")
                with st.spinner(
                    "Running image analysis. First-time model downloads may take several minutes..."
                ):
                    completed = subprocess.run(
                        [
                            sys.executable, "-c", integrated_worker,
                            str(input_path), str(vehicle_output_path), str(metadata_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,
                        env={
                            **os.environ,
                            "CUDA_VISIBLE_DEVICES": "-1",
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1",
                            "TOKENIZERS_PARALLELISM": "false",
                        },
                    )

                if completed.returncode != 0:
                    st.error("Image analysis failed. The main Streamlit app remains protected.")
                    if completed.returncode < 0:
                        st.warning(
                            f"The analysis worker terminated with signal {-completed.returncode}; "
                            "this may indicate a native-library crash or memory limit."
                        )
                    st.code((completed.stderr or completed.stdout or "No diagnostics returned.")[-9000:])
                elif not metadata_path.is_file():
                    st.error("Image analysis finished without metadata output.")
                    st.code((completed.stderr or "No diagnostics returned.")[-9000:])
                else:
                    with metadata_path.open("r", encoding="utf-8") as f:
                        analysis = json.load(f)

                    vehicle_counts = analysis.get("vehicle_counts", {})
                    vehicle_total = int(analysis.get("vehicle_total", 0))
                    inferred_weather = analysis["weather"]
                    inferred_visibility = analysis["visibility"]
                    inferred_traffic = analysis["traffic_density"]

                    st.markdown("### Image analysis")
                    img_col1, img_col2 = st.columns(2)
                    with img_col1:
                        st.markdown("**Vehicle detection**")
                        if vehicle_output_path.is_file():
                            st.image(str(vehicle_output_path), width="stretch")
                        st.metric("Visible vehicles", vehicle_total)
                    with img_col2:
                        st.markdown("**Weather classification**")
                        st.metric("Estimated weather", inferred_weather.title())
                        st.metric("Weather classifier confidence", f"{analysis.get('weather_confidence', 0.0):.1%}")

                    st.markdown("### Automatically inferred conditions")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Weather", inferred_weather.title())
                    m2.metric("Traffic", inferred_traffic.title())
                    m3.metric("Estimated visibility", inferred_visibility.title())
                    if vehicle_counts:
                        st.write("Vehicle counts:", vehicle_counts)
                    st.caption(
                        "The weather label describes visible image appearance and may be incorrect. "
                        "Vehicle counts from a single image are only a rough traffic-density estimate."
                    )

                    # Pass inferred weather and traffic conditions to the existing DNN.
                    # Current time is used because a still image cannot reliably reveal time of day.
                    inference_time = datetime.now().time().replace(second=0, microsecond=0)
                    image_processed_input = preprocess_input(
                        weather=inferred_weather,
                        visibility=inferred_visibility,
                        traffic_density=inferred_traffic,
                        selected_time=inference_time,
                        is_peak_hour=calculate_peak_hour(inference_time.hour),
                        preprocessor=preprocessor,
                        expected_features=EXPECTED_FEATURES,
                    )
                    image_risk_score = predict_risk(model, image_processed_input)
                    image_risk_category = get_risk_category(image_risk_score)
                    category_class = {
                        "Low Risk": "low-risk",
                        "Medium Risk": "medium-risk",
                        "High Risk": "high-risk",
                    }[image_risk_category]

                    st.markdown("### Risk estimate from inferred conditions")
                    st.markdown(
                        f'<div class="result-card"><div class="result-title">'
                        f'Image-Based Risk Score</div>'
                        f'<div class="score">{image_risk_score:.4f}</div>'
                        f'<div class="{category_class}">{image_risk_category}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.metric("Vehicles counted", vehicle_total)

                    st.markdown("### Why this risk was estimated")
                    st.markdown(
                        f"- **Weather:** the image classifier estimated **{inferred_weather}** conditions. "
                        "Rain or fog may affect traction or visibility."
                    )
                    st.markdown(
                        f"- **Traffic:** {vehicle_total} car/bus/truck/motorcycle instance(s) were visible, "
                        f"mapped to **{inferred_traffic}** traffic density for this estimate."
                    )
                    st.markdown(
                        f"- **Visibility:** the application approximated visibility as **{inferred_visibility}** "
                        "from the image's weather classification."
                    )
                    st.markdown(
                        f"- **Time:** the DNN used the current time ({inference_time.strftime('%H:%M')}) "
                        "because a single image cannot reliably reveal the time of day."
                    )
                    st.warning(
                        "This is an experimental estimate. Weather classification and vehicle detection can be wrong, "
                        "and the DNN was trained on structured inputs rather than raw images. The inferred conditions "
                        "and thresholds need validation against labelled real-world data before practical use."
                    )

        except subprocess.TimeoutExpired:
            st.error("Image analysis exceeded the 10-minute time limit. First-time model downloads may be the cause.")
        except Exception as exc:
            st.error("Image-based risk prediction could not be completed.")
            st.exception(exc)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    '<div class="footer-text">DL-Based Road Accident Risk Prediction | '
    'Deep Learning + Explainable AI</div>',
    unsafe_allow_html=True,
)
