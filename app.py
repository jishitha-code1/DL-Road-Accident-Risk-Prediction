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
DAMAGE_MODEL_PATH = MODEL_DIR / "best.pt"

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
