
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

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
MODEL_PATH = MODEL_DIR / "improved_model.keras"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


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
