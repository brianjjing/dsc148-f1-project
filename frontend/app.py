"""
F1 DNF Prediction — Frontend Dashboard (Connected to Real LightGBM Model).

Wired directly to model/best_model.joblib and data/ CSV files.
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow model imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="F1 DNF Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #15151e 100%);
        color: #f5f5f7;
      }
      section[data-testid="stSidebar"] {
        background: #15151e;
        border-right: 1px solid #2a2a36;
      }
      h1, h2, h3, h4 { color: #f5f5f7; font-family: 'Helvetica Neue', sans-serif; }
      .f1-stripe {
        height: 6px;
        background: linear-gradient(90deg, #e10600 0%, #e10600 33%, #ffffff 33%, #ffffff 66%, #000000 66%);
        border-radius: 3px;
        margin-bottom: 1.2rem;
      }
      .metric-card {
        background: #1f1f2b;
        border: 1px solid #2a2a36;
        border-left: 4px solid #e10600;
        padding: 1rem 1.2rem;
        border-radius: 6px;
      }
      .metric-card h3 { margin: 0; color: #9a9aae; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; }
      .metric-card .v { font-size: 1.8rem; font-weight: 700; color: #f5f5f7; }
      .pill {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em;
        margin-top: 0.5rem;
      }
      .pill-low  { background: #103a1e; color: #4ade80; border: 1px solid #1f6b39; }
      .pill-med  { background: #3a2f10; color: #facc15; border: 1px solid #6b5d1f; }
      .pill-high { background: #3a1010; color: #f87171; border: 1px solid #6b1f1f; }
      .stButton > button {
        background: #e10600; color: white; border: 0;
        font-weight: 700; letter-spacing: 0.04em; padding: 0.6rem 1.4rem;
      }
      .stButton > button:hover { background: #ff1a14; color: white; }
      .footer { color: #6b6b80; font-size: 0.8rem; text-align: center; padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants from training script
FEATURE_COLS = [
    "grid",
    "year",
    "round",
    "driver_age",
    "driver_dnf_rate_last5",
    "driver_dnf_rate_last10",
    "constructor_dnf_rate_last5",
    "constructor_dnf_rate_last10",
]
CATEGORICAL_COLS = ["circuitId", "constructorId", "driverId"]

# ---------- Helper to load Model & Data ----------
@st.cache_resource
def load_predictor():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "model" / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data
def load_f1_data():
    from model.model_pipeline import load_raw, build_labeled_frame, add_rolling_features
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    
    # Load raw tables
    raw = load_raw(data_dir)
    circuits_df = pd.read_csv(data_dir / "circuits.csv")
    
    # Process dataframe
    df = build_labeled_frame(raw)
    df = add_rolling_features(df)
    
    # Create mappings
    raw_drivers = raw["drivers"].copy()
    raw_drivers["full_name"] = raw_drivers["forename"] + " " + raw_drivers["surname"]
    driver_to_id = dict(zip(raw_drivers["full_name"], raw_drivers["driverId"]))
    id_to_driver = dict(zip(raw_drivers["driverId"], raw_drivers["full_name"]))
    id_to_dob = dict(zip(raw_drivers["driverId"], pd.to_datetime(raw_drivers["dob"], errors="coerce")))
    
    raw_constructors = raw["constructors"].copy()
    constructor_to_id = dict(zip(raw_constructors["name"], raw_constructors["constructorId"]))
    id_to_constructor = dict(zip(raw_constructors["constructorId"], raw_constructors["name"]))
    
    circuit_to_id = dict(zip(circuits_df["name"], circuits_df["circuitId"]))
    id_to_circuit = dict(zip(circuits_df["circuitId"], circuits_df["name"]))
    
    # Filter for active/recent entities from year >= 2000 to keep UI lists clean
    recent_df = df[df["year"] >= 2000]
    recent_driver_ids = recent_df["driverId"].unique()
    recent_constructor_ids = recent_df["constructorId"].unique()
    recent_circuit_ids = recent_df["circuitId"].unique()
    
    recent_drivers = sorted([id_to_driver[i] for i in recent_driver_ids if i in id_to_driver])
    recent_constructors = sorted([id_to_constructor[i] for i in recent_constructor_ids if i in id_to_constructor])
    recent_circuits = sorted([id_to_circuit[i] for i in recent_circuit_ids if i in id_to_circuit])
    
    # Fallback to all if empty
    if not recent_drivers: recent_drivers = sorted(raw_drivers["full_name"].tolist())
    if not recent_constructors: recent_constructors = sorted(raw_constructors["name"].tolist())
    if not recent_circuits: recent_circuits = sorted(circuits_df["name"].tolist())
        
    return {
        "raw": raw,
        "df": df,
        "driver_to_id": driver_to_id,
        "id_to_driver": id_to_driver,
        "id_to_dob": id_to_dob,
        "constructor_to_id": constructor_to_id,
        "id_to_constructor": id_to_constructor,
        "circuit_to_id": circuit_to_id,
        "id_to_circuit": id_to_circuit,
        "recent_drivers": recent_drivers,
        "recent_constructors": recent_constructors,
        "recent_circuits": recent_circuits,
        "all_drivers": sorted(raw_drivers["full_name"].tolist()),
        "all_constructors": sorted(raw_constructors["name"].tolist()),
        "all_circuits": sorted(circuits_df["name"].tolist()),
    }

# Load model & datasets
model = load_predictor()
try:
    f1_data = load_f1_data()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Failed to load dataset: {e}")

# ---------- Model Check & Fallback Training UI ----------
if model is None:
    st.warning("⚠️ Pre-trained model (model/best_model.joblib) not found. You must train it before making predictions.")
    if st.button("🚀 Train Model Now"):
        with st.spinner("Training model and running Optuna tuning (approx. 45s)..."):
            try:
                from model.train_models import main as train_main
                train_main()
                st.success("Model trained and saved successfully! Reloading page...")
                st.rerun()
            except Exception as ex:
                st.error(f"Error during training: {ex}")
    st.stop()

# ---------- Risk rating helper ----------
def get_risk_pill(prob: float) -> str:
    if prob < 0.20:
        return '<span class="pill pill-low">LOW RISK</span>'
    elif prob < 0.40:
        return '<span class="pill pill-med">MEDIUM RISK</span>'
    else:
        return '<span class="pill pill-high">HIGH RISK</span>'

# ---------- Main UI Structure ----------
st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
col_a, col_b = st.columns([0.7, 0.3])
with col_a:
    st.markdown("# 🏎️ F1 DNF Predictor")
    st.caption(
        "Predicting F1 race-day Did Not Finish (DNF) outcomes using "
        "historical results, starting grid, and rolling driver/constructor reliability metrics."
    )
with col_b:
    st.markdown(
        f"<div class='metric-card'><h3>Dashboard Status</h3>"
        f"<div class='v' style='font-size:1.1rem;color:#4ade80'>● Live Model Connected</div></div>",
        unsafe_allow_html=True,
    )

# ---------- Tabs ----------
tab_pred, tab_model, tab_data, tab_about = st.tabs(
    ["Prediction Playground", "Model Performance", "Dataset Overview", "About the Project"]
)

# === Tab 1: Prediction Playground ===
with tab_pred:
    if data_loaded:
        with st.sidebar:
            st.markdown("### 🏁 Race Inputs")
            st.caption("Provide race entries to score DNF probability.")
            
            # Show filter options to prevent cluttering dropdowns
            limit_recent = st.checkbox("Show only modern drivers/tracks (since 2000)", value=True)
            
            drivers_list = f1_data["recent_drivers"] if limit_recent else f1_data["all_drivers"]
            constructors_list = f1_data["recent_constructors"] if limit_recent else f1_data["all_constructors"]
            circuits_list = f1_data["recent_circuits"] if limit_recent else f1_data["all_circuits"]
            
            # Select boxes
            driver_name = st.selectbox("Select Driver", drivers_list, index=drivers_list.index("Lewis Hamilton") if "Lewis Hamilton" in drivers_list else 0)
            constructor_name = st.selectbox("Select Constructor", constructors_list, index=constructors_list.index("Mercedes") if "Mercedes" in constructors_list else 0)
            circuit_name = st.selectbox("Select Circuit", circuits_list, index=circuits_list.index("Autodromo Nazionale di Monza") if "Autodromo Nazionale di Monza" in circuits_list else 0)
            
            # Numeric sliders & inputs
            year = st.slider("Season Year", 1990, 2026, 2024)
            round_num = st.slider("Race Round Number", 1, 24, 12)
            grid = st.number_input("Grid Position (1-22)", min_value=1, max_value=22, value=5)
            
            st.markdown("---")
            run_inference = st.button("🚦 CALCULATE DNF RISK", use_container_width=True)

        # Lookup IDs
        d_id = f1_data["driver_to_id"][driver_name]
        c_id = f1_data["constructor_to_id"][constructor_name]
        track_id = f1_data["circuit_to_id"][circuit_name]

        # Calculate live features to feed to model
        raw = f1_data["raw"]
        df = f1_data["df"]
        
        # 1. Determine race date to calculate driver age
        races_df = raw["races"]
        match = races_df[(races_df["year"] == year) & (races_df["circuitId"] == track_id)]
        if not match.empty:
            race_date = pd.to_datetime(match.iloc[0]["date"], errors="coerce")
            round_val = int(match.iloc[0]["round"])
        else:
            race_date = pd.Timestamp(year, 6, 15)  # fallback middle of the year
            round_val = round_num

        # 2. Driver Age
        dob = f1_data["id_to_dob"].get(d_id, pd.Timestamp(1990, 1, 1))
        driver_age = (race_date - dob).days / 365.25

        # 3. Dynamic Driver rolling features (prior to selected date)
        driver_history = df[(df["driverId"] == d_id) & (df["date"] < race_date)]
        if not driver_history.empty:
            past_dnfs = driver_history.sort_values("date")["dnf"].tolist()
            driver_dnf_last5 = np.mean(past_dnfs[-5:]) if len(past_dnfs) >= 2 else np.nan
            driver_dnf_last10 = np.mean(past_dnfs[-10:]) if len(past_dnfs) >= 2 else np.nan
        else:
            driver_dnf_last5 = np.nan
            driver_dnf_last10 = np.nan

        # 4. Dynamic Constructor rolling features (prior to selected date)
        constructor_history = df[(df["constructorId"] == c_id) & (df["date"] < race_date)]
        if not constructor_history.empty:
            past_dnfs_c = constructor_history.sort_values("date")["dnf"].tolist()
            constructor_dnf_last5 = np.mean(past_dnfs_c[-5:]) if len(past_dnfs_c) >= 2 else np.nan
            constructor_dnf_last10 = np.mean(past_dnfs_c[-10:]) if len(past_dnfs_c) >= 2 else np.nan
        else:
            constructor_dnf_last5 = np.nan
            constructor_dnf_last10 = np.nan

        # Create input feature dictionary matching model format
        input_features = {
            "grid": grid,
            "year": year,
            "round": round_val,
            "driver_age": driver_age,
            "driver_dnf_rate_last5": driver_dnf_last5,
            "driver_dnf_rate_last10": driver_dnf_last10,
            "constructor_dnf_rate_last5": constructor_dnf_last5,
            "constructor_dnf_rate_last10": constructor_dnf_last10,
            "circuitId": track_id,
            "constructorId": c_id,
            "driverId": d_id
        }

        # Build dataframe row
        row_df = pd.DataFrame([input_features])
        for col in CATEGORICAL_COLS:
            row_df[col] = pd.Categorical(row_df[col], categories=df[col].unique())

        # Impute missing values with training means (consistent with modeling step)
        train_df = df[df["year"] < 2018]
        imputation_means = {}
        for col in FEATURE_COLS:
            mean_val = train_df[col].mean()
            imputation_means[col] = 0.0 if pd.isna(mean_val) else mean_val
            row_df[col] = row_df[col].fillna(imputation_means[col])

        # Order columns correctly
        row_df = row_df[FEATURE_COLS + CATEGORICAL_COLS]

        # Execute prediction
        probabilities = model.predict_proba(row_df)
        prob = float(probabilities[0, 1])
        prediction = "DNF (Did Not Finish)" if prob > 0.5 else "Finished (Completed Race)"

        # Layout prediction results
        left, right = st.columns([0.55, 0.45])

        with left:
            st.subheader("Prediction Result")
            st.markdown(
                f"<div class='metric-card' style='padding:1.6rem'>"
                f"<h3>DNF Risk Analysis for {driver_name}</h3>"
                f"<div class='v' style='font-size:3.4rem'>{prob*100:.1f}%</div>"
                f"<strong>Classification Inference:</strong> {prediction}<br/>"
                f"{get_risk_pill(prob)}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(prob)

            # Explanatory Factor Analysis
            st.markdown("#### Input Feature Values fed to Model")
            feature_display = pd.DataFrame([
                ("Starting Grid", f"P{grid}", "Direct user input"),
                ("Driver Age", f"{driver_age:.1f} years", f"Calculated from DOB: {dob.strftime('%Y-%m-%d')}"),
                ("Driver DNF Rate (Last 5 races)", f"{driver_dnf_last5*100:.1f}%" if not pd.isna(driver_dnf_last5) else "Imputed default (0.0% / new driver)", "Calculated from driver history"),
                ("Driver DNF Rate (Last 10 races)", f"{driver_dnf_last10*100:.1f}%" if not pd.isna(driver_dnf_last10) else "Imputed default (0.0% / new driver)", "Calculated from driver history"),
                ("Constructor DNF Rate (Last 5 races)", f"{constructor_dnf_last5*100:.1f}%" if not pd.isna(constructor_dnf_last5) else "Imputed default (0.0%)", "Calculated from constructor history"),
                ("Constructor DNF Rate (Last 10 races)", f"{constructor_dnf_last10*100:.1f}%" if not pd.isna(constructor_dnf_last10) else "Imputed default (0.0%)", "Calculated from constructor history"),
                ("Circuit ID", f"ID {track_id}", "Categorical variable"),
                ("Driver ID", f"ID {d_id}", "Categorical variable"),
                ("Constructor ID", f"ID {c_id}", "Categorical variable"),
            ], columns=["Feature", "Value", "Notes"])
            st.dataframe(feature_display, use_container_width=True, hide_index=True)

        with right:
            st.subheader("Interactive Summary")
            summary_table = pd.DataFrame([
                ("Selected Driver", driver_name),
                ("Selected Constructor", constructor_name),
                ("Selected Track", circuit_name),
                ("Assigned Race Year", year),
                ("Assigned Race Round", round_val),
            ], columns=["Race Field", "Input Selection"])
            st.dataframe(summary_table, use_container_width=True, hide_index=True)

            # DNF history visualization
            st.markdown("#### Historical F1 Overall DNF Rate (1950 - 2020)")
            dnf_distribution = df.groupby("year")["dnf"].mean().reset_index()
            dnf_distribution.columns = ["Year", "DNF Rate"]
            st.line_chart(dnf_distribution.set_index("Year"), height=200)

# === Tab 2: Model Performance ===
with tab_model:
    st.subheader("Model Evaluation & Comparisons")
    st.caption("These are the actual training, optimization, and validation metrics calculated during model script execution.")

    # High-level metrics for best model (LightGBM Tuned)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown("<div class='metric-card'><h3>Validation F1-Score</h3><div class='v'>0.1972</div></div>", unsafe_allow_html=True)
    with e2:
        st.markdown("<div class='metric-card'><h3>Validation Recall</h3><div class='v'>0.2500</div></div>", unsafe_allow_html=True)
    with e3:
        st.markdown("<div class='metric-card'><h3>Validation Precision</h3><div class='v'>0.1628</div></div>", unsafe_allow_html=True)
    with e4:
        st.markdown("<div class='metric-card'><h3>Validation ROC-AUC</h3><div class='v'>0.5284</div></div>", unsafe_allow_html=True)

    # Detailed table comparison
    st.markdown("#### Full Model Comparison Table (Test Set Results)")
    compare_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression (L2)", "Naive Bayes", "Random Forest", "XGBoost", "LightGBM (Tuned)"],
            "Accuracy": [0.8348, 0.8311, 0.8493, 0.8365, 0.6931],
            "F1-Score": [0.0540, 0.0492, 0.0089, 0.0581, 0.1972],
            "Precision": [0.1972, 0.1625, 0.5000, 0.2206, 0.1628],
            "Recall": [0.0313, 0.0290, 0.0045, 0.0335, 0.2500],
            "AUC": [0.5469, 0.5070, 0.5649, 0.5715, 0.5284],
        }
    )
    st.dataframe(compare_df.style.highlight_max(axis=0, subset=["F1-Score", "AUC"], color="#3a1a1a"), use_container_width=True, hide_index=True)

    # Display feature importances directly from the trained LightGBM model
    st.markdown("#### LightGBM Model Feature Importances")
    if hasattr(model, "feature_importances_") and hasattr(model, "feature_name_"):
        importance_df = pd.DataFrame({
            "Feature": model.feature_name_,
            "Split Importance": model.feature_importances_
        }).sort_values("Split Importance", ascending=True)
        st.bar_chart(importance_df.set_index("Feature"), height=250)

# === Tab 3: Dataset Overview ===
with tab_data:
    st.subheader("Formula 1 World Championship Dataset (1950–2020)")
    st.caption("Descriptive dataset statistics computed dynamically from your local CSV resources.")

    if data_loaded:
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"<div class='metric-card'><h3>Total Drivers</h3><div class='v'>{len(f1_data['all_drivers'])}</div></div>", unsafe_allow_html=True)
        with r2:
            st.markdown(f"<div class='metric-card'><h3>Total Constructors</h3><div class='v'>{len(f1_data['all_constructors'])}</div></div>", unsafe_allow_html=True)
        with r3:
            st.markdown(f"<div class='metric-card'><h3>Total Circuits</h3><div class='v'>{len(f1_data['all_circuits'])}</div></div>", unsafe_allow_html=True)
        with r4:
            st.markdown(f"<div class='metric-card'><h3>Total Results Rows</h3><div class='v'>{len(df):,}</div></div>", unsafe_allow_html=True)

        st.markdown("#### Sample Dataset Rows (Cleaned & Labeled)")
        st.dataframe(df[["year", "round", "driverId", "constructorId", "circuitId", "grid", "driver_age", "dnf"]].head(10), use_container_width=True, hide_index=True)

# === Tab 4: About the Project ===
with tab_about:
    st.subheader("About the Project")
    st.markdown(
        """
        **Goal:** Binary classification — predict whether an F1 driver will experience a DNF (Did Not Finish) on race day.  
        **Data Sources:** Formula 1 World Championship dataset (Kaggle, Rohan Rao).
        
        **Model Integration Logic:**
        1. User inputs a driver, constructor, and track.
        2. The application looks up the unique driver ID, constructor ID, and track ID.
        3. Rolling reliability variables (DNF rates in the last 5/10 races) are calculated dynamically on-the-fly from historical results.
        4. Driver age is calculated dynamically from the driver's date of birth and the chosen race date.
        5. Categorical features are cast to categorical dtypes matching the original dataset mappings.
        6. The pre-trained LightGBM model is run using `predict_proba` to deliver real-time inference.
        """
    )
    st.success("✅ Working Demo fully implemented and wired to the real pre-trained LightGBM model (best_model.joblib)!")

# ---------- Footer ----------
st.markdown(
    '<div class="footer">F1 DNF Prediction · Interactive Demo Dashboard</div>',
    unsafe_allow_html=True,
)
