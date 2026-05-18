"""
F1 DNF Prediction — Frontend MVP (look-only).

Wire the real model in `predict_dnf()` once the artifact is frozen
(see DEADLINES.md, May 30 I/O freeze).
"""

import random
from datetime import datetime

import pandas as pd
import streamlit as st

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

# ---------- Mock reference data ----------
DRIVERS = [
    "Lewis Hamilton", "Max Verstappen", "Charles Leclerc", "Sergio Perez",
    "Carlos Sainz", "Lando Norris", "George Russell", "Fernando Alonso",
    "Oscar Piastri", "Pierre Gasly", "Esteban Ocon", "Valtteri Bottas",
]
CONSTRUCTORS = [
    "Mercedes", "Red Bull", "Ferrari", "McLaren", "Aston Martin",
    "Alpine", "Williams", "Alfa Romeo", "Haas", "AlphaTauri",
]
CIRCUITS = [
    "Monza", "Silverstone", "Monaco", "Spa-Francorchamps", "Suzuka",
    "Interlagos", "Bahrain", "Circuit de Catalunya", "Hungaroring", "COTA",
]
WEATHERS = ["Dry", "Light Rain", "Heavy Rain", "Mixed"]


# ---------- Mock model ----------
def predict_dnf(features: dict) -> dict:
    """Stub. Replace with real model.predict_proba once artifact is frozen."""
    random.seed(hash(tuple(sorted(features.items()))) & 0xFFFFFFFF)
    base = 0.18
    if features["weather"] == "Heavy Rain":
        base += 0.22
    elif features["weather"] == "Light Rain":
        base += 0.10
    elif features["weather"] == "Mixed":
        base += 0.14
    base += max(0, features["grid_position"] - 10) * 0.012
    base += features["laps_planned"] / 1000
    base -= features["driver_experience"] * 0.003
    base += random.uniform(-0.05, 0.05)
    prob = float(max(0.02, min(0.95, base)))

    contributions = [
        ("Weather: " + features["weather"], 0.22 if "Rain" in features["weather"] else 0.05),
        (f"Grid position #{features['grid_position']}", max(0, features["grid_position"] - 10) * 0.012),
        (f"Constructor: {features['constructor']}", random.uniform(0.02, 0.12)),
        (f"Circuit: {features['circuit']}", random.uniform(0.01, 0.09)),
        ("Driver experience", -features["driver_experience"] * 0.003),
    ]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"probability": prob, "contributions": contributions}


def risk_pill(prob: float) -> str:
    if prob < 0.25:
        return '<span class="pill pill-low">LOW RISK</span>'
    if prob < 0.5:
        return '<span class="pill pill-med">MEDIUM RISK</span>'
    return '<span class="pill pill-high">HIGH RISK</span>'


# ---------- Sidebar — input form ----------
with st.sidebar:
    st.markdown("### 🏁 Race Inputs")
    st.caption("Configure the entry to score for DNF probability.")

    driver = st.selectbox("Driver", DRIVERS, index=1)
    constructor = st.selectbox("Constructor", CONSTRUCTORS, index=1)
    circuit = st.selectbox("Circuit", CIRCUITS, index=0)
    season = st.slider("Season", 1990, 2024, 2023)
    grid_position = st.number_input("Grid position", min_value=1, max_value=22, value=5)
    laps_planned = st.number_input("Planned laps", min_value=40, max_value=80, value=58)
    driver_experience = st.slider("Driver experience (race starts)", 0, 350, 120)
    weather = st.selectbox("Weather", WEATHERS, index=0)
    qualifying_gap = st.number_input("Qualifying gap to pole (s)", value=0.45, step=0.05, format="%.2f")

    st.markdown("---")
    run_pred = st.button("🚦 PREDICT DNF", use_container_width=True)


# ---------- Header ----------
st.markdown('<div class="f1-stripe"></div>', unsafe_allow_html=True)
col_a, col_b = st.columns([0.7, 0.3])
with col_a:
    st.markdown("# 🏎️ F1 DNF Predictor")
    st.caption(
        "DSC148 group project — predicting whether a driver will fail to finish a race. "
        "Built on the Formula 1 World Championship dataset (1950–2020)."
    )
with col_b:
    st.markdown(
        f"<div class='metric-card'><h3>Last Updated</h3>"
        f"<div class='v' style='font-size:1.1rem'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</div></div>",
        unsafe_allow_html=True,
    )


# ---------- Tabs ----------
tab_pred, tab_model, tab_data, tab_about = st.tabs(
    ["Prediction", "Model Performance", "Dataset Overview", "About"]
)

# === Prediction tab ===
with tab_pred:
    features = {
        "driver": driver,
        "constructor": constructor,
        "circuit": circuit,
        "season": season,
        "grid_position": grid_position,
        "laps_planned": laps_planned,
        "driver_experience": driver_experience,
        "weather": weather,
        "qualifying_gap": qualifying_gap,
    }

    result = predict_dnf(features) if run_pred else predict_dnf(features)
    prob = result["probability"]

    left, right = st.columns([0.55, 0.45])

    with left:
        st.subheader("Prediction")
        st.markdown(
            f"<div class='metric-card' style='padding:1.6rem'>"
            f"<h3>DNF probability for {driver} — {circuit} {season}</h3>"
            f"<div class='v' style='font-size:3.4rem'>{prob*100:.1f}%</div>"
            f"{risk_pill(prob)}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(prob)

        st.markdown("#### Top Contributing Factors")
        contrib_df = pd.DataFrame(
            result["contributions"],
            columns=["Factor", "Δ probability"],
        )
        st.dataframe(
            contrib_df.style.format({"Δ probability": "{:+.3f}"}),
            use_container_width=True, hide_index=True,
        )

    with right:
        st.subheader("Entry Summary")
        summary = pd.DataFrame(
            [
                ("Driver", driver),
                ("Constructor", constructor),
                ("Circuit", circuit),
                ("Season", season),
                ("Grid", f"P{grid_position}"),
                ("Planned laps", laps_planned),
                ("Experience", f"{driver_experience} starts"),
                ("Weather", weather),
                ("Quali gap to pole", f"+{qualifying_gap:.2f}s"),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("#### Historical DNF rate")
        hist = pd.DataFrame(
            {
                "Season": list(range(2014, 2024)),
                "DNF rate": [0.21, 0.19, 0.22, 0.17, 0.18, 0.16, 0.14, 0.17, 0.15, 0.16],
            }
        ).set_index("Season")
        st.line_chart(hist, height=200)


# === Model Performance tab ===
with tab_model:
    st.subheader("Model Comparison")
    st.caption("Snapshot from the latest training run. Replace with frozen v1 metrics on May 23.")

    m1, m2, m3, m4 = st.columns(4)
    for col, name, val in [
        (m1, "AUC (best)", "0.812"),
        (m2, "F1 (best)", "0.541"),
        (m3, "Precision", "0.598"),
        (m4, "Recall", "0.495"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-card'><h3>{name}</h3><div class='v'>{val}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Metrics by model")
    metrics_df = pd.DataFrame(
        {
            "Model": ["Baseline (logistic)", "Random Forest", "Gradient Boosting", "XGBoost (tuned)"],
            "AUC": [0.701, 0.778, 0.795, 0.812],
            "F1": [0.402, 0.490, 0.521, 0.541],
            "Precision": [0.481, 0.553, 0.580, 0.598],
            "Recall": [0.346, 0.440, 0.474, 0.495],
        }
    )
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("#### AUC by season (time-based split)")
    auc_df = pd.DataFrame(
        {
            "Season": list(range(2016, 2024)),
            "Baseline": [0.68, 0.69, 0.70, 0.70, 0.71, 0.71, 0.70, 0.70],
            "XGBoost":  [0.74, 0.76, 0.78, 0.79, 0.80, 0.81, 0.81, 0.81],
        }
    ).set_index("Season")
    st.line_chart(auc_df, height=260)


# === Dataset tab ===
with tab_data:
    st.subheader("Dataset Overview")
    st.caption("Kaggle — Formula 1 World Championship (1950–2020), Rohan Rao.")

    c1, c2, c3, c4 = st.columns(4)
    for col, name, val in [
        (c1, "Races", "1,079"),
        (c2, "Drivers", "857"),
        (c3, "Results rows", "25,840"),
        (c4, "Seasons", "1950–2020"),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-card'><h3>{name}</h3><div class='v'>{val}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### DNF distribution by status")
    status_df = pd.DataFrame(
        {
            "Status": ["Finished", "Engine", "Collision", "Gearbox", "Hydraulics", "Accident", "Electrical", "Other"],
            "Count":  [18020, 1620, 1180, 880, 540, 760, 690, 2150],
        }
    ).set_index("Status")
    st.bar_chart(status_df, height=280)

    st.markdown("#### Sample of joined `results × status × races`")
    sample = pd.DataFrame(
        {
            "season": [2022, 2022, 2021, 2021, 2020],
            "race": ["Monza", "Silverstone", "Spa", "Monaco", "Suzuka"],
            "driver": ["M. Verstappen", "L. Hamilton", "C. Leclerc", "S. Perez", "F. Alonso"],
            "grid": [7, 5, 4, 8, 14],
            "status": ["Finished", "Finished", "Hydraulics", "Finished", "Collision"],
            "dnf": [0, 0, 1, 0, 1],
        }
    )
    st.dataframe(sample, use_container_width=True, hide_index=True)


# === About tab ===
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **Course:** DSC148 (UCSD, Spring 2026)
        **Goal:** Binary classification — predict whether a driver **did not finish (DNF)** a race.
        **Data:** Formula 1 World Championship dataset (Kaggle, Rohan Rao).

        **Front-end status:** MVP shell (this app). Real model is wired in `predict_dnf()`
        once the `model.joblib` artifact + feature column list are frozen
        (see `DEADLINES.md`, May 30 I/O contract).

        **Deliverable mapping (DEADLINES.md):**
        - ✅ May 30 — FE MVP wired to agreed I/O contract (in progress)
        - ⏳ June 6 — FE uses final model path
        - ⏳ June 7 — Submit
        """
    )
    st.info("Replace mock metrics and sample rows with the frozen model output before June 6 integration.")


# ---------- Footer ----------
st.markdown(
    '<div class="footer">DSC148 · F1 DNF Prediction · Frontend MVP</div>',
    unsafe_allow_html=True,
)
