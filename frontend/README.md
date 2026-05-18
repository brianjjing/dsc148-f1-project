# Front end — F1 DNF Predictor

Streamlit dashboard for the DSC148 DNF prediction project. Currently a **look-only MVP** — predictions come from a stub in `predict_dnf()` and metrics/sample tables are mock data. Swap them out once the model artifact is frozen (see `DEADLINES.md`, May 30 I/O contract).

## Run

From the project root:

```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```

The app opens at http://localhost:8501.

## Wiring the real model

Replace the body of `predict_dnf(features)` in `app.py` with:

```python
import joblib
model = joblib.load("path/to/model.joblib")
X = build_feature_row(features)        # match the frozen feature column list
prob = float(model.predict_proba(X)[0, 1])
return {"probability": prob, "contributions": top_contributions(model, X)}
```

Update the **Model Performance** tab tables with the real metrics from the May 23 frozen run, and the **Dataset Overview** numbers from the joined `results × status × races` table.
