# Frontend — F1 DNF Predictor Dashboard

Streamlit dashboard for the Formula 1 DNF prediction project. It loads the pre-trained LightGBM model (`best_model.joblib`) and queries the local F1 CSV files (`data/`) to calculate driver age and rolling reliability features on-the-fly, producing real-time risk predictions.

## Run

From the project root:

```bash
python -m streamlit run frontend/app.py
```

The app will start a local server and open at `http://localhost:8501`.

## Features
- **Real-time Inference:** Accepts user selections for driver, constructor, track, grid, and year, and delivers classification predictions ("Finished" vs "DNF") with a confidence score.
- **Dynamic Feature Preprocessing:** Automatically maps selected names to dataset IDs, computes age from DOB, and derives rolling DNF rates (past 5 and 10 races) from the historical dataset.
- **Model Comparisons:** Displays test set metrics (Accuracy, F1-Score, Precision, Recall, AUC) across various classifiers and plots native LightGBM feature importances.
- **Dataset Overview:** Provides dataset statistics (drivers, constructors, results counts) and distributions of failure causes.
