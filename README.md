---
title: F1 DNF Predictor
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: streamlit
app_file: frontend/app.py
pinned: false
---

# Formula 1 DNF Prediction

**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/EricGan64/F1-project)

This project uses historical Formula 1 race data to build a model that predicts whether a driver **did not finish (DNF)** a race.

## Dataset

**Source:** [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=status.csv) on Kaggle (Rohan Rao).

The dataset is tabular F1 history from 1950 through 2020, split into multiple CSV files (for example races, drivers, constructors, results, lap times, pit stops, and qualifying).

For DNF work, `results.csv` (per-driver race outcome) and `status.csv` (finish status and failure-related labels) are the main tables for defining labels and joining to races, drivers, and constructors for features.

## Project goal

Train a **prediction model for DNF** — treated here as binary classification (finished the race vs did not finish). Label construction, feature engineering, and model choice are defined in the project notebook or code.

## Setup

1. **Install Dependencies:**
   From the project root, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Obtain the Dataset:**
   * **Automatic:** Open [exploration_template.ipynb](exploration_template.ipynb) and run the setup cells to automatically download and extract the dataset from Google Drive into a `data/` folder.
   * **Manual:** Download the [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) dataset from Kaggle and place the CSV files inside a folder named `data/` at the project root.

## Running the Notebooks

For imports and paths to resolve correctly, always launch Jupyter or start your notebook kernel with the **project root directory** (`dsc148-f1-project/`) as the working directory.

* **Exploratory Data Analysis:** Open and run [exploration_template.ipynb](exploration_template.ipynb) to start exploring the CSVs and verifying the data structure.
* **Model Training & Tuning:** Open and run [model/all_models.ipynb](model/all_models.ipynb) to train baseline models, run the Optuna hyperparameter optimization, and view the evaluation visualizations.

*(Optional)* You can also run the entire training, tuning, and feature ablation workflow as a standalone script in your terminal by executing:
```bash
python model/train_models.py
```