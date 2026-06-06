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
**Read the Paper:** [Project Report (PDF)](https://cas-bridge.xethub.hf.co/xet-bridge-us/6a2242d4a37a93bb548e7c01/8defabaa08020771edf8dd34cf19ad9a8f761b0274e7ffa95a123d7bfd7265be?Expires=1780766178&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jYXMtYnJpZGdlLnhldGh1Yi5oZi5jby94ZXQtYnJpZGdlLXVzLzZhMjI0MmQ0YTM3YTkzYmI1NDhlN2MwMS84ZGVmYWJhYTA4MDIwNzcxZWRmOGRkMzRjZjE5YWQ5YThmNzYxYjAyNzRlN2ZmYTk1YTEyM2Q3YmZkNzI2NWJlKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc4MDc2NjE3OH19fV19&Signature=MEQCIA7LaqDGCIpogSA8O4%7EN6T%7ElwcfVgGwxyP9dHYwuQvdQAiAZWULdg2AD3vCVoLO0TAb0AIcYv2mEOoeBGhpyCM-RvQ__&Key-Pair-Id=K1LYXO563TGWFU&X-Xet-Cas-Uid=public&response-content-disposition=inline%3B+filename*%3DUTF-8%27%27dsc148_v2.pdf%3B+filename%3D%22dsc148_v2.pdf%22%3B&response-content-type=application%2Fpdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=cas%2F20260606%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260606T161618Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=57d6a8423f18ad8d8d7b2f01789afb90457aaaa04dfacd8fc3f272529363a81a)

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