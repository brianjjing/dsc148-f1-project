# DSC148 — Formula 1 DNF prediction

This project uses historical Formula 1 race data to build a model that predicts whether a driver **did not finish (DNF)** a race.

## Dataset

**Source:** [Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020?select=status.csv) on Kaggle (Rohan Rao).

The dataset is tabular F1 history from 1950 through 2020, split into multiple CSV files (for example races, drivers, constructors, results, lap times, pit stops, and qualifying).

For DNF work, `results.csv` (per-driver race outcome) and `status.csv` (finish status and failure-related labels) are the main tables for defining labels and joining to races, drivers, and constructors for features.

## Project goal

Train a **prediction model for DNF** — treated here as binary classification (finished the race vs did not finish). Label construction, feature engineering, and model choice are defined in the project notebook or code.

## Setup

Download the dataset from Kaggle and place the CSV files in a local folder (for example `data/`).

From the project root, run `pip install -r requirements.txt` and open `notebooks/exploration_template.ipynb` to start exploring the CSVs (use the project root as the notebook working directory, or the notebook resolves `data/` automatically).
