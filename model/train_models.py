import os
import sys
from pathlib import Path
import warnings
import json
import joblib

# Add project root to sys.path to allow imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings("ignore")

# Import the dataset builder from the existing pipeline
from model.model_pipeline import build_dataset

def preprocess_data(data):
    # Impute missing values in continuous features with mean from training data
    num_cols = [c for c in data.feature_cols if c not in data.categorical_cols]
    
    # We will do this via sklearn pipeline or pandas
    X_train = data.X_train.copy()
    X_test = data.X_test.copy()
    
    # Simple pandas imputation for simplicity across different models
    for col in num_cols:
        mean_val = X_train[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        X_train[col] = X_train[col].fillna(mean_val)
        X_test[col] = X_test[col].fillna(mean_val)
        
    return X_train, data.y_train, X_test, data.y_test

def train_and_eval_logistic(X_train, y_train, X_test, y_test, categorical_cols):
    # Logistic regression requires scaling and one-hot encoding
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42, class_weight='balanced'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

def train_and_eval_naive_bayes(X_train, y_train, X_test, y_test, categorical_cols):
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GaussianNB())
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

def train_and_eval_rf(X_train, y_train, X_test, y_test, categorical_cols):
    # Tree models can use OrdinalEncoder for simplicity and speed
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

def train_and_eval_xgb(X_train, y_train, X_test, y_test, categorical_cols):
    num_cols = [c for c in X_train.columns if c not in categorical_cols]
    
    # We must ordinal encode categories for XGBoost or one-hot encode
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
        ])
    
    # Adjust scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, 
                                         scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

def run_optuna_tuning(data, X_train, y_train):
    """Bayesian optimization using Optuna to tune LightGBM on a temporal train/val split."""
    # Split training data into train (< 2016) and val (2016-2017) to prevent leakage
    train_mask = data.X_train['year'] < 2016
    val_mask = data.X_train['year'] >= 2016
    
    X_tr = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_val = X_train[val_mask]
    y_val = y_train[val_mask]
    
    # Calculate scale_pos_weight base
    neg_tr = (y_tr == 0).sum()
    pos_tr = (y_tr == 1).sum()
    default_spw = neg_tr / pos_tr if pos_tr > 0 else 1.0
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5 * default_spw, 2.0 * default_spw),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        # We optimize for F1-score as we have class imbalance
        return f1_score(y_val, preds)
        
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30)
    
    print(f"Best Optuna F1-Score on Validation: {study.best_value:.4f}")
    print("Best params:", study.best_params)
    return study.best_params

def train_and_eval_lgb_tuned(X_train, y_train, X_test, y_test, categorical_cols, best_params):
    # LightGBM handles categorical variables directly if pandas dtypes are 'category'
    # best_params are used to construct the final classifier
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    default_spw = neg / pos if pos > 0 else 1.0
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    params.update(best_params)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }

def run_ablation_study(X_train, y_train, X_test, y_test, best_params):
    """Measure F1-score drop as key features are removed."""
    print("\n--- Ablation Study ---")
    
    # 1. Full Model
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    params.update(best_params)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    full_f1 = f1_score(y_test, model.predict(X_test))
    print(f"Full Model F1-Score: {full_f1:.4f}")
    
    results = [{"Configuration": "Full Model (LightGBM)", "F1-Score": full_f1, "Delta F1": 0.0}]
    
    # Helper function to eval drop
    def eval_subset(drop_cols, config_name):
        cols_to_use = [c for c in X_train.columns if c not in drop_cols]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train[cols_to_use], y_train)
        f1 = f1_score(y_test, m.predict(X_test[cols_to_use]))
        delta = f1 - full_f1
        print(f"w/o {config_name}: F1 = {f1:.4f} (Delta = {delta:+.4f})")
        results.append({"Configuration": f"w/o {config_name}", "F1-Score": f1, "Delta F1": delta})
        
    # 2. w/o Rolling Constructor Reliability
    eval_subset(["constructor_dnf_rate_last5", "constructor_dnf_rate_last10"], "Rolling Constructor Reliability")
    
    # 3. w/o Rolling Driver Reliability
    eval_subset(["driver_dnf_rate_last5", "driver_dnf_rate_last10"], "Rolling Driver Reliability")
    
    # 4. w/o Grid Position
    eval_subset(["grid"], "Grid Position")
    
    return results

def main():
    print("Loading data and building dataset...")
    data = build_dataset(test_year_start=2018)
    
    X_train, y_train, X_test, y_test = preprocess_data(data)
    
    print("\n--- Training Baseline Models ---")
    logistic_results = train_and_eval_logistic(X_train, y_train, X_test, y_test, data.categorical_cols)
    print("Logistic Regression (L2):", logistic_results)
    
    nb_results = train_and_eval_naive_bayes(X_train, y_train, X_test, y_test, data.categorical_cols)
    print("Naive Bayes:", nb_results)
    
    print("\n--- Training Advanced Models ---")
    rf_results = train_and_eval_rf(X_train, y_train, X_test, y_test, data.categorical_cols)
    print("Random Forest:", rf_results)
    
    xgb_results = train_and_eval_xgb(X_train, y_train, X_test, y_test, data.categorical_cols)
    print("XGBoost:", xgb_results)
    
    print("\n--- Running Hyperparameter Optimization (Optuna) ---")
    best_params = run_optuna_tuning(data, X_train, y_train)
    
    print("\n--- Training Tuned LightGBM Model ---")
    best_model, lgb_tuned_results = train_and_eval_lgb_tuned(X_train, y_train, X_test, y_test, data.categorical_cols, best_params)
    print("Tuned LightGBM:", lgb_tuned_results)
    
    # Print comparison table
    print("\n--- Model Comparison Table ---")
    df_compare = pd.DataFrame({
        "Logistic Regression": logistic_results,
        "Naive Bayes": nb_results,
        "Random Forest": rf_results,
        "XGBoost": xgb_results,
        "LightGBM (Tuned)": lgb_tuned_results
    }).T
    print(df_compare.to_string())
    
    # Run ablation study
    ablation_results = run_ablation_study(X_train, y_train, X_test, y_test, best_params)
    
    # Save the best model and parameters
    joblib.dump(best_model, project_root / "model" / "best_model.joblib")
    with open(project_root / "model" / "best_params.json", "w") as f:
        json.dump(best_params, f)
        
    print("\nSaved best model to model/best_model.joblib")

if __name__ == "__main__":
    main()
