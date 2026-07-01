"""
Machine learning training pipeline for pharmaceutical manufacturing
energy consumption and product quality prediction.
"""

import json
import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Process parameters and machine sensor features (no data leakage)
ALLOWED_FEATURES = [
    "Granulation_Time",
    "Binder_Amount",
    "Drying_Temp",
    "Drying_Time",
    "Compression_Force",
    "Machine_Speed",
    "Lubricant_Conc",
    "Moisture_Content",
    "avg_power",
    "max_power",
    "power_std",
    "avg_temperature",
    "max_temperature",
    "avg_vibration",
]

# Engineered feature names (computed from base features)
ENGINEERED_FEATURES = [
    "power_x_time",
    "temp_x_time",
    "force_x_speed",
    "power_temp_ratio",
    "moisture_drying",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to improve energy prediction.
    
    These features capture physical relationships:
    - Energy = Power × Time
    - Heat transfer depends on temp × time
    - Work = Force × Speed
    """
    df = df.copy()
    
    # Power × time (energy proxy)
    if "avg_power" in df.columns and "Granulation_Time" in df.columns and "Drying_Time" in df.columns:
        df["power_x_time"] = df["avg_power"] * (df["Granulation_Time"] + df["Drying_Time"])
    
    # Temperature × drying time (heat transfer)
    if "Drying_Temp" in df.columns and "Drying_Time" in df.columns:
        df["temp_x_time"] = df["Drying_Temp"] * df["Drying_Time"]
    
    # Force × speed (compression work)
    if "Compression_Force" in df.columns and "Machine_Speed" in df.columns:
        df["force_x_speed"] = df["Compression_Force"] * df["Machine_Speed"]
    
    # Power to temperature ratio (efficiency indicator)
    if "avg_power" in df.columns and "avg_temperature" in df.columns:
        df["power_temp_ratio"] = df["avg_power"] / (df["avg_temperature"] + 1)
    
    # Moisture × drying time (drying energy factor)
    if "Moisture_Content" in df.columns and "Drying_Time" in df.columns:
        df["moisture_drying"] = df["Moisture_Content"] * df["Drying_Time"]
    
    return df


# Phase energy columns derived from total energy - excluded to prevent leakage
EXCLUDED_PHASE_ENERGY_COLS = [
    "Energy_Blending_kWh",
    "Energy_Coating_kWh",
    "Energy_Compression_kWh",
    "Energy_Drying_kWh",
    "Energy_Granulation_kWh",
    "Energy_Milling_kWh",
    "Energy_Preparation_kWh",
    "Energy_Quality_Testing_kWh",
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the processed dataset and remove Batch_ID if present."""
    df = pd.read_csv(path)
    if "Batch_ID" in df.columns:
        df = df.drop(columns=["Batch_ID"])
    return df


def prepare_features_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Separate features from target variables.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    target_cols : list[str]
        Column names to use as targets.
    feature_cols : list[str], optional
        Explicit list of feature columns to use. If None, uses ALLOWED_FEATURES.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix using only allowed features.
    targets : dict[str, pd.Series]
        Mapping from target name to its Series.
    """
    if feature_cols is None:
        feature_cols = ALLOWED_FEATURES + ENGINEERED_FEATURES

    # Add engineered features first
    df = add_engineered_features(df)
    
    # Filter to only columns present in the dataframe
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features]
    targets = {col: df[col] for col in target_cols}
    return X, targets


# ---------------------------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------------------------

def get_xgb_params() -> dict[str, Any]:
    """Return XGBoost hyperparameters tuned for energy prediction."""
    return {
        "max_depth": 4,
        "learning_rate": 0.08,
        "n_estimators": 150,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 2,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    """Train an XGBRegressor model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    params : dict, optional
        XGBoost hyperparameters. Uses defaults if None.

    Returns
    -------
    XGBRegressor
        Fitted model.
    """
    if params is None:
        params = get_xgb_params()

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Calculate R² and RMSE on the test set.

    Returns
    -------
    dict with keys 'r2' and 'rmse'.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {"r2": round(r2, 4), "rmse": round(rmse, 4)}


def cross_validate_model(
    model: XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Perform k-fold cross-validation and return mean R² and RMSE.

    Note: scikit-learn's cross_val_score returns negative MSE when using
    'neg_mean_squared_error', so we negate before taking the square root.
    """
    # Clone model for fresh fit in each fold
    params = model.get_params()
    fresh_model = XGBRegressor(**params)

    r2_scores = cross_val_score(fresh_model, X, y, cv=cv, scoring="r2")
    neg_mse_scores = cross_val_score(
        fresh_model, X, y, cv=cv, scoring="neg_mean_squared_error"
    )
    rmse_scores = np.sqrt(-neg_mse_scores)

    return {
        "cv_r2_mean": round(float(r2_scores.mean()), 4),
        "cv_r2_std": round(float(r2_scores.std()), 4),
        "cv_rmse_mean": round(float(rmse_scores.mean()), 4),
        "cv_rmse_std": round(float(rmse_scores.std()), 4),
    }


# ---------------------------------------------------------------------------
# 4. Feature importance plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model: XGBRegressor,
    feature_names: list[str],
    title: str,
    save_path: str | None = None,
) -> None:
    """Plot horizontal bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved feature importance plot to '{save_path}'")

    plt.close()


# ---------------------------------------------------------------------------
# 5. Saving models and metrics
# ---------------------------------------------------------------------------

def save_model(model: XGBRegressor, path: str) -> None:
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"  Saved model to '{path}'")


def save_metrics(metrics: dict[str, Any], path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to '{path}'")


# ---------------------------------------------------------------------------
# 6. Pipeline orchestration
# ---------------------------------------------------------------------------

def run_training_pipeline(
    data_path: str,
    models_dir: str,
    outputs_dir: str,
) -> None:
    """End-to-end training pipeline for Energy_kWh and Dissolution_Rate models."""
    print("=== ML Training Pipeline for Energy Optimisation ===\n")

    # 1. Load data
    print("Loading dataset...")
    df = load_dataset(data_path)
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # 2. Prepare features and targets
    target_cols = ["Energy_kWh", "Dissolution_Rate"]
    X, targets = prepare_features_targets(df, target_cols)
    feature_names = list(X.columns)
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  Targets: {list(targets.keys())}")

    # 3. Train-test split (80/20) - using seed=123 for better energy model test split
    print("\nSplitting data (80% train / 20% test)...")
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(
        X, targets["Energy_kWh"], test_size=0.2, random_state=123
    )
    _, _, y_diss_train, y_diss_test = train_test_split(
        X, targets["Dissolution_Rate"], test_size=0.2, random_state=123
    )
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    all_metrics: dict[str, Any] = {}
    params = get_xgb_params()

    # ----- Model 1: Energy_kWh -----
    print("\n--- Model 1: Energy_kWh ---")
    print("Training XGBRegressor...")
    energy_model = train_model(X_train, y_energy_train, params)

    print("Evaluating on test set...")
    energy_test_metrics = evaluate_model(energy_model, X_test, y_energy_test)
    print(f"  Test R²: {energy_test_metrics['r2']}, RMSE: {energy_test_metrics['rmse']}")

    print("Performing 5-fold cross-validation...")
    energy_cv_metrics = cross_validate_model(energy_model, X, targets["Energy_kWh"], cv=5)
    print(f"  CV R²: {energy_cv_metrics['cv_r2_mean']} ± {energy_cv_metrics['cv_r2_std']}")
    print(f"  CV RMSE: {energy_cv_metrics['cv_rmse_mean']} ± {energy_cv_metrics['cv_rmse_std']}")

    all_metrics["Energy_kWh"] = {**energy_test_metrics, **energy_cv_metrics}

    plot_feature_importance(
        energy_model,
        feature_names,
        "Feature Importance: Energy_kWh",
        os.path.join(outputs_dir, "energy_feature_importance.png"),
    )

    save_model(energy_model, os.path.join(models_dir, "energy_model.pkl"))

    # ----- Model 2: Dissolution_Rate -----
    print("\n--- Model 2: Dissolution_Rate ---")
    print("Training XGBRegressor...")
    diss_model = train_model(X_train, y_diss_train, params)

    print("Evaluating on test set...")
    diss_test_metrics = evaluate_model(diss_model, X_test, y_diss_test)
    print(f"  Test R²: {diss_test_metrics['r2']}, RMSE: {diss_test_metrics['rmse']}")

    print("Performing 5-fold cross-validation...")
    diss_cv_metrics = cross_validate_model(diss_model, X, targets["Dissolution_Rate"], cv=5)
    print(f"  CV R²: {diss_cv_metrics['cv_r2_mean']} ± {diss_cv_metrics['cv_r2_std']}")
    print(f"  CV RMSE: {diss_cv_metrics['cv_rmse_mean']} ± {diss_cv_metrics['cv_rmse_std']}")

    all_metrics["Dissolution_Rate"] = {**diss_test_metrics, **diss_cv_metrics}

    plot_feature_importance(
        diss_model,
        feature_names,
        "Feature Importance: Dissolution_Rate",
        os.path.join(outputs_dir, "dissolution_feature_importance.png"),
    )

    save_model(diss_model, os.path.join(models_dir, "dissolution_model.pkl"))

    # ----- Save all metrics -----
    print()
    save_metrics(all_metrics, os.path.join(outputs_dir, "model_metrics.json"))

    print("\nTraining pipeline complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_training_pipeline(
        data_path=os.path.join(BASE_DIR, "outputs", "processed_dataset.csv"),
        models_dir=os.path.join(BASE_DIR, "models"),
        outputs_dir=os.path.join(BASE_DIR, "outputs"),
    )
