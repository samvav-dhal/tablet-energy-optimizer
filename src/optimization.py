"""
Energy optimization module for pharmaceutical tablet manufacturing.

Uses trained ML models to find process parameters that minimize
energy consumption while maintaining tablet quality constraints.
"""

import os
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from .train_models import add_engineered_features, ENGINEERED_FEATURES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Controllable process parameter ranges
PARAM_RANGES = {
    "Granulation_Time": (30, 90),
    "Binder_Amount": (5, 12),
    "Drying_Temp": (55, 75),
    "Drying_Time": (15, 40),
    "Compression_Force": (12, 18),
    "Machine_Speed": (40, 70),
    "Lubricant_Conc": (0.5, 2),
    "Moisture_Content": (1.5, 3),
}

# Sensor features (not controllable) - use values from LOW-ENERGY historical batches
# These are machine observations that correlate with energy consumption.
# Using values from low-energy batches (bottom 20%) gives the model context
# for energy-efficient operation rather than average conditions.
SENSOR_FEATURE_DEFAULTS = {
    "avg_power": 21.0,       # Low-energy batches avg ~21.0 (vs 23.1 overall)
    "max_power": 57.5,       # Low-energy batches avg ~57.5 (vs 59.2 overall)
    "power_std": 15.9,       # Low-energy batches avg ~15.9 (vs 16.3 overall)
    "avg_temperature": 35.2, # Similar across batches
    "max_temperature": 66.5, # Low-energy batches avg ~66.5 (vs 68.4 overall)
    "avg_vibration": 2.8,    # Low-energy batches avg ~2.8 (vs 3.0 overall)
}

# Quality constraint
MIN_DISSOLUTION_RATE = 85.0

# Carbon emission factor (kg CO2 per kWh)
CARBON_FACTOR = 0.7


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

def load_models(
    energy_model_path: str,
    dissolution_model_path: str,
) -> tuple[Any, Any]:
    """Load trained energy and dissolution prediction models."""
    energy_model = joblib.load(energy_model_path)
    dissolution_model = joblib.load(dissolution_model_path)
    return energy_model, dissolution_model


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------

def generate_lhs_samples(
    param_ranges: dict[str, tuple[float, float]],
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate candidate configurations using Latin Hypercube Sampling.

    Parameters
    ----------
    param_ranges : dict
        Mapping from parameter name to (min, max) range.
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataframe with one column per parameter.
    """
    param_names = list(param_ranges.keys())
    n_dims = len(param_names)

    sampler = LatinHypercube(d=n_dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # values in [0, 1]

    # Scale to actual parameter ranges
    scaled_samples = np.zeros_like(unit_samples)
    for i, name in enumerate(param_names):
        low, high = param_ranges[name]
        scaled_samples[:, i] = low + unit_samples[:, i] * (high - low)

    return pd.DataFrame(scaled_samples, columns=param_names)


def add_sensor_features(
    samples: pd.DataFrame,
    sensor_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add constant sensor feature columns to sampled configurations.

    Since sensor readings are observations (not controllable inputs), we use
    representative values derived from training data statistics.
    """
    if sensor_defaults is None:
        sensor_defaults = SENSOR_FEATURE_DEFAULTS

    df = samples.copy()
    for col, value in sensor_defaults.items():
        df[col] = value
    return df


# ---------------------------------------------------------------------------
# 3. Prediction
# ---------------------------------------------------------------------------

def predict_outcomes(
    samples: pd.DataFrame,
    energy_model: Any,
    dissolution_model: Any,
) -> pd.DataFrame:
    """Predict energy consumption and dissolution rate for each configuration.

    Also computes carbon emissions from predicted energy.
    """
    df = samples.copy()

    # Ensure feature order matches model training
    base_features = [
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
    
    # Add engineered features
    df = add_engineered_features(df)
    
    feature_order = base_features + ENGINEERED_FEATURES
    X = df[feature_order]

    df["Predicted_Energy_kWh"] = energy_model.predict(X)
    df["Predicted_Dissolution_Rate"] = dissolution_model.predict(X)
    df["Carbon_kg"] = df["Predicted_Energy_kWh"] * CARBON_FACTOR

    return df


# ---------------------------------------------------------------------------
# 4. Filtering and ranking
# ---------------------------------------------------------------------------

def filter_by_quality(
    df: pd.DataFrame,
    min_dissolution: float = MIN_DISSOLUTION_RATE,
) -> pd.DataFrame:
    """Keep only configurations meeting the dissolution rate constraint."""
    return df[df["Predicted_Dissolution_Rate"] >= min_dissolution].copy()


def rank_by_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Sort configurations by predicted energy (lowest first)."""
    return df.sort_values("Predicted_Energy_kWh").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Pareto frontier
# ---------------------------------------------------------------------------

def compute_pareto_frontier(
    df: pd.DataFrame,
    energy_col: str = "Predicted_Energy_kWh",
    dissolution_col: str = "Predicted_Dissolution_Rate",
) -> pd.DataFrame:
    """Compute Pareto optimal solutions.

    A configuration is Pareto optimal if no other configuration has:
      - lower energy AND higher dissolution rate.

    Parameters
    ----------
    df : pd.DataFrame
        Optimization results with energy and dissolution predictions.
    energy_col : str
        Column name for energy (to minimize).
    dissolution_col : str
        Column name for dissolution (to maximize).

    Returns
    -------
    pd.DataFrame
        Subset of df containing only Pareto optimal configurations,
        sorted by dissolution rate.
    """
    # Sort by dissolution descending (easier to find Pareto front)
    sorted_df = df.sort_values(dissolution_col, ascending=False).reset_index(drop=True)

    pareto_indices = []
    min_energy_seen = float("inf")

    for idx, row in sorted_df.iterrows():
        energy = row[energy_col]
        # If this point has lower energy than all points with higher dissolution,
        # it's on the Pareto frontier
        if energy < min_energy_seen:
            pareto_indices.append(idx)
            min_energy_seen = energy

    pareto_df = sorted_df.loc[pareto_indices].copy()
    # Sort by dissolution rate for plotting
    pareto_df = pareto_df.sort_values(dissolution_col).reset_index(drop=True)
    return pareto_df


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_optimization_results(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame | None = None,
    n_best: int = 10,
    save_path: str | None = None,
) -> None:
    """Create scatter plot of Energy vs Dissolution with Pareto frontier.

    Parameters
    ----------
    df : pd.DataFrame
        All feasible solutions (ranked by energy).
    pareto_df : pd.DataFrame, optional
        Pareto optimal solutions. If None, will be computed.
    n_best : int
        Number of top solutions to highlight.
    save_path : str, optional
        Path to save the plot.
    """
    if pareto_df is None:
        pareto_df = compute_pareto_frontier(df)

    # Sort Pareto frontier by dissolution rate for proper line connection
    pareto_df = pareto_df.sort_values("Predicted_Dissolution_Rate")

    plt.figure(figsize=(12, 8))

    # All feasible solutions (light blue scatter)
    plt.scatter(
        df["Predicted_Dissolution_Rate"],
        df["Predicted_Energy_kWh"],
        alpha=0.3,
        c="lightsteelblue",
        label="Feasible solutions",
        s=15,
    )

    # Pareto frontier: line with markers (orange)
    plt.plot(
        pareto_df["Predicted_Dissolution_Rate"],
        pareto_df["Predicted_Energy_kWh"],
        color="orange",
        marker="o",
        markersize=8,
        markeredgecolor="darkorange",
        linewidth=2,
        label="Pareto frontier",
        zorder=4,
    )

    # Highlight top N lowest energy solutions (red)
    best = df.head(n_best)
    plt.scatter(
        best["Predicted_Dissolution_Rate"],
        best["Predicted_Energy_kWh"],
        c="red",
        s=120,
        edgecolors="black",
        linewidths=1.5,
        label=f"Top {n_best} lowest energy",
        zorder=5,
    )

    # Quality constraint line
    plt.axvline(
        x=MIN_DISSOLUTION_RATE,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Min dissolution ({MIN_DISSOLUTION_RATE}%)",
    )

    plt.xlabel("Predicted Dissolution Rate (%)", fontsize=12)
    plt.ylabel("Predicted Energy Consumption (kWh)", fontsize=12)
    plt.title("Energy Optimization: Pareto Frontier Analysis", fontsize=14)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Saved optimization plot to '{save_path}'")

    plt.close()


# ---------------------------------------------------------------------------
# 7. Output
# ---------------------------------------------------------------------------

def save_results(df: pd.DataFrame, path: str) -> None:
    """Save optimization results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} optimization results to '{path}'")


# ---------------------------------------------------------------------------
# 8. Pipeline orchestration
# ---------------------------------------------------------------------------

def run_optimization(
    energy_model_path: str,
    dissolution_model_path: str,
    output_dir: str,
    n_samples: int = 7500,
) -> pd.DataFrame:
    """End-to-end energy optimization pipeline.

    Parameters
    ----------
    energy_model_path : str
        Path to trained energy prediction model.
    dissolution_model_path : str
        Path to trained dissolution prediction model.
    output_dir : str
        Directory to save results and plots.
    n_samples : int
        Number of LHS samples to generate.

    Returns
    -------
    pd.DataFrame
        Ranked optimization results (feasible solutions only).
    """
    print("=== Energy Optimization Pipeline ===\n")

    # 1. Load models
    print("Loading trained models...")
    energy_model, dissolution_model = load_models(
        energy_model_path, dissolution_model_path
    )

    # 2. Generate candidate configurations
    print(f"Generating {n_samples:,} candidate configurations (LHS)...")
    samples = generate_lhs_samples(PARAM_RANGES, n_samples=n_samples)
    samples = add_sensor_features(samples)
    print(f"  Parameter ranges: {list(PARAM_RANGES.keys())}")

    # 3. Predict outcomes
    print("Predicting energy consumption and dissolution rate...")
    results = predict_outcomes(samples, energy_model, dissolution_model)

    # 4. Filter by quality constraint
    print(f"Filtering by quality constraint (Dissolution_Rate >= {MIN_DISSOLUTION_RATE}%)...")
    feasible = filter_by_quality(results)
    print(f"  {len(feasible):,} / {len(results):,} configurations meet quality constraint")

    # 5. Rank by energy
    print("Ranking feasible solutions by energy consumption...")
    ranked = rank_by_energy(feasible)

    # 6. Display top solutions
    print("\n--- Top 5 Energy-Efficient Configurations ---")
    display_cols = [
        "Granulation_Time",
        "Binder_Amount",
        "Drying_Temp",
        "Drying_Time",
        "Compression_Force",
        "Machine_Speed",
        "Lubricant_Conc",
        "Moisture_Content",
        "Predicted_Energy_kWh",
        "Predicted_Dissolution_Rate",
        "Carbon_kg",
    ]
    print(ranked[display_cols].head(5).to_string(index=False))

    # 7. Compute Pareto frontier
    print("\nComputing Pareto frontier...")
    pareto_df = compute_pareto_frontier(ranked)
    print(f"  Found {len(pareto_df)} Pareto optimal solutions")

    # Display Pareto solutions summary
    print("\n--- Pareto Frontier Summary ---")
    print(f"  Dissolution range: {pareto_df['Predicted_Dissolution_Rate'].min():.2f}% - {pareto_df['Predicted_Dissolution_Rate'].max():.2f}%")
    print(f"  Energy range: {pareto_df['Predicted_Energy_kWh'].min():.2f} - {pareto_df['Predicted_Energy_kWh'].max():.2f} kWh")

    # 8. Plot results with Pareto frontier
    print("\nGenerating visualization...")
    plot_optimization_results(
        ranked,
        pareto_df=pareto_df,
        n_best=10,
        save_path=os.path.join(output_dir, "optimization_scatter.png"),
    )

    # 9. Save results
    save_results(ranked, os.path.join(output_dir, "optimization_results.csv"))
    
    # Save Pareto frontier separately
    pareto_path = os.path.join(output_dir, "pareto_frontier.csv")
    pareto_df.to_csv(pareto_path, index=False)
    print(f"  Saved {len(pareto_df)} Pareto optimal solutions to '{pareto_path}'")

    print("\nOptimization complete.")
    return ranked


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_optimization(
        energy_model_path=os.path.join(BASE_DIR, "models", "energy_model.pkl"),
        dissolution_model_path=os.path.join(BASE_DIR, "models", "dissolution_model.pkl"),
        output_dir=os.path.join(BASE_DIR, "outputs"),
        n_samples=7500,
    )
