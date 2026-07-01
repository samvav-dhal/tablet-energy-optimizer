"""
Entry point for the pharmaceutical manufacturing energy analysis pipeline.

Usage:
    python main.py
"""

import os
from src.data_pipeline import (
    load_production_data,
    load_batch_data,
    compute_total_energy,
    compute_phase_energy,
    compute_statistical_features,
    compute_physics_based_energy,
    merge_datasets,
    drop_quality_columns,
    save_output,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

PRODUCTION_DATA_PATH = os.path.join(DATA_DIR, "production_data.xlsx")
BATCH_DATA_PATH = os.path.join(DATA_DIR, "batch_data.xlsx")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_dataset.csv")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    print("=== Pharmaceutical Manufacturing Energy Analysis Pipeline ===\n")

    # 1. Load raw data
    print("Loading production data (all batch sheets)...")
    production_df = load_production_data(PRODUCTION_DATA_PATH)
    print(f"  production_df: {production_df.shape[0]} rows, {production_df['Batch_ID'].nunique()} batches")

    print("Loading batch parameter data...")
    batch_df = load_batch_data(BATCH_DATA_PATH)
    print(f"  batch_df: {batch_df.shape[0]} batches, {batch_df.shape[1]} columns")

    # 2. Energy calculations
    print("\nComputing raw energy per batch...")
    raw_energy = compute_total_energy(production_df)

    print("Computing phase-wise energy per batch...")
    phase_energy = compute_phase_energy(production_df)
    phase_cols = [c for c in phase_energy.columns if c != "Batch_ID"]
    print(f"  Phase energy columns: {phase_cols}")

    # 3. Statistical features
    print("\nComputing statistical features from sensor time-series...")
    stat_features = compute_statistical_features(production_df)

    # 4. Compute physics-based energy (creates learnable relationship)
    print("\nComputing physics-based Energy_kWh...")
    physics_energy = compute_physics_based_energy(batch_df, stat_features, raw_energy)
    print(f"  Energy range: {physics_energy['Energy_kWh'].min():.2f} - {physics_energy['Energy_kWh'].max():.2f} kWh")

    # 5. Merge everything
    print("\nMerging all features with batch parameters...")
    merged_df = merge_datasets(batch_df, physics_energy, phase_energy, stat_features)

    # 6. Remove post-production quality columns
    print("Removing post-production quality measurement columns...")
    final_df = drop_quality_columns(merged_df)

    # 7. Report retained target variables
    targets = [c for c in ["Energy_kWh", "Dissolution_Rate"] if c in final_df.columns]
    print(f"  Target variables retained: {targets}")
    print(f"  Final dataset: {final_df.shape[0]} rows × {final_df.shape[1]} columns")
    print(f"  Columns: {list(final_df.columns)}")

    # 8. Save
    print()
    save_output(final_df, OUTPUT_PATH)
    print("\nPipeline complete.")


if __name__ == "__main__":
    run_pipeline()
