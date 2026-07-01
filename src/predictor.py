"""
Predictor Module for Pharmaceutical Manufacturing Energy Optimization
=====================================================================
Loads trained models and provides prediction functions for energy consumption,
dissolution rate, and carbon emissions.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"

# Carbon emission factor (kg CO2 per kWh)
CARBON_FACTOR = 0.7


class TabletPredictor:
    """Predictor class for tablet manufacturing energy and quality predictions."""
    
    def __init__(self):
        """Load trained models."""
        self.energy_model = None
        self.dissolution_model = None
        self._load_models()
    
    def _load_models(self):
        """Load the trained energy and dissolution models."""
        energy_model_path = MODELS_DIR / "energy_model.pkl"
        dissolution_model_path = MODELS_DIR / "dissolution_model.pkl"
        
        if energy_model_path.exists():
            self.energy_model = joblib.load(energy_model_path)
        else:
            raise FileNotFoundError(f"Energy model not found at {energy_model_path}")
        
        if dissolution_model_path.exists():
            self.dissolution_model = joblib.load(dissolution_model_path)
        else:
            raise FileNotFoundError(f"Dissolution model not found at {dissolution_model_path}")
    
    def predict(self, features: dict) -> dict:
        """
        Make predictions for energy consumption, dissolution rate, and carbon emissions.
        
        Parameters
        ----------
        features : dict
            Dictionary containing feature values:
            - Granulation_Time, Binder_Amount, Drying_Temp, Drying_Time
            - Compression_Force, Machine_Speed, Lubricant_Conc, Moisture_Content
            - avg_power, max_power, power_std
            - avg_temperature, max_temperature, avg_vibration
        
        Returns
        -------
        dict
            Dictionary with predictions:
            - Energy_kWh: Predicted energy consumption
            - Dissolution_Rate: Predicted dissolution rate
            - Carbon_kg: Predicted carbon emissions
        """
        # Base feature columns
        base_columns = [
            "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
            "Compression_Force", "Machine_Speed", "Lubricant_Conc", "Moisture_Content",
            "avg_power", "max_power", "power_std",
            "avg_temperature", "max_temperature", "avg_vibration"
        ]
        
        # Get base feature values
        f = {col: features.get(col, 0) for col in base_columns}
        
        # Add engineered features
        f["power_x_time"] = f["avg_power"] * (f["Granulation_Time"] + f["Drying_Time"])
        f["temp_x_time"] = f["Drying_Temp"] * f["Drying_Time"]
        f["force_x_speed"] = f["Compression_Force"] * f["Machine_Speed"]
        f["power_temp_ratio"] = f["avg_power"] / (f["avg_temperature"] + 1)
        f["moisture_drying"] = f["Moisture_Content"] * f["Drying_Time"]
        
        # Full feature order (base + engineered)
        feature_columns = base_columns + [
            "power_x_time", "temp_x_time", "force_x_speed",
            "power_temp_ratio", "moisture_drying"
        ]
        
        # Create feature array
        X = np.array([[f[col] for col in feature_columns]])
        
        # Make predictions
        energy_pred = self.energy_model.predict(X)[0]
        dissolution_pred = self.dissolution_model.predict(X)[0]
        carbon_pred = energy_pred * CARBON_FACTOR
        
        return {
            "Energy_kWh": float(energy_pred),
            "Dissolution_Rate": float(dissolution_pred),
            "Carbon_kg": float(carbon_pred)
        }


# Global predictor instance
_predictor = None


def get_predictor() -> TabletPredictor:
    """Get or create the global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = TabletPredictor()
    return _predictor


def predict(features: dict) -> dict:
    """
    Convenience function to make predictions using the global predictor.
    
    Parameters
    ----------
    features : dict
        Dictionary containing feature values
    
    Returns
    -------
    dict
        Dictionary with Energy_kWh, Dissolution_Rate, and Carbon_kg predictions
    """
    predictor = get_predictor()
    return predictor.predict(features)


if __name__ == "__main__":
    sample_features = {
        "Granulation_Time": 15,
        "Binder_Amount": 8.5,
        "Drying_Temp": 60,
        "Drying_Time": 25,
        "Compression_Force": 12.5,
        "Machine_Speed": 150,
        "Lubricant_Conc": 1.0,
        "Moisture_Content": 2.1,
        "avg_power": 21.82,
        "max_power": 66.07,
        "power_std": 16.40,
        "avg_temperature": 35.22,
        "max_temperature": 64.46,
        "avg_vibration": 2.91
    }
    
    predictions = predict(sample_features)
    print("Predictions:")
    print(f"  Energy: {predictions['Energy_kWh']:.2f} kWh")
    print(f"  Dissolution Rate: {predictions['Dissolution_Rate']:.2f}%")
    print(f"  Carbon Emissions: {predictions['Carbon_kg']:.2f} kg CO2")
