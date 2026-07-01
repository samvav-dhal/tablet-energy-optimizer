"""
Real-Time Machine Sensor Streaming Simulator
=============================================
Simulates real-time sensor data streaming for pharmaceutical manufacturing
processes with rolling statistics and predictions.

Includes adaptive optimization integration for real-time parameter adjustment
recommendations and anomaly detection.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Generator, Optional, Callable, TYPE_CHECKING

from .predictor import predict

if TYPE_CHECKING:
    from .adaptive_optimizer import AdaptiveOptimizationOrchestrator

# Paths
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"


class RollingStatistics:
    """Maintains rolling statistics for sensor data within a batch."""
    
    def __init__(self):
        """Initialize empty rolling statistics."""
        self.reset()
    
    def reset(self):
        """Reset all statistics for a new batch."""
        self.power_values = []
        self.temperature_values = []
        self.vibration_values = []
    
    def update(self, power: float, temperature: float, vibration: float):
        """
        Update rolling statistics with new sensor readings.
        
        Parameters
        ----------
        power : float
            Power consumption reading (kW)
        temperature : float
            Temperature reading (°C)
        vibration : float
            Vibration reading (mm/s)
        """
        self.power_values.append(power)
        self.temperature_values.append(temperature)
        self.vibration_values.append(vibration)
    
    def get_features(self) -> dict:
        """
        Compute current rolling statistics.
        
        Returns
        -------
        dict
            Dictionary with rolling statistics:
            - avg_power, max_power, power_std
            - avg_temperature, max_temperature
            - avg_vibration
        """
        if not self.power_values:
            return {
                "avg_power": 0.0,
                "max_power": 0.0,
                "power_std": 0.0,
                "avg_temperature": 0.0,
                "max_temperature": 0.0,
                "avg_vibration": 0.0
            }
        
        return {
            "avg_power": float(np.mean(self.power_values)),
            "max_power": float(np.max(self.power_values)),
            "power_std": float(np.std(self.power_values)) if len(self.power_values) > 1 else 0.0,
            "avg_temperature": float(np.mean(self.temperature_values)),
            "max_temperature": float(np.max(self.temperature_values)),
            "avg_vibration": float(np.mean(self.vibration_values))
        }
    
    @property
    def count(self) -> int:
        """Return the number of readings collected."""
        return len(self.power_values)


class SensorDataStream:
    """
    Simulates real-time sensor data streaming for pharmaceutical manufacturing.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        delay: float = 0.1,
        process_params: Optional[dict] = None
    ):
        """
        Initialize the sensor data stream.
        
        Parameters
        ----------
        data_path : str, optional
            Path to the production data file (CSV or Excel).
            Defaults to data/production_data.csv
        delay : float
            Delay between emitting rows (seconds). Default: 0.1
        process_params : dict, optional
            Default process parameters for predictions
        """
        self.delay = delay
        self.data = self._load_data(data_path)
        self.rolling_stats = RollingStatistics()
        self.current_batch = None

        # Per-batch process parameters loaded from processed_dataset.csv
        self._batch_params = self._load_batch_params()

        # Fallback defaults (used only if a batch is not found in processed_dataset)
        # Values are mid-range of actual production data
        self._default_params = process_params or {
            "Granulation_Time": 18,
            "Binder_Amount": 9.0,
            "Drying_Temp": 58,
            "Drying_Time": 30,
            "Compression_Force": 11.0,
            "Machine_Speed": 180,
            "Lubricant_Conc": 1.5,
            "Moisture_Content": 2.0
        }
    
    def _load_batch_params(self) -> dict:
        """
        Load per-batch process parameters from outputs/processed_dataset.csv.
        Returns a dict keyed by Batch_ID with their individual parameter dicts.
        """
        param_cols = [
            "Batch_ID", "Granulation_Time", "Binder_Amount", "Drying_Temp",
            "Drying_Time", "Compression_Force", "Machine_Speed",
            "Lubricant_Conc", "Moisture_Content"
        ]
        processed_path = OUTPUTS_DIR / "processed_dataset.csv"
        if not processed_path.exists():
            return {}
        df = pd.read_csv(processed_path, usecols=param_cols)
        return {
            row["Batch_ID"]: {col: row[col] for col in param_cols if col != "Batch_ID"}
            for _, row in df.iterrows()
        }

    def get_process_params(self, batch_id: str) -> dict:
        """Return process parameters for a specific batch, falling back to defaults."""
        return self._batch_params.get(batch_id, self._default_params)

    def _load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load production data from CSV or Excel file."""
        if data_path:
            path = Path(data_path)
        else:
            csv_path = DATA_DIR / "production_data.csv"
            xlsx_path = DATA_DIR / "production_data.xlsx"
            
            if csv_path.exists():
                path = csv_path
            elif xlsx_path.exists():
                path = xlsx_path
            else:
                raise FileNotFoundError("Production data not found")
        
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def set_process_params(self, params: dict):
        """Override default fallback process parameters."""
        self._default_params.update(params)
    
    def stream_batch(
        self,
        batch_id: str,
        callback: Optional[Callable] = None
    ) -> Generator[dict, None, None]:
        """
        Stream sensor data for a specific batch.
        
        Parameters
        ----------
        batch_id : str
            The batch ID to stream
        callback : callable, optional
            Function to call with each emitted record
        
        Yields
        ------
        dict
            Dictionary containing:
            - row: Original sensor data
            - rolling_stats: Current rolling statistics
            - features: Complete feature set for prediction
            - predictions: Model predictions (Energy_kWh, Dissolution_Rate, Carbon_kg)
        """
        batch_data = self.data[self.data["Batch_ID"] == batch_id]
        
        if batch_data.empty:
            raise ValueError(f"Batch '{batch_id}' not found in data")
        
        self.rolling_stats.reset()
        self.current_batch = batch_id

        # Use this batch's actual process parameters
        process_params = self.get_process_params(batch_id)

        print(f"\n{'='*60}")
        print(f"Starting stream for Batch: {batch_id}")
        print(f"Total records: {len(batch_data)}")
        print(f"Process params: {process_params}")
        print(f"{'='*60}\n")

        for idx, row in batch_data.iterrows():
            power = row.get("Power_Consumption_kW", 0)
            temperature = row.get("Temperature_C", 0)
            vibration = row.get("Vibration_mm_s", 0)

            self.rolling_stats.update(power, temperature, vibration)
            rolling_features = self.rolling_stats.get_features()
            # Merge this batch's process params with live rolling stats
            features = {**process_params, **rolling_features}
            predictions = predict(features)
            
            result = {
                "row": row.to_dict(),
                "rolling_stats": rolling_features,
                "features": features,
                "predictions": predictions
            }
            
            if callback:
                callback(result)
            
            yield result
            time.sleep(self.delay)
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_id} streaming complete")
        print(f"Total readings processed: {self.rolling_stats.count}")
        print(f"Final predictions: {predictions}")
        print(f"{'='*60}\n")
    
    def stream_all(self, callback: Optional[Callable] = None) -> Generator[dict, None, None]:
        """Stream all batches sequentially."""
        batch_ids = self.data["Batch_ID"].unique()
        for batch_id in batch_ids:
            yield from self.stream_batch(batch_id, callback)
    
    def get_batch_ids(self) -> list:
        """Return list of unique batch IDs in the data."""
        return self.data["Batch_ID"].unique().tolist()


def print_stream_record(record: dict):
    """Callback function to print streaming record details."""
    row = record["row"]
    stats = record["rolling_stats"]
    preds = record["predictions"]
    
    print(f"Time: {row.get('Time_Minutes', 'N/A'):>4} min | "
          f"Phase: {row.get('Phase', 'N/A'):<15} | "
          f"Power: {row.get('Power_Consumption_kW', 0):>6.2f} kW | "
          f"Temp: {row.get('Temperature_C', 0):>5.1f}°C")
    print(f"  Rolling -> Avg Power: {stats['avg_power']:>6.2f} | "
          f"Max Power: {stats['max_power']:>6.2f} | "
          f"Avg Temp: {stats['avg_temperature']:>5.1f}°C")
    print(f"  Predict -> Energy: {preds['Energy_kWh']:>6.2f} kWh | "
          f"Dissolution: {preds['Dissolution_Rate']:>5.2f}% | "
          f"Carbon: {preds['Carbon_kg']:>6.2f} kg")
    print("-" * 70)


def run_simulation(
    batch_id: Optional[str] = None,
    delay: float = 0.1,
    process_params: Optional[dict] = None,
    verbose: bool = True
) -> list:
    """
    Run a streaming simulation for a batch or all batches.
    
    Parameters
    ----------
    batch_id : str, optional
        Specific batch ID to stream. If None, streams first batch.
    delay : float
        Delay between rows in seconds. Default: 0.1
    process_params : dict, optional
        Process parameters for predictions
    verbose : bool
        Whether to print progress. Default: True
    
    Returns
    -------
    list
        List of all emitted records
    """
    stream = SensorDataStream(delay=delay, process_params=process_params)
    callback = print_stream_record if verbose else None
    results = []
    
    if batch_id:
        for record in stream.stream_batch(batch_id, callback):
            results.append(record)
    else:
        batch_ids = stream.get_batch_ids()
        if batch_ids:
            for record in stream.stream_batch(batch_ids[0], callback):
                results.append(record)
    
    return results


# ---------------------------------------------------------------------------
# Adaptive Streaming Integration
# ---------------------------------------------------------------------------

class AdaptiveSensorStream:
    """
    Sensor data stream with integrated adaptive optimization.
    
    Wraps SensorDataStream and processes each reading through the
    adaptive optimization orchestrator for real-time monitoring,
    anomaly detection, and parameter adjustment recommendations.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        delay: float = 0.1,
        orchestrator: Optional["AdaptiveOptimizationOrchestrator"] = None,
    ):
        """
        Initialize adaptive sensor stream.
        
        Parameters
        ----------
        data_path : str, optional
            Path to production data file
        delay : float
            Delay between readings (seconds)
        orchestrator : AdaptiveOptimizationOrchestrator, optional
            Custom orchestrator instance. If None, uses global instance.
        """
        self.base_stream = SensorDataStream(data_path=data_path, delay=delay)
        self._orchestrator = orchestrator
        self._adaptive_callbacks: list[Callable] = []
    
    @property
    def orchestrator(self):
        """Get the adaptive orchestrator, initializing if needed."""
        if self._orchestrator is None:
            from .adaptive_optimizer import get_orchestrator
            self._orchestrator = get_orchestrator()
        return self._orchestrator
    
    def add_adaptive_callback(self, callback: Callable):
        """Register callback for adaptive events."""
        self._adaptive_callbacks.append(callback)
    
    def stream_batch_adaptive(
        self,
        batch_id: str,
        signature_name: str,
        callback: Optional[Callable] = None,
    ) -> Generator[dict, None, dict]:
        """
        Stream sensor data with adaptive optimization integration.
        
        Parameters
        ----------
        batch_id : str
            Batch ID to stream
        signature_name : str
            Golden Signature name to track against
        callback : callable, optional
            Function to call with each record
            
        Yields
        ------
        dict
            Extended record with adaptive optimization data:
            - row: Original sensor data
            - rolling_stats: Rolling statistics
            - features: Complete feature set
            - predictions: Model predictions
            - adaptive: Adaptive optimization results
              - drift_metrics: Prediction drift info
              - anomalies: Detected anomalies
              - recommendations: Parameter adjustments
              - trajectory_point: Current trajectory
        
        Returns
        -------
        dict
            Session summary after streaming completes
        """
        # Get process parameters for this batch
        process_params = self.base_stream.get_process_params(batch_id)
        
        # Start adaptive session
        session = self.orchestrator.start_session(
            batch_id=batch_id,
            signature_name=signature_name,
            initial_params=process_params,
        )
        
        print(f"\n{'='*70}")
        print(f"Starting ADAPTIVE stream for Batch: {batch_id}")
        print(f"Tracking against signature: {signature_name}")
        print(f"Target outcomes: {session.target_outcomes}")
        print(f"{'='*70}\n")
        
        # Stream with adaptive processing
        for base_record in self.base_stream.stream_batch(batch_id, callback=None):
            row = base_record["row"]
            features = base_record["features"]
            
            # Process through adaptive pipeline
            adaptive_result = self.orchestrator.process_reading(
                power_kw=row.get("Power_Consumption_kW", 0),
                temperature_c=row.get("Temperature_C", 0),
                vibration_mm_s=row.get("Vibration_mm_s", 0),
                time_minutes=int(row.get("Time_Minutes", 0)),
                phase=row.get("Phase", "Unknown"),
                features=features,
            )
            
            # Combine results
            record = {
                **base_record,
                "adaptive": adaptive_result,
            }
            
            # Call user callback
            if callback:
                callback(record)
            
            # Call adaptive callbacks for events
            if adaptive_result.get("requires_action"):
                for cb in self._adaptive_callbacks:
                    cb(adaptive_result)
            
            yield record
        
        # End session and return summary
        summary = self.orchestrator.end_session()
        
        print(f"\n{'='*70}")
        print(f"Adaptive stream complete for Batch: {batch_id}")
        print(f"Adjustments made: {len(session.adjustments_made)}")
        print(f"Total events: {len(session.events)}")
        print(f"{'='*70}\n")
        
        return summary
    
    def get_batch_ids(self) -> list:
        """Return list of unique batch IDs."""
        return self.base_stream.get_batch_ids()


def print_adaptive_record(record: dict):
    """Callback to print adaptive streaming record."""
    row = record["row"]
    stats = record["rolling_stats"]
    preds = record["predictions"]
    adaptive = record.get("adaptive", {})
    
    # Basic info
    print(f"Time: {row.get('Time_Minutes', 'N/A'):>4} min | "
          f"Phase: {row.get('Phase', 'N/A'):<15} | "
          f"Power: {row.get('Power_Consumption_kW', 0):>6.2f} kW | "
          f"Temp: {row.get('Temperature_C', 0):>5.1f}°C")
    
    # Predictions
    print(f"  Predict -> Energy: {preds['Energy_kWh']:>6.2f} kWh | "
          f"Dissolution: {preds['Dissolution_Rate']:>5.2f}% | "
          f"Carbon: {preds['Carbon_kg']:>6.2f} kg")
    
    # Cumulative energy
    cum_energy = adaptive.get("cumulative_energy", 0)
    print(f"  Cumulative -> Energy: {cum_energy:>6.2f} kWh")
    
    # Anomalies
    anomalies = adaptive.get("anomalies", [])
    if anomalies:
        print(f"  ⚠️  ANOMALIES: {len(anomalies)} detected")
        for a in anomalies:
            print(f"      - {a['sensor_type']}: {a['value']:.2f} (z={a['z_score']:.2f}) [{a['severity']}]")
    
    # Drift
    drift_metrics = adaptive.get("drift_metrics", {})
    for target, metrics in drift_metrics.items():
        if metrics.get("exceeds_threshold"):
            print(f"  ⚡ DRIFT [{target}]: {metrics['drift_pct']:.1f}% "
                  f"(predicted={metrics['predicted']:.2f}, actual={metrics['actual']:.2f})")
    
    # Recommendations
    recommendations = adaptive.get("recommendations", [])
    if recommendations:
        print(f"  💡 RECOMMENDATIONS: {len(recommendations)} available")
        for r in recommendations[:2]:  # Show top 2
            print(f"      - {r['parameter']}: {r['current_value']:.2f} → {r['recommended_value']:.2f} "
                  f"({r['change_pct']:+.1f}%)")
    
    print("-" * 70)


def run_adaptive_simulation(
    batch_id: Optional[str] = None,
    signature_name: str = "Energy Champion",
    delay: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    Run a streaming simulation with adaptive optimization.
    
    Parameters
    ----------
    batch_id : str, optional
        Batch ID to stream. If None, uses first available.
    signature_name : str
        Golden Signature to track against
    delay : float
        Delay between readings (seconds)
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    dict
        Session summary including trajectory and adjustments
    """
    stream = AdaptiveSensorStream(delay=delay)
    callback = print_adaptive_record if verbose else None
    
    if batch_id is None:
        batch_ids = stream.get_batch_ids()
        if not batch_ids:
            raise ValueError("No batches found in data")
        batch_id = batch_ids[0]
    
    results = []
    summary = None
    
    for record in stream.stream_batch_adaptive(batch_id, signature_name, callback):
        results.append(record)
    
    # Get final status
    status = stream.orchestrator.get_status()
    
    return {
        "batch_id": batch_id,
        "signature_name": signature_name,
        "records_processed": len(results),
        "final_status": status,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate real-time sensor data streaming")
    parser.add_argument("--batch", "-b", type=str, default=None, help="Batch ID to stream")
    parser.add_argument("--delay", "-d", type=float, default=0.5, help="Delay between records (seconds)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    stream = SensorDataStream(delay=args.delay)
    batch_id = args.batch or stream.get_batch_ids()[0]
    
    print(f"Streaming batch: {batch_id}")
    results = run_simulation(batch_id=batch_id, delay=args.delay, verbose=not args.quiet)
    print(f"\nSimulation complete. Processed {len(results)} records.")
