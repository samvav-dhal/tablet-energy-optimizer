"""
Adaptive Optimization Module for Pharmaceutical Manufacturing
=============================================================

Implements dynamic real-time optimization algorithms for energy and emission
goal optimization along with multi-objective targets.

Key Components:
- RealTimePerformanceTracker: Tracks actual vs predicted outcomes, detects drift
- StreamingAnomalyDetector: Detects anomalies in sensor data streams
- AdaptiveConstraintManager: Dynamically adjusts constraints based on trajectory
- AdaptiveFeedbackController: Compares streaming data against Golden Signature targets
- ParameterAdjustmentEngine: Generates parameter adjustment recommendations
- DynamicGoalOptimizer: Real-time Pareto re-optimization for remaining phases
- EnergyEmissionBalancer: Specialized energy-emission tradeoff optimizer
- AdaptiveOptimizationOrchestrator: Main coordinator for all components

This module bridges the gap between streaming sensor data, optimization,
and continuous learning to enable real-time adaptive decision making.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Callable, Generator, Any, Literal
from collections import deque
from enum import Enum
import copy
import warnings

from .golden_signature import (
    GoldenSignatureFramework,
    GoldenSignature,
    TargetConfig,
    AVAILABLE_TARGETS,
    PARAM_RANGES,
    SENSOR_DEFAULTS,
    CARBON_FACTOR,
    get_framework,
    OptimizationDirection,
)
from .continuous_learning import (
    ContinuousLearningEngine,
    LearningEvent,
    LearningEventType,
    get_learning_engine,
)
from .predictor import predict as default_predict, get_predictor


# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
ADAPTIVE_CONFIG_PATH = OUTPUTS_DIR / "adaptive_config.json"
ADAPTATION_HISTORY_PATH = OUTPUTS_DIR / "adaptation_history.json"

# Manufacturing phases in order
MANUFACTURING_PHASES = [
    "Preparation", "Granulation", "Drying", "Compression",
    "Blending", "Coating", "Quality_Testing", "Milling"
]

# Parameters that can be adjusted mid-batch (others require restart)
# Note: The model has limited sensitivity to most parameters
# Including sensor-influenced params that have more effect
MID_BATCH_CONTROLLABLE_PARAMS = [
    "Machine_Speed", 
    "Compression_Force",
    "Drying_Temp",          # Has effect on quality (-0.27 per 10°C)
    "avg_temperature",      # Sensor reading - most model-sensitive (+0.04 quality, +0.28 energy per 15°C)
]

# Default configuration
DEFAULT_CONFIG = {
    "drift_threshold_pct": 5.0,         # Trigger alert when drift > 5%
    "anomaly_z_threshold": 3.0,         # Z-score threshold for anomalies
    "anomaly_iqr_multiplier": 1.5,      # IQR multiplier for outlier detection
    "adjustment_limit_pct": 10.0,       # Max ±10% adjustment per intervention
    "min_readings_for_stats": 5,        # Minimum readings before calculating stats
    "energy_buffer_pct": 5.0,           # Energy budget buffer percentage
    "required_confirmations": 2,        # Confirmations before auto-apply
    "auto_apply_mode": False,           # Manual approval by default
}


# ---------------------------------------------------------------------------
# Event Types & Data Classes
# ---------------------------------------------------------------------------

class AdaptiveEventType(Enum):
    """Types of adaptive optimization events."""
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    DRIFT_DETECTED = "drift_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_BREACH = "threshold_breach"
    CONSTRAINT_ADJUSTED = "constraint_adjusted"
    ADJUSTMENT_RECOMMENDED = "adjustment_recommended"
    ADJUSTMENT_APPLIED = "adjustment_applied"
    ADJUSTMENT_REJECTED = "adjustment_rejected"
    GOAL_REOPTIMIZED = "goal_reoptimized"
    TRAJECTORY_UPDATE = "trajectory_update"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class OrchestratorState(Enum):
    """State machine states for the orchestrator."""
    IDLE = "idle"
    MONITORING = "monitoring"
    DETECTING = "detecting"
    ADJUSTING = "adjusting"
    VALIDATING = "validating"


@dataclass
class AdaptiveEvent:
    """Record of an adaptive optimization event."""
    event_type: AdaptiveEventType
    timestamp: str
    batch_id: str
    severity: AlertSeverity = AlertSeverity.INFO
    signature_name: str | None = None
    message: str = ""
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "batch_id": self.batch_id,
            "severity": self.severity.value,
            "signature_name": self.signature_name,
            "message": self.message,
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AdaptiveEvent":
        return cls(
            event_type=AdaptiveEventType(data["event_type"]),
            timestamp=data["timestamp"],
            batch_id=data["batch_id"],
            severity=AlertSeverity(data.get("severity", "info")),
            signature_name=data.get("signature_name"),
            message=data.get("message", ""),
            details=data.get("details", {}),
        )


@dataclass
class DriftMetrics:
    """Metrics for tracking prediction drift."""
    target_name: str
    predicted: float
    actual: float
    drift_pct: float
    drift_abs: float
    exceeds_threshold: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnomalyAlert:
    """Alert for detected anomaly."""
    sensor_type: str
    value: float
    expected_range: tuple[float, float]
    z_score: float
    severity: AlertSeverity
    timestamp: str
    phase: str
    
    def to_dict(self) -> dict:
        return {
            "sensor_type": self.sensor_type,
            "value": self.value,
            "expected_range": list(self.expected_range),
            "z_score": self.z_score,
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "phase": self.phase,
        }


@dataclass
class ParameterAdjustment:
    """Recommended parameter adjustment."""
    parameter: str
    current_value: float
    recommended_value: float
    change_pct: float
    expected_impact: dict[str, float]  # target -> expected change
    confidence: float
    reason: str
    can_apply_mid_batch: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrajectoryPoint:
    """A point on the batch trajectory."""
    time_minutes: int
    phase: str
    cumulative_energy: float
    predicted_total_energy: float
    target_energy: float
    quality_estimate: float
    target_quality: float
    on_track: bool
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdaptiveSession:
    """Active adaptive optimization session for a batch."""
    batch_id: str
    signature_name: str
    started_at: str
    target_outcomes: dict[str, float]
    current_params: dict[str, float]
    trajectory: list[TrajectoryPoint] = field(default_factory=list)
    adjustments_made: list[ParameterAdjustment] = field(default_factory=list)
    events: list[AdaptiveEvent] = field(default_factory=list)
    state: OrchestratorState = OrchestratorState.IDLE
    
    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "signature_name": self.signature_name,
            "started_at": self.started_at,
            "target_outcomes": self.target_outcomes,
            "current_params": self.current_params,
            "trajectory": [t.to_dict() for t in self.trajectory],
            "adjustments_made": [a.to_dict() for a in self.adjustments_made],
            "events": [e.to_dict() for e in self.events],
            "state": self.state.value,
        }


# ---------------------------------------------------------------------------
# Configuration Manager
# ---------------------------------------------------------------------------

class AdaptiveConfigManager:
    """Manages adaptive optimization configuration."""
    
    def __init__(self, config_path: Path | None = None):
        self.config_path = config_path or ADAPTIVE_CONFIG_PATH
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                saved = json.load(f)
                # Merge with defaults to handle new config options
                config = {**DEFAULT_CONFIG, **saved}
        else:
            config = DEFAULT_CONFIG.copy()
        return config
    
    def save(self):
        """Save current configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def update(self, updates: dict):
        """Update multiple configuration values."""
        self.config.update(updates)


# ---------------------------------------------------------------------------
# Adaptation History
# ---------------------------------------------------------------------------

class AdaptationHistory:
    """Stores and manages adaptation event history."""
    
    def __init__(self, history_path: Path | None = None):
        self.history_path = history_path or ADAPTATION_HISTORY_PATH
        self.events: list[AdaptiveEvent] = []
        self._load()
    
    def _load(self):
        """Load history from file."""
        if self.history_path.exists():
            with open(self.history_path, "r") as f:
                data = json.load(f)
                self.events = [AdaptiveEvent.from_dict(e) for e in data.get("events", [])]
    
    def save(self):
        """Save history to file."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump({
                "events": [e.to_dict() for e in self.events],
                "last_updated": datetime.now().isoformat(),
            }, f, indent=2)
    
    def add_event(self, event: AdaptiveEvent):
        """Add an event to history."""
        self.events.append(event)
        self.save()
    
    def get_events(
        self,
        event_type: AdaptiveEventType | None = None,
        batch_id: str | None = None,
        limit: int | None = None,
    ) -> list[AdaptiveEvent]:
        """Query events with optional filters."""
        filtered = self.events
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if batch_id:
            filtered = [e for e in filtered if e.batch_id == batch_id]
        if limit:
            filtered = filtered[-limit:]
        return filtered
    
    def get_adjustments(self, batch_id: str | None = None) -> list[AdaptiveEvent]:
        """Get adjustment events."""
        events = self.get_events(event_type=AdaptiveEventType.ADJUSTMENT_APPLIED, batch_id=batch_id)
        events += self.get_events(event_type=AdaptiveEventType.ADJUSTMENT_RECOMMENDED, batch_id=batch_id)
        return sorted(events, key=lambda e: e.timestamp)
    
    def get_summary(self) -> dict:
        """Get summary statistics of adaptation history."""
        if not self.events:
            return {"total_events": 0}
        
        type_counts = {}
        for e in self.events:
            type_counts[e.event_type.value] = type_counts.get(e.event_type.value, 0) + 1
        
        return {
            "total_events": len(self.events),
            "event_counts": type_counts,
            "unique_batches": len(set(e.batch_id for e in self.events)),
            "first_event": self.events[0].timestamp if self.events else None,
            "last_event": self.events[-1].timestamp if self.events else None,
        }


# ---------------------------------------------------------------------------
# Real-Time Performance Tracker
# ---------------------------------------------------------------------------

class RealTimePerformanceTracker:
    """
    Tracks actual vs predicted outcomes per batch in rolling windows.
    Detects prediction drift and triggers alerts when thresholds exceeded.
    """
    
    def __init__(
        self,
        drift_threshold_pct: float = 5.0,
        predictor_fn: Callable | None = None,
    ):
        self.drift_threshold_pct = drift_threshold_pct
        self.predictor_fn = predictor_fn or default_predict
        
        # Rolling tracking data
        self._predictions: dict[str, list[float]] = {}  # target -> predictions
        self._actuals: dict[str, list[float]] = {}      # target -> actuals
        self._timestamps: list[str] = []
        self._cumulative_energy: float = 0.0
        self._readings_count: int = 0
        
        # Callbacks
        self._drift_callbacks: list[Callable] = []
    
    def reset(self):
        """Reset tracker for a new batch."""
        self._predictions = {"energy": [], "quality": [], "carbon": []}
        self._actuals = {"energy": [], "quality": [], "carbon": []}
        self._timestamps = []
        self._cumulative_energy = 0.0
        self._readings_count = 0
    
    def add_drift_callback(self, callback: Callable):
        """Register callback for drift detection."""
        self._drift_callbacks.append(callback)
    
    def update(
        self,
        features: dict,
        actual_power_kw: float,
        time_minutes: int,
    ) -> dict[str, DriftMetrics]:
        """
        Update tracker with new reading and compute drift.
        
        Parameters
        ----------
        features : dict
            Current feature values for prediction
        actual_power_kw : float
            Actual power consumption reading
        time_minutes : int
            Current time in batch (minutes)
            
        Returns
        -------
        dict[str, DriftMetrics]
            Drift metrics for each target
        """
        self._readings_count += 1
        self._timestamps.append(datetime.now().isoformat())
        
        # Update cumulative actual energy (power in kW * 1 minute / 60)
        self._cumulative_energy += actual_power_kw / 60.0
        
        # Get predictions
        predictions = self.predictor_fn(features)
        
        # Store predictions
        self._predictions["energy"].append(predictions["Energy_kWh"])
        self._predictions["carbon"].append(predictions["Carbon_kg"])
        self._predictions["quality"].append(predictions["Dissolution_Rate"])
        
        # Store actuals (energy is cumulative, quality is latest prediction as proxy)
        self._actuals["energy"].append(self._cumulative_energy)
        self._actuals["carbon"].append(self._cumulative_energy * CARBON_FACTOR)
        # Quality actual is only known at end - use prediction as proxy
        self._actuals["quality"].append(predictions["Dissolution_Rate"])
        
        # Compute drift metrics
        drift_metrics = self._compute_drift()
        
        # Trigger callbacks for exceeded thresholds
        for target, metrics in drift_metrics.items():
            if metrics.exceeds_threshold:
                for callback in self._drift_callbacks:
                    callback(target, metrics)
        
        return drift_metrics
    
    def _compute_drift(self) -> dict[str, DriftMetrics]:
        """Compute drift metrics for all targets."""
        metrics = {}
        
        for target in ["energy", "quality", "carbon"]:
            if not self._predictions[target]:
                continue
            
            # Use most recent values
            predicted = self._predictions[target][-1]
            actual = self._actuals[target][-1]
            
            # Compute drift
            drift_abs = abs(actual - predicted)
            drift_pct = (drift_abs / (abs(predicted) + 1e-10)) * 100
            exceeds = drift_pct > self.drift_threshold_pct
            
            metrics[target] = DriftMetrics(
                target_name=target,
                predicted=predicted,
                actual=actual,
                drift_pct=drift_pct,
                drift_abs=drift_abs,
                exceeds_threshold=exceeds,
            )
        
        return metrics
    
    def get_trajectory(self) -> dict:
        """Get current trajectory data."""
        return {
            "readings_count": self._readings_count,
            "cumulative_energy": self._cumulative_energy,
            "cumulative_carbon": self._cumulative_energy * CARBON_FACTOR,
            "predictions": {k: v[-1] if v else None for k, v in self._predictions.items()},
            "actuals": {k: v[-1] if v else None for k, v in self._actuals.items()},
            "energy_history": list(zip(self._actuals["energy"], self._predictions["energy"])),
        }
    
    def get_energy_projection(self, total_expected_minutes: int) -> dict:
        """
        Project final energy based on current consumption rate.
        
        Parameters
        ----------
        total_expected_minutes : int
            Expected total batch duration in minutes
            
        Returns
        -------
        dict
            Projection with current rate, projected total, and comparison to prediction
        """
        if self._readings_count < 2:
            return {"status": "insufficient_data"}
        
        # Current consumption rate (kWh per minute)
        rate = self._cumulative_energy / self._readings_count
        
        # Project total
        projected_total = rate * total_expected_minutes
        
        # Compare to latest prediction
        latest_prediction = self._predictions["energy"][-1] if self._predictions["energy"] else 0
        
        return {
            "current_rate_kwh_per_min": rate,
            "readings_count": self._readings_count,
            "cumulative_energy_kwh": self._cumulative_energy,
            "projected_total_kwh": projected_total,
            "predicted_total_kwh": latest_prediction,
            "projection_vs_prediction_pct": ((projected_total - latest_prediction) / (latest_prediction + 1e-10)) * 100,
        }


# ---------------------------------------------------------------------------
# Streaming Anomaly Detector
# ---------------------------------------------------------------------------

class StreamingAnomalyDetector:
    """
    Detects anomalies in streaming sensor data using statistical methods.
    Supports Z-score and IQR-based outlier detection.
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_readings: int = 5,
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_readings = min_readings
        
        # Rolling buffers for each sensor type
        self._buffers: dict[str, deque] = {
            "power": deque(maxlen=100),
            "temperature": deque(maxlen=100),
            "vibration": deque(maxlen=100),
        }
        
        # Alert callbacks
        self._anomaly_callbacks: list[Callable] = []
        
        # Recent alerts (to avoid duplicate alerts)
        self._recent_alerts: deque = deque(maxlen=10)
    
    def reset(self):
        """Reset detector for a new batch."""
        for buffer in self._buffers.values():
            buffer.clear()
        self._recent_alerts.clear()
    
    def add_anomaly_callback(self, callback: Callable):
        """Register callback for anomaly detection."""
        self._anomaly_callbacks.append(callback)
    
    def update(
        self,
        power: float,
        temperature: float,
        vibration: float,
        phase: str,
    ) -> list[AnomalyAlert]:
        """
        Update with new sensor readings and check for anomalies.
        
        Returns
        -------
        list[AnomalyAlert]
            List of detected anomalies (empty if none)
        """
        self._buffers["power"].append(power)
        self._buffers["temperature"].append(temperature)
        self._buffers["vibration"].append(vibration)
        
        alerts = []
        timestamp = datetime.now().isoformat()
        
        # Check each sensor for anomalies
        for sensor_type, value in [("power", power), ("temperature", temperature), ("vibration", vibration)]:
            alert = self._check_anomaly(sensor_type, value, phase, timestamp)
            if alert:
                alerts.append(alert)
                # Notify callbacks
                for callback in self._anomaly_callbacks:
                    callback(alert)
        
        return alerts
    
    def _check_anomaly(
        self,
        sensor_type: str,
        value: float,
        phase: str,
        timestamp: str,
    ) -> AnomalyAlert | None:
        """Check if a value is anomalous for the given sensor type."""
        buffer = self._buffers[sensor_type]
        
        if len(buffer) < self.min_readings:
            return None
        
        values = np.array(buffer)
        mean = np.mean(values)
        std = np.std(values)
        
        # Z-score detection
        if std > 0:
            z_score = (value - mean) / std
        else:
            z_score = 0.0
        
        # IQR detection
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        # Determine if anomaly
        is_z_anomaly = abs(z_score) > self.z_threshold
        is_iqr_anomaly = value < lower_bound or value > upper_bound
        
        if not (is_z_anomaly or is_iqr_anomaly):
            return None
        
        # Determine severity
        if abs(z_score) > self.z_threshold * 2:
            severity = AlertSeverity.CRITICAL
        elif abs(z_score) > self.z_threshold * 1.5:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        # Avoid duplicate alerts
        alert_key = f"{sensor_type}_{phase}_{severity.value}"
        if alert_key in self._recent_alerts:
            return None
        self._recent_alerts.append(alert_key)
        
        return AnomalyAlert(
            sensor_type=sensor_type,
            value=value,
            expected_range=(lower_bound, upper_bound),
            z_score=z_score,
            severity=severity,
            timestamp=timestamp,
            phase=phase,
        )
    
    def get_statistics(self) -> dict:
        """Get current rolling statistics for all sensors."""
        stats = {}
        for sensor_type, buffer in self._buffers.items():
            if len(buffer) < 2:
                stats[sensor_type] = {"status": "insufficient_data"}
                continue
            
            values = np.array(buffer)
            stats[sensor_type] = {
                "count": len(buffer),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
        return stats


# ---------------------------------------------------------------------------
# Adaptive Constraint Manager
# ---------------------------------------------------------------------------

class AdaptiveConstraintManager:
    """
    Dynamically adjusts constraints based on batch trajectory.
    
    For example, if Phase 1 used 20% more energy than target,
    tighten remaining phase budgets to compensate.
    """
    
    def __init__(
        self,
        energy_buffer_pct: float = 5.0,
    ):
        self.energy_buffer_pct = energy_buffer_pct
        
        # Phase-level energy budgets (fraction of total)
        self._default_phase_budgets = {
            "Preparation": 0.05,
            "Granulation": 0.25,
            "Drying": 0.30,
            "Compression": 0.15,
            "Blending": 0.10,
            "Coating": 0.08,
            "Quality_Testing": 0.02,
            "Milling": 0.05,
        }
        
        # Dynamic state
        self._target_total_energy: float = 100.0
        self._consumed_energy: float = 0.0
        self._completed_phases: list[str] = []
        self._phase_actuals: dict[str, float] = {}
        self._adjusted_budgets: dict[str, float] = {}
        
        # Callbacks for constraint changes
        self._constraint_callbacks: list[Callable] = []
    
    def reset(self, target_total_energy: float):
        """Reset for a new batch with target energy."""
        self._target_total_energy = target_total_energy
        self._consumed_energy = 0.0
        self._completed_phases = []
        self._phase_actuals = {}
        self._adjusted_budgets = self._default_phase_budgets.copy()
    
    def add_constraint_callback(self, callback: Callable):
        """Register callback for constraint changes."""
        self._constraint_callbacks.append(callback)
    
    def update_phase_completion(
        self,
        phase: str,
        actual_energy: float,
    ) -> dict:
        """
        Update after a phase completes and rebalance remaining budgets.
        
        Parameters
        ----------
        phase : str
            Completed phase name
        actual_energy : float
            Actual energy consumed in this phase
            
        Returns
        -------
        dict
            Updated budget information
        """
        if phase in self._completed_phases:
            return self.get_budget_status()
        
        self._completed_phases.append(phase)
        self._phase_actuals[phase] = actual_energy
        self._consumed_energy += actual_energy
        
        # Calculate deviation
        budgeted = self._target_total_energy * self._adjusted_budgets.get(phase, 0.1)
        deviation = actual_energy - budgeted
        deviation_pct = (deviation / (budgeted + 1e-10)) * 100
        
        # Rebalance remaining phases if significant deviation
        if abs(deviation_pct) > 5.0:
            self._rebalance_budgets(deviation)
        
        # Notify callbacks
        for callback in self._constraint_callbacks:
            callback(phase, deviation_pct, self._adjusted_budgets)
        
        return self.get_budget_status()
    
    def _rebalance_budgets(self, deviation: float):
        """Redistribute energy budget among remaining phases."""
        remaining_phases = [p for p in self._default_phase_budgets.keys() 
                          if p not in self._completed_phases]
        
        if not remaining_phases:
            return
        
        # Total remaining budget fraction
        remaining_budget_frac = sum(self._adjusted_budgets[p] for p in remaining_phases)
        
        # Adjust each remaining phase proportionally
        adjustment_per_phase = -deviation / len(remaining_phases)
        
        for phase in remaining_phases:
            original = self._adjusted_budgets[phase] * self._target_total_energy
            # Apply adjustment with limits (don't go below 50% or above 150% of original)
            adjusted = original + adjustment_per_phase
            adjusted = max(original * 0.5, min(original * 1.5, adjusted))
            self._adjusted_budgets[phase] = adjusted / self._target_total_energy
    
    def get_phase_budget(self, phase: str) -> float:
        """Get current energy budget for a phase in kWh."""
        return self._target_total_energy * self._adjusted_budgets.get(phase, 0.1)
    
    def get_remaining_budget(self) -> float:
        """Get remaining energy budget for uncompleted phases."""
        remaining = self._target_total_energy - self._consumed_energy
        return max(0, remaining)
    
    def get_budget_status(self) -> dict:
        """Get complete budget status."""
        remaining_phases = [p for p in self._default_phase_budgets.keys() 
                          if p not in self._completed_phases]
        
        return {
            "target_total_energy": self._target_total_energy,
            "consumed_energy": self._consumed_energy,
            "remaining_budget": self.get_remaining_budget(),
            "completed_phases": self._completed_phases,
            "phase_actuals": self._phase_actuals,
            "adjusted_budgets": {p: self._target_total_energy * self._adjusted_budgets[p] 
                                for p in remaining_phases},
            "on_track": self._consumed_energy <= self._target_total_energy * (
                sum(self._default_phase_budgets[p] for p in self._completed_phases) 
                + self.energy_buffer_pct / 100
            ),
        }
    
    def can_relax_constraint(self, target: str, current_value: float) -> bool:
        """Check if a constraint can be relaxed given current trajectory."""
        if target == "energy":
            # Can relax if under budget
            return self._consumed_energy < (
                self._target_total_energy * 
                sum(self._default_phase_budgets[p] for p in self._completed_phases) * 0.9
            )
        return False


# ---------------------------------------------------------------------------
# Adaptive Feedback Controller
# ---------------------------------------------------------------------------

class AdaptiveFeedbackController:
    """
    Compares streaming predictions vs targets from active Golden Signature.
    Generates deviation scores and parameter adjustment recommendations.
    """
    
    def __init__(
        self,
        framework: GoldenSignatureFramework | None = None,
        deviation_threshold: float = 5.0,
    ):
        self.framework = framework or get_framework()
        self._active_signature: GoldenSignature | None = None
        self._target_outcomes: dict[str, float] = {}
        self._latest_predictions: dict[str, float] = {}
        self._deviation_history: list[dict] = []
        self._deviation_threshold = deviation_threshold
    
    def set_deviation_threshold(self, threshold: float):
        """Set the deviation threshold (%) for triggering action."""
        self._deviation_threshold = threshold
    
    def set_signature(self, signature_name: str):
        """Set the active Golden Signature for comparison."""
        self._active_signature = self.framework.get_signature(signature_name)
        if self._active_signature:
            self._target_outcomes = self._active_signature.predicted_outcomes.copy()
        self._deviation_history = []
    
    def update(
        self,
        predictions: dict[str, float],
        cumulative_energy: float,
        progress_pct: float,
    ) -> dict:
        """
        Update with latest predictions and compute deviations.
        
        Parameters
        ----------
        predictions : dict
            Latest predictions (Energy_kWh, Dissolution_Rate, Carbon_kg)
        cumulative_energy : float
            Cumulative energy consumed so far
        progress_pct : float
            Batch progress as percentage (0-100)
            
        Returns
        -------
        dict
            Deviation analysis with recommendations flag
        """
        self._latest_predictions = predictions.copy()
        
        if not self._active_signature:
            return {"status": "no_active_signature"}
        
        # Map prediction keys to target keys
        key_map = {
            "Energy_kWh": "energy",
            "Dissolution_Rate": "quality",
            "Carbon_kg": "carbon",
        }
        
        deviations = {}
        requires_action = False
        
        for pred_key, target_key in key_map.items():
            if pred_key not in predictions:
                continue
            
            predicted = predictions[pred_key]
            
            # Get target - try multiple possible keys
            target = (self._target_outcomes.get(target_key) or 
                     self._target_outcomes.get(pred_key) or
                     self._target_outcomes.get(f"{target_key}_target"))
            
            if target is None:
                continue
            
            # Compute deviation
            deviation_abs = predicted - target
            deviation_pct = (deviation_abs / (abs(target) + 1e-10)) * 100
            
            # Determine if this is a problem (use drift_threshold from config if available)
            # Default to 5% if not specified
            deviation_threshold = getattr(self, '_deviation_threshold', 5.0)
            target_def = AVAILABLE_TARGETS.get(target_key)
            is_problematic = False
            if target_def:
                if target_def.direction == OptimizationDirection.MINIMIZE:
                    is_problematic = deviation_pct > deviation_threshold  # Over target is bad
                else:
                    is_problematic = deviation_pct < -deviation_threshold  # Under target is bad
            
            deviations[target_key] = {
                "predicted": predicted,
                "target": target,
                "deviation_abs": deviation_abs,
                "deviation_pct": deviation_pct,
                "is_problematic": is_problematic,
            }
            
            if is_problematic:
                requires_action = True
        
        # Store in history
        self._deviation_history.append({
            "timestamp": datetime.now().isoformat(),
            "progress_pct": progress_pct,
            "deviations": deviations,
        })
        
        return {
            "status": "ok",
            "deviations": deviations,
            "requires_action": requires_action,
            "progress_pct": progress_pct,
            "cumulative_energy": cumulative_energy,
        }
    
    def get_priority_targets(self) -> list[str]:
        """Get list of priority targets from active signature."""
        if not self._active_signature:
            return []
        return [t.target_name for t in self._active_signature.primary_targets]
    
    def get_deviation_trend(self, target: str) -> dict:
        """Get deviation trend for a target over time."""
        if not self._deviation_history:
            return {"status": "no_data"}
        
        points = []
        for entry in self._deviation_history:
            if target in entry["deviations"]:
                points.append({
                    "timestamp": entry["timestamp"],
                    "progress_pct": entry["progress_pct"],
                    "deviation_pct": entry["deviations"][target]["deviation_pct"],
                })
        
        if len(points) < 2:
            return {"status": "insufficient_data", "points": points}
        
        # Calculate trend
        deviations = [p["deviation_pct"] for p in points]
        trend = "stable"
        if len(deviations) >= 3:
            recent_avg = np.mean(deviations[-3:])
            earlier_avg = np.mean(deviations[:-3]) if len(deviations) > 3 else deviations[0]
            if recent_avg > earlier_avg + 2:
                trend = "worsening"
            elif recent_avg < earlier_avg - 2:
                trend = "improving"
        
        return {
            "status": "ok",
            "points": points,
            "trend": trend,
            "latest_deviation_pct": deviations[-1],
        }


# ---------------------------------------------------------------------------
# Parameter Adjustment Engine
# ---------------------------------------------------------------------------

class ParameterAdjustmentEngine:
    """
    Generates parameter adjustment recommendations based on sensitivity analysis.
    Prioritizes adjustments by impact/cost ratio.
    """
    
    def __init__(
        self,
        adjustment_limit_pct: float = 10.0,
        predictor_fn: Callable | None = None,
    ):
        self.adjustment_limit_pct = adjustment_limit_pct
        self.predictor_fn = predictor_fn or default_predict
        
        # Sensitivity coefficients (approximate impact of 1% param change on outcomes)
        # Positive = increasing param reduces energy/increases quality
        self._sensitivity = {
            "Machine_Speed": {"energy": -0.3, "quality": 0.1, "throughput": 0.5},
            "Compression_Force": {"energy": 0.2, "quality": 0.4, "stability": 0.3},
            "Drying_Temp": {"energy": 0.5, "quality": 0.2, "carbon": 0.5},
            "Drying_Time": {"energy": 0.4, "quality": 0.15, "carbon": 0.4},
            "Granulation_Time": {"energy": 0.3, "quality": 0.2, "carbon": 0.3},
            "Binder_Amount": {"energy": 0.1, "quality": 0.3, "stability": 0.2},
            "Lubricant_Conc": {"energy": 0.05, "quality": 0.1, "stability": 0.15},
            "Moisture_Content": {"energy": 0.15, "quality": -0.2, "stability": -0.1},
        }
    
    def generate_recommendations(
        self,
        current_params: dict[str, float],
        deviations: dict[str, dict],
        mid_batch_only: bool = True,
    ) -> list[ParameterAdjustment]:
        """
        Generate parameter adjustment recommendations.
        
        Parameters
        ----------
        current_params : dict
            Current process parameter values
        deviations : dict
            Current deviations from targets
        mid_batch_only : bool
            If True, only recommend parameters that can be adjusted mid-batch
            
        Returns
        -------
        list[ParameterAdjustment]
            Sorted list of recommended adjustments (best first)
        """
        recommendations = []
        
        # Filter to adjustable parameters
        adjustable = MID_BATCH_CONTROLLABLE_PARAMS if mid_batch_only else list(PARAM_RANGES.keys())
        
        for param in adjustable:
            if param not in current_params:
                continue
            
            current_value = current_params[param]
            param_range = PARAM_RANGES.get(param, (current_value * 0.8, current_value * 1.2))
            
            # Determine best direction based on deviations
            adjustment = self._compute_adjustment(
                param, current_value, param_range, deviations
            )
            
            if adjustment:
                recommendations.append(adjustment)
        
        # Sort by expected impact (higher is better)
        recommendations.sort(
            key=lambda a: sum(abs(v) for v in a.expected_impact.values()),
            reverse=True
        )
        
        return recommendations
    
    def _compute_adjustment(
        self,
        param: str,
        current_value: float,
        param_range: tuple[float, float],
        deviations: dict[str, dict],
    ) -> ParameterAdjustment | None:
        """Compute optimal adjustment for a single parameter."""
        sensitivity = self._sensitivity.get(param, {})
        if not sensitivity:
            return None
        
        # Determine adjustment direction based on problematic deviations
        total_improvement = 0.0
        expected_impact = {}
        
        for target, dev_info in deviations.items():
            if not dev_info.get("is_problematic"):
                continue
            
            target_sens = sensitivity.get(target, 0)
            if target_sens == 0:
                continue
            
            # If over target (positive deviation) for minimize, or under for maximize
            deviation_pct = dev_info["deviation_pct"]
            target_def = AVAILABLE_TARGETS.get(target)
            
            if target_def:
                if target_def.direction == OptimizationDirection.MINIMIZE:
                    # Want to decrease outcome, so adjust param in direction of negative sensitivity
                    direction = -1 if target_sens > 0 else 1
                else:
                    # Want to increase outcome
                    direction = 1 if target_sens > 0 else -1
            else:
                direction = -1 if deviation_pct > 0 else 1
            
            # Calculate expected improvement
            change_pct = direction * min(self.adjustment_limit_pct, abs(deviation_pct) / 2)
            expected_change = change_pct * target_sens
            expected_impact[target] = expected_change
            total_improvement += abs(expected_change)
        
        if total_improvement < 0.5:  # Not worth adjusting
            return None
        
        # Calculate recommended value
        avg_direction = sum(1 if v > 0 else -1 for v in expected_impact.values()) / max(1, len(expected_impact))
        change_pct = avg_direction * min(self.adjustment_limit_pct, total_improvement)
        
        recommended = current_value * (1 + change_pct / 100)
        recommended = max(param_range[0], min(param_range[1], recommended))
        
        actual_change_pct = ((recommended - current_value) / current_value) * 100
        
        if abs(actual_change_pct) < 1:  # Less than 1% change not worth it
            return None
        
        return ParameterAdjustment(
            parameter=param,
            current_value=current_value,
            recommended_value=recommended,
            change_pct=actual_change_pct,
            expected_impact=expected_impact,
            confidence=min(0.9, total_improvement / 10),
            reason=self._generate_reason(param, expected_impact),
            can_apply_mid_batch=param in MID_BATCH_CONTROLLABLE_PARAMS,
        )
    
    def _generate_reason(self, param: str, expected_impact: dict) -> str:
        """Generate human-readable reason for adjustment."""
        impacts = []
        for target, change in expected_impact.items():
            direction = "decrease" if change < 0 else "increase"
            impacts.append(f"{direction} {target} by ~{abs(change):.1f}%")
        
        return f"Adjusting {param} to " + ", ".join(impacts)


# ---------------------------------------------------------------------------
# Dynamic Goal Optimizer
# ---------------------------------------------------------------------------

class DynamicGoalOptimizer:
    """
    Performs real-time Pareto re-optimization for remaining batch phases.
    Supports goal relaxation/tightening based on trajectory.
    """
    
    def __init__(
        self,
        predictor_fn: Callable | None = None,
    ):
        self.predictor_fn = predictor_fn or default_predict
    
    def reoptimize_remaining(
        self,
        current_params: dict[str, float],
        consumed_energy: float,
        remaining_budget: float,
        target_quality: float,
        n_candidates: int = 500,
    ) -> list[dict]:
        """
        Generate optimized parameter suggestions for remaining phases.
        
        Parameters
        ----------
        current_params : dict
            Current process parameters
        consumed_energy : float
            Energy already consumed (kWh)
        remaining_budget : float
            Remaining energy budget (kWh)
        target_quality : float
            Target quality (dissolution rate %)
        n_candidates : int
            Number of candidates to evaluate
            
        Returns
        -------
        list[dict]
            Pareto-optimal parameter configurations
        """
        # Only optimize mid-batch controllable parameters
        adjustable = {k: v for k, v in PARAM_RANGES.items() 
                     if k in MID_BATCH_CONTROLLABLE_PARAMS}
        
        if not adjustable:
            return []
        
        # Generate candidates with small variations
        candidates = []
        base_features = {**current_params, **SENSOR_DEFAULTS}
        
        for _ in range(n_candidates):
            candidate = current_params.copy()
            for param, (low, high) in adjustable.items():
                # Add random perturbation within ±10%
                current = current_params.get(param, (low + high) / 2)
                delta = np.random.uniform(-0.1, 0.1) * current
                candidate[param] = max(low, min(high, current + delta))
            
            # Predict outcomes
            features = {**candidate, **SENSOR_DEFAULTS}
            predictions = self.predictor_fn(features)
            
            # Apply penalty for exceeding budget
            predicted_energy = predictions["Energy_kWh"]
            if predicted_energy > consumed_energy + remaining_budget:
                continue
            
            candidates.append({
                "params": candidate,
                "predictions": predictions,
                "energy": predicted_energy,
                "quality": predictions["Dissolution_Rate"],
            })
        
        if not candidates:
            return []
        
        # Find Pareto frontier (minimize energy, maximize quality)
        pareto = self._find_pareto_frontier(candidates)
        
        return pareto[:5]  # Return top 5 options
    
    def _find_pareto_frontier(self, candidates: list[dict]) -> list[dict]:
        """Find Pareto-optimal solutions."""
        pareto = []
        
        for candidate in candidates:
            is_dominated = False
            to_remove = []
            
            for i, p in enumerate(pareto):
                # Check if candidate dominates p (lower energy AND higher quality)
                if (candidate["energy"] <= p["energy"] and 
                    candidate["quality"] >= p["quality"] and
                    (candidate["energy"] < p["energy"] or candidate["quality"] > p["quality"])):
                    to_remove.append(i)
                
                # Check if p dominates candidate
                elif (p["energy"] <= candidate["energy"] and 
                      p["quality"] >= candidate["quality"] and
                      (p["energy"] < candidate["energy"] or p["quality"] > candidate["quality"])):
                    is_dominated = True
                    break
            
            if not is_dominated:
                for i in reversed(to_remove):
                    pareto.pop(i)
                pareto.append(candidate)
        
        # Sort by energy
        pareto.sort(key=lambda x: x["energy"])
        return pareto
    
    def compute_scenarios(
        self,
        current_params: dict,
        consumed_energy: float,
        remaining_minutes: int,
    ) -> dict:
        """
        Compute pessimistic, realistic, and optimistic scenarios.
        
        Returns
        -------
        dict
            Three scenarios with projected outcomes
        """
        base_features = {**current_params, **SENSOR_DEFAULTS}
        base_predictions = self.predictor_fn(base_features)
        
        scenarios = {}
        
        # Realistic: current trajectory
        scenarios["realistic"] = {
            "projected_energy": base_predictions["Energy_kWh"],
            "projected_quality": base_predictions["Dissolution_Rate"],
            "projected_carbon": base_predictions["Carbon_kg"],
        }
        
        # Pessimistic: 15% worse
        scenarios["pessimistic"] = {
            "projected_energy": base_predictions["Energy_kWh"] * 1.15,
            "projected_quality": base_predictions["Dissolution_Rate"] * 0.95,
            "projected_carbon": base_predictions["Carbon_kg"] * 1.15,
        }
        
        # Optimistic: 10% better
        scenarios["optimistic"] = {
            "projected_energy": base_predictions["Energy_kWh"] * 0.90,
            "projected_quality": min(100, base_predictions["Dissolution_Rate"] * 1.05),
            "projected_carbon": base_predictions["Carbon_kg"] * 0.90,
        }
        
        return scenarios


# ---------------------------------------------------------------------------
# Energy Emission Balancer
# ---------------------------------------------------------------------------

class EnergyEmissionBalancer:
    """
    Specialized algorithm for energy-emission tradeoff optimization.
    Supports dynamic carbon budget allocation across phases.
    """
    
    def __init__(self, carbon_factor: float = CARBON_FACTOR):
        self.carbon_factor = carbon_factor
        
        # Phase carbon intensity (relative to average)
        self._phase_intensity = {
            "Preparation": 0.8,
            "Granulation": 1.2,
            "Drying": 1.5,  # Most energy-intensive
            "Compression": 1.0,
            "Blending": 0.7,
            "Coating": 0.9,
            "Quality_Testing": 0.3,
            "Milling": 0.6,
        }
    
    def allocate_carbon_budget(
        self,
        total_carbon_budget: float,
        completed_phases: list[str],
        consumed_carbon: float,
    ) -> dict[str, float]:
        """
        Allocate remaining carbon budget across phases.
        
        Parameters
        ----------
        total_carbon_budget : float
            Total carbon budget in kg CO2
        completed_phases : list[str]
            Phases already completed
        consumed_carbon : float
            Carbon already consumed
            
        Returns
        -------
        dict[str, float]
            Carbon allocation per remaining phase
        """
        remaining_budget = total_carbon_budget - consumed_carbon
        remaining_phases = [p for p in MANUFACTURING_PHASES if p not in completed_phases]
        
        if not remaining_phases or remaining_budget <= 0:
            return {}
        
        # Calculate total intensity of remaining phases
        total_intensity = sum(self._phase_intensity.get(p, 1.0) for p in remaining_phases)
        
        # Allocate proportionally to intensity
        allocations = {}
        for phase in remaining_phases:
            intensity = self._phase_intensity.get(phase, 1.0)
            allocations[phase] = (intensity / total_intensity) * remaining_budget
        
        return allocations
    
    def suggest_catchup_plan(
        self,
        over_budget_by: float,
        remaining_phases: list[str],
    ) -> dict:
        """
        Generate a catch-up plan when over carbon budget.
        
        Parameters
        ----------
        over_budget_by : float
            Amount over budget in kg CO2
        remaining_phases : list[str]
            Phases still to complete
            
        Returns
        -------
        dict
            Catch-up plan with recommendations
        """
        if not remaining_phases:
            return {
                "status": "no_remaining_phases",
                "message": "No phases remaining for catch-up",
            }
        
        # Find phases with highest intensity for reduction
        phase_potentials = []
        for phase in remaining_phases:
            intensity = self._phase_intensity.get(phase, 1.0)
            # Potential savings = 20% of typical consumption for high-intensity phases
            potential_savings = intensity * over_budget_by * 0.2
            phase_potentials.append({
                "phase": phase,
                "intensity": intensity,
                "potential_savings": potential_savings,
            })
        
        phase_potentials.sort(key=lambda x: x["potential_savings"], reverse=True)
        
        # Generate recommendations
        recommendations = []
        cumulative_savings = 0.0
        
        for pp in phase_potentials:
            if cumulative_savings >= over_budget_by:
                break
            
            recommendation = {
                "phase": pp["phase"],
                "action": self._get_reduction_action(pp["phase"]),
                "expected_savings": pp["potential_savings"],
            }
            recommendations.append(recommendation)
            cumulative_savings += pp["potential_savings"]
        
        return {
            "status": "plan_generated",
            "over_budget_by": over_budget_by,
            "recommendations": recommendations,
            "expected_total_savings": cumulative_savings,
            "can_achieve_target": cumulative_savings >= over_budget_by,
        }
    
    def _get_reduction_action(self, phase: str) -> str:
        """Get recommended action for carbon reduction in a phase."""
        actions = {
            "Drying": "Reduce drying temperature by 5°C and extend time proportionally",
            "Granulation": "Reduce granulation time by 10% with optimized mixing",
            "Compression": "Reduce compression force to minimum acceptable level",
            "Blending": "Use energy-efficient blending profile",
            "Coating": "Optimize coating spray rate for efficiency",
            "Milling": "Reduce milling intensity to minimum specification",
        }
        return actions.get(phase, "Optimize phase parameters for energy efficiency")


# ---------------------------------------------------------------------------
# Adaptive Optimization Orchestrator
# ---------------------------------------------------------------------------

class AdaptiveOptimizationOrchestrator:
    """
    Main coordinator for adaptive optimization.
    
    Connects streaming data, performance tracking, anomaly detection,
    constraint management, and continuous learning into a unified system.
    """
    
    def __init__(
        self,
        config: AdaptiveConfigManager | None = None,
        framework: GoldenSignatureFramework | None = None,
        learning_engine: ContinuousLearningEngine | None = None,
    ):
        # Configuration
        self.config = config or AdaptiveConfigManager()
        
        # External systems
        self.framework = framework or get_framework()
        self.learning_engine = learning_engine
        
        # Internal components
        self.performance_tracker = RealTimePerformanceTracker(
            drift_threshold_pct=self.config.get("drift_threshold_pct", 5.0)
        )
        self.anomaly_detector = StreamingAnomalyDetector(
            z_threshold=self.config.get("anomaly_z_threshold", 3.0),
            iqr_multiplier=self.config.get("anomaly_iqr_multiplier", 1.5),
        )
        self.constraint_manager = AdaptiveConstraintManager(
            energy_buffer_pct=self.config.get("energy_buffer_pct", 5.0)
        )
        self.feedback_controller = AdaptiveFeedbackController(
            framework=self.framework,
            deviation_threshold=self.config.get("drift_threshold_pct", 5.0)
        )
        self.adjustment_engine = ParameterAdjustmentEngine(
            adjustment_limit_pct=self.config.get("adjustment_limit_pct", 10.0)
        )
        self.goal_optimizer = DynamicGoalOptimizer()
        self.energy_balancer = EnergyEmissionBalancer()
        
        # History
        self.history = AdaptationHistory()
        
        # Session state
        self._active_session: AdaptiveSession | None = None
        self._pending_adjustments: list[ParameterAdjustment] = []
        self._callbacks: list[Callable] = []
        
        # Register internal callbacks
        self.performance_tracker.add_drift_callback(self._on_drift_detected)
        self.anomaly_detector.add_anomaly_callback(self._on_anomaly_detected)
        self.constraint_manager.add_constraint_callback(self._on_constraint_adjusted)
    
    def add_callback(self, callback: Callable):
        """Register callback for orchestrator events."""
        self._callbacks.append(callback)
    
    def set_learning_engine(self, engine: ContinuousLearningEngine):
        """
        Connect a continuous learning engine to receive adjustment feedback.
        
        When adjustments are applied, the adjusted parameters and re-predicted
        outcomes will be automatically fed to the learning engine for potential
        Golden Signature benchmark updates.
        
        Parameters
        ----------
        engine : ContinuousLearningEngine
            The learning engine to connect
        """
        self.learning_engine = engine
    
    def _emit_event(self, event: AdaptiveEvent):
        """Emit event to all callbacks and history."""
        self.history.add_event(event)
        if self._active_session:
            self._active_session.events.append(event)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                warnings.warn(f"Callback error: {e}")
    
    def refresh_config(self):
        """
        Refresh internal components with current config values.
        
        Call this after updating config to propagate changes to all components.
        """
        # Update feedback controller's deviation threshold
        self.feedback_controller.set_deviation_threshold(
            self.config.get("drift_threshold_pct", 5.0)
        )
        # Update anomaly detector
        self.anomaly_detector.z_threshold = self.config.get("anomaly_z_threshold", 3.0)
        # Update adjustment engine
        self.adjustment_engine.adjustment_limit_pct = self.config.get("adjustment_limit_pct", 10.0)
        # Update performance tracker
        self.performance_tracker.drift_threshold_pct = self.config.get("drift_threshold_pct", 5.0)
    
    def start_session(
        self,
        batch_id: str,
        signature_name: str,
        initial_params: dict[str, float],
    ) -> AdaptiveSession:
        """
        Start a new adaptive optimization session for a batch.
        
        Parameters
        ----------
        batch_id : str
            Batch identifier
        signature_name : str
            Golden Signature to track against
        initial_params : dict
            Initial process parameters
            
        Returns
        -------
        AdaptiveSession
            The new session object
        """
        # Get target signature
        signature = self.framework.get_signature(signature_name)
        if not signature:
            raise ValueError(f"Signature '{signature_name}' not found")
        
        # Reset all components
        self.performance_tracker.reset()
        self.anomaly_detector.reset()
        
        target_energy = signature.predicted_outcomes.get("energy", 
                        signature.predicted_outcomes.get("Energy_kWh", 100.0))
        self.constraint_manager.reset(target_energy)
        self.feedback_controller.set_signature(signature_name)
        
        # Create session
        self._active_session = AdaptiveSession(
            batch_id=batch_id,
            signature_name=signature_name,
            started_at=datetime.now().isoformat(),
            target_outcomes=signature.predicted_outcomes.copy(),
            current_params=initial_params.copy(),
            state=OrchestratorState.MONITORING,
        )
        
        # Emit session started event
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.SESSION_STARTED,
            timestamp=datetime.now().isoformat(),
            batch_id=batch_id,
            signature_name=signature_name,
            message=f"Adaptive optimization session started for batch {batch_id}",
            details={
                "signature": signature_name,
                "target_energy": target_energy,
                "initial_params": initial_params,
            },
        ))
        
        return self._active_session
    
    def process_reading(
        self,
        power_kw: float,
        temperature_c: float,
        vibration_mm_s: float,
        time_minutes: int,
        phase: str,
        features: dict[str, float],
    ) -> dict:
        """
        Process a single sensor reading through the adaptive pipeline.
        
        Parameters
        ----------
        power_kw : float
            Power consumption (kW)
        temperature_c : float
            Temperature (°C)
        vibration_mm_s : float
            Vibration (mm/s)
        time_minutes : int
            Time in batch (minutes)
        phase : str
            Current manufacturing phase
        features : dict
            Complete feature set for prediction
            
        Returns
        -------
        dict
            Processing results including any recommendations
        """
        if not self._active_session:
            return {"status": "no_active_session"}
        
        self._active_session.state = OrchestratorState.DETECTING
        
        # Update performance tracker
        drift_metrics = self.performance_tracker.update(
            features=features,
            actual_power_kw=power_kw,
            time_minutes=time_minutes,
        )
        
        # Check for anomalies
        anomalies = self.anomaly_detector.update(
            power=power_kw,
            temperature=temperature_c,
            vibration=vibration_mm_s,
            phase=phase,
        )
        
        # Get predictions and update feedback controller
        trajectory = self.performance_tracker.get_trajectory()
        predictions = trajectory["predictions"]
        
        # Estimate progress (assuming ~120 minutes total batch time)
        progress_pct = min(100, (time_minutes / 120) * 100)
        
        feedback = self.feedback_controller.update(
            predictions={
                "Energy_kWh": predictions.get("energy", 0),
                "Dissolution_Rate": predictions.get("quality", 0),
                "Carbon_kg": predictions.get("carbon", 0),
            },
            cumulative_energy=trajectory["cumulative_energy"],
            progress_pct=progress_pct,
        )
        
        # Generate trajectory point
        target_energy = self._active_session.target_outcomes.get("energy", 
                        self._active_session.target_outcomes.get("Energy_kWh", 100))
        target_quality = self._active_session.target_outcomes.get("quality",
                        self._active_session.target_outcomes.get("Dissolution_Rate", 90))
        
        trajectory_point = TrajectoryPoint(
            time_minutes=time_minutes,
            phase=phase,
            cumulative_energy=trajectory["cumulative_energy"],
            predicted_total_energy=predictions.get("energy", 0) or 0,
            target_energy=target_energy,
            quality_estimate=predictions.get("quality", 0) or 0,
            target_quality=target_quality,
            on_track=not feedback.get("requires_action", False),
        )
        self._active_session.trajectory.append(trajectory_point)
        
        # Generate recommendations if needed
        recommendations = []
        auto_applied = None
        if feedback.get("requires_action"):
            self._active_session.state = OrchestratorState.ADJUSTING
            recommendations = self.adjustment_engine.generate_recommendations(
                current_params=self._active_session.current_params,
                deviations=feedback.get("deviations", {}),
                mid_batch_only=True,
            )
            self._pending_adjustments = recommendations
            
            if recommendations:
                self._emit_event(AdaptiveEvent(
                    event_type=AdaptiveEventType.ADJUSTMENT_RECOMMENDED,
                    timestamp=datetime.now().isoformat(),
                    batch_id=self._active_session.batch_id,
                    severity=AlertSeverity.WARNING,
                    signature_name=self._active_session.signature_name,
                    message=f"Parameter adjustments recommended: {len(recommendations)} suggestions",
                    details={
                        "recommendations": [r.to_dict() for r in recommendations],
                        "deviations": feedback.get("deviations", {}),
                    },
                ))
                
                # Auto-apply best adjustment if enabled
                if self.config.get("auto_apply_mode", False) and recommendations:
                    best = recommendations[0]
                    if best.can_apply_mid_batch:
                        result = self.apply_adjustment(0)
                        if result.get("status") == "applied":
                            auto_applied = best.to_dict()
        
        self._active_session.state = OrchestratorState.MONITORING
        
        return {
            "status": "ok",
            "time_minutes": time_minutes,
            "phase": phase,
            "drift_metrics": {k: v.to_dict() for k, v in drift_metrics.items()},
            "anomalies": [a.to_dict() for a in anomalies],
            "deviations": feedback.get("deviations", {}),
            "requires_action": feedback.get("requires_action", False),
            "recommendations": [r.to_dict() for r in recommendations],
            "trajectory_point": trajectory_point.to_dict(),
            "cumulative_energy": trajectory["cumulative_energy"],
            "auto_applied": auto_applied,
        }
    
    def apply_adjustment(self, adjustment_index: int = 0) -> dict:
        """
        Apply a pending adjustment recommendation.
        
        Parameters
        ----------
        adjustment_index : int
            Index of adjustment in pending list (default: 0 = best)
            
        Returns
        -------
        dict
            Result of applying adjustment
        """
        if not self._active_session:
            return {"status": "no_active_session"}
        
        if not self._pending_adjustments:
            return {"status": "no_pending_adjustments"}
        
        if adjustment_index >= len(self._pending_adjustments):
            return {"status": "invalid_index"}
        
        adjustment = self._pending_adjustments[adjustment_index]
        
        # Validate mid-batch applicability
        if not adjustment.can_apply_mid_batch:
            return {
                "status": "cannot_apply_mid_batch",
                "parameter": adjustment.parameter,
            }
        
        # Apply adjustment
        old_value = self._active_session.current_params.get(adjustment.parameter)
        self._active_session.current_params[adjustment.parameter] = adjustment.recommended_value
        self._active_session.adjustments_made.append(adjustment)
        
        # Emit event
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.ADJUSTMENT_APPLIED,
            timestamp=datetime.now().isoformat(),
            batch_id=self._active_session.batch_id,
            severity=AlertSeverity.INFO,
            signature_name=self._active_session.signature_name,
            message=f"Applied adjustment: {adjustment.parameter} {old_value:.2f} -> {adjustment.recommended_value:.2f}",
            details=adjustment.to_dict(),
        ))
        
        # Feed adjusted params to continuous learning engine if available
        cl_updates = []
        if self.learning_engine is not None:
            try:
                # Re-predict with adjusted params
                adjusted_params = self._active_session.current_params.copy()
                full_params = {**SENSOR_DEFAULTS, **adjusted_params}
                new_predictions = default_predict(full_params)
                new_outcomes = {
                    "energy": new_predictions.get("Energy_kWh", 0),
                    "quality": new_predictions.get("Dissolution_Rate", 85),
                    "carbon": new_predictions.get("Energy_kWh", 0) * CARBON_FACTOR,
                    "Energy_kWh": new_predictions.get("Energy_kWh", 0),
                    "Dissolution_Rate": new_predictions.get("Dissolution_Rate", 85),
                }
                # Feed to continuous learning
                cl_result = self.learning_engine.process_batch(
                    batch_id=f"{self._active_session.batch_id}_adj_{len(self._active_session.adjustments_made)}",
                    params=adjusted_params,
                    actual_outcomes=new_outcomes,
                )
                cl_updates = cl_result.get("updates_made", [])
            except Exception as e:
                warnings.warn(f"Failed to feed adjustment to continuous learning: {e}")
        
        # Clear from pending
        self._pending_adjustments.pop(adjustment_index)
        
        return {
            "status": "applied",
            "parameter": adjustment.parameter,
            "old_value": old_value,
            "new_value": adjustment.recommended_value,
            "change_pct": adjustment.change_pct,
            "cl_updates": cl_updates,
        }
    
    def reject_adjustment(self, adjustment_index: int = 0) -> dict:
        """Reject a pending adjustment."""
        if not self._pending_adjustments:
            return {"status": "no_pending_adjustments"}
        
        if adjustment_index >= len(self._pending_adjustments):
            return {"status": "invalid_index"}
        
        adjustment = self._pending_adjustments.pop(adjustment_index)
        
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.ADJUSTMENT_REJECTED,
            timestamp=datetime.now().isoformat(),
            batch_id=self._active_session.batch_id if self._active_session else "unknown",
            severity=AlertSeverity.INFO,
            message=f"Rejected adjustment for {adjustment.parameter}",
            details=adjustment.to_dict(),
        ))
        
        return {"status": "rejected", "parameter": adjustment.parameter}
    
    def complete_phase(self, phase: str, actual_energy: float) -> dict:
        """
        Mark a phase as complete and update constraints.
        
        Parameters
        ----------
        phase : str
            Completed phase name
        actual_energy : float
            Actual energy consumed in the phase
            
        Returns
        -------
        dict
            Updated budget status
        """
        budget_status = self.constraint_manager.update_phase_completion(phase, actual_energy)
        
        return {
            "status": "phase_completed",
            "phase": phase,
            "actual_energy": actual_energy,
            "budget_status": budget_status,
        }
    
    def end_session(
        self,
        final_outcomes: dict[str, float] | None = None,
    ) -> dict:
        """
        End the adaptive optimization session.
        
        Parameters
        ----------
        final_outcomes : dict, optional
            Final actual outcomes for the batch
            
        Returns
        -------
        dict
            Session summary
        """
        if not self._active_session:
            return {"status": "no_active_session"}
        
        session = self._active_session
        
        # Feed to continuous learning if outcomes provided
        if final_outcomes and self.learning_engine:
            self.learning_engine.process_batch(
                batch_id=session.batch_id,
                params=session.current_params,
                actual_outcomes=final_outcomes,
            )
        
        # Emit session ended event
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.SESSION_ENDED,
            timestamp=datetime.now().isoformat(),
            batch_id=session.batch_id,
            signature_name=session.signature_name,
            message=f"Adaptive optimization session ended for batch {session.batch_id}",
            details={
                "adjustments_made": len(session.adjustments_made),
                "total_events": len(session.events),
                "final_outcomes": final_outcomes,
            },
        ))
        
        summary = session.to_dict()
        self._active_session = None
        self._pending_adjustments = []
        
        return {
            "status": "session_ended",
            "summary": summary,
        }
    
    def get_status(self) -> dict:
        """Get current orchestrator status."""
        if not self._active_session:
            return {
                "status": "idle",
                "has_active_session": False,
            }
        
        trajectory = self.performance_tracker.get_trajectory()
        budget = self.constraint_manager.get_budget_status()
        
        return {
            "status": "active",
            "has_active_session": True,
            "batch_id": self._active_session.batch_id,
            "signature_name": self._active_session.signature_name,
            "state": self._active_session.state.value,
            "readings_processed": trajectory["readings_count"],
            "cumulative_energy": trajectory["cumulative_energy"],
            "pending_adjustments": len(self._pending_adjustments),
            "adjustments_made": len(self._active_session.adjustments_made),
            "budget_status": budget,
        }
    
    def get_trajectory(self) -> list[dict]:
        """Get the current batch trajectory."""
        if not self._active_session:
            return []
        return [t.to_dict() for t in self._active_session.trajectory]
    
    def get_scenarios(self) -> dict:
        """Get projected scenarios for the current batch."""
        if not self._active_session:
            return {"status": "no_active_session"}
        
        trajectory = self.performance_tracker.get_trajectory()
        remaining_minutes = 120 - trajectory["readings_count"]
        
        return self.goal_optimizer.compute_scenarios(
            current_params=self._active_session.current_params,
            consumed_energy=trajectory["cumulative_energy"],
            remaining_minutes=max(0, remaining_minutes),
        )
    
    # -----------------------------------------------------------------------
    # Internal Callbacks
    # -----------------------------------------------------------------------
    
    def _on_drift_detected(self, target: str, metrics: DriftMetrics):
        """Handle drift detection callback."""
        if not self._active_session:
            return
        
        severity = AlertSeverity.WARNING if metrics.drift_pct < 10 else AlertSeverity.CRITICAL
        
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.DRIFT_DETECTED,
            timestamp=datetime.now().isoformat(),
            batch_id=self._active_session.batch_id,
            severity=severity,
            signature_name=self._active_session.signature_name,
            message=f"Prediction drift detected for {target}: {metrics.drift_pct:.1f}%",
            details=metrics.to_dict(),
        ))
    
    def _on_anomaly_detected(self, alert: AnomalyAlert):
        """Handle anomaly detection callback."""
        if not self._active_session:
            return
        
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.ANOMALY_DETECTED,
            timestamp=datetime.now().isoformat(),
            batch_id=self._active_session.batch_id,
            severity=alert.severity,
            signature_name=self._active_session.signature_name,
            message=f"Anomaly detected in {alert.sensor_type}: {alert.value:.2f} (z={alert.z_score:.2f})",
            details=alert.to_dict(),
        ))
    
    def _on_constraint_adjusted(self, phase: str, deviation_pct: float, new_budgets: dict):
        """Handle constraint adjustment callback."""
        if not self._active_session:
            return
        
        self._emit_event(AdaptiveEvent(
            event_type=AdaptiveEventType.CONSTRAINT_ADJUSTED,
            timestamp=datetime.now().isoformat(),
            batch_id=self._active_session.batch_id,
            severity=AlertSeverity.INFO,
            signature_name=self._active_session.signature_name,
            message=f"Constraints adjusted after {phase}: {deviation_pct:.1f}% deviation",
            details={
                "phase": phase,
                "deviation_pct": deviation_pct,
                "new_budgets": new_budgets,
            },
        ))


# ---------------------------------------------------------------------------
# Global Instance & Factory Functions
# ---------------------------------------------------------------------------

_orchestrator: AdaptiveOptimizationOrchestrator | None = None


def get_orchestrator() -> AdaptiveOptimizationOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AdaptiveOptimizationOrchestrator()
    return _orchestrator


def create_orchestrator(
    config: dict | None = None,
    framework: GoldenSignatureFramework | None = None,
    learning_engine: ContinuousLearningEngine | None = None,
) -> AdaptiveOptimizationOrchestrator:
    """
    Create a new orchestrator with custom configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration overrides
    framework : GoldenSignatureFramework, optional
        Custom framework instance
    learning_engine : ContinuousLearningEngine, optional
        Custom learning engine instance
        
    Returns
    -------
    AdaptiveOptimizationOrchestrator
        New orchestrator instance
    """
    config_manager = AdaptiveConfigManager()
    if config:
        config_manager.update(config)
    
    return AdaptiveOptimizationOrchestrator(
        config=config_manager,
        framework=framework,
        learning_engine=learning_engine,
    )
