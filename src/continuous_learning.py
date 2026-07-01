"""
Continuous Learning Module for Golden Signature Framework
==========================================================

Implements self-improving systems that automatically update Golden Signature
benchmarks when production performance exceeds current signatures.

Key Components:
- PerformanceMonitor: Tracks real-time batch outcomes against benchmarks
- BenchmarkValidator: Validates improvements before updating (requires N confirmations)
- LearningHistory: Stores all learning events for analysis
- ContinuousLearningEngine: Main orchestrator for simulated learning

The system simulates receiving new sensor data and batch results,
comparing them against existing Golden Signatures, and updating
benchmarks when consistent improvements are detected.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Callable, Generator, Any
from collections import deque
from enum import Enum

from .golden_signature import (
    GoldenSignatureFramework,
    GoldenSignature,
    TargetConfig,
    AVAILABLE_TARGETS,
    SENSOR_DEFAULTS,
    PARAM_RANGES,
    CARBON_FACTOR,
    get_framework,
)
from .predictor import predict as default_predictor


# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
DATA_DIR = Path(__file__).parent.parent / "data"
LEARNING_HISTORY_PATH = OUTPUTS_DIR / "learning_history.json"
PERFORMANCE_HISTORY_PATH = OUTPUTS_DIR / "performance_history.json"


class LearningEventType(Enum):
    """Types of learning events."""
    BATCH_PROCESSED = "batch_processed"
    IMPROVEMENT_DETECTED = "improvement_detected"
    IMPROVEMENT_VALIDATED = "improvement_validated"
    BENCHMARK_UPDATED = "benchmark_updated"
    BENCHMARK_ROLLBACK = "benchmark_rollback"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class LearningEvent:
    """Record of a learning event."""
    event_type: LearningEventType
    timestamp: str
    signature_name: str
    batch_id: str
    old_score: float | None = None
    new_score: float | None = None
    improvement_pct: float | None = None
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "signature_name": self.signature_name,
            "batch_id": self.batch_id,
            "old_score": self.old_score,
            "new_score": self.new_score,
            "improvement_pct": self.improvement_pct,
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LearningEvent":
        return cls(
            event_type=LearningEventType(data["event_type"]),
            timestamp=data["timestamp"],
            signature_name=data["signature_name"],
            batch_id=data["batch_id"],
            old_score=data.get("old_score"),
            new_score=data.get("new_score"),
            improvement_pct=data.get("improvement_pct"),
            details=data.get("details", {}),
        )


@dataclass
class BatchPerformance:
    """Performance metrics for a single batch."""
    batch_id: str
    timestamp: str
    params: dict[str, float]
    outcomes: dict[str, float]
    signature_scores: dict[str, float]  # name -> score
    
    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Performance Monitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Monitors batch performance against Golden Signature benchmarks.
    
    Compares each incoming batch's performance metrics against all
    active Golden Signatures and flags potential improvements.
    """
    
    def __init__(
        self,
        framework: GoldenSignatureFramework,
        improvement_threshold: float = 0.02,  # 2% improvement threshold
    ):
        self.framework = framework
        self.improvement_threshold = improvement_threshold
        self._performance_history: list[BatchPerformance] = []
    
    def evaluate_batch(
        self,
        batch_id: str,
        params: dict[str, float],
        outcomes: dict[str, float],
    ) -> dict[str, dict]:
        """
        Evaluate a batch's performance against all Golden Signatures.
        
        Returns:
        --------
        dict mapping signature_name -> {
            "current_score": float,
            "benchmark_score": float,
            "improvement": float,
            "exceeds_benchmark": bool
        }
        """
        results = {}
        signature_scores = {}
        
        for sig_name in self.framework.list_signatures():
            sig = self.framework.get_signature(sig_name)
            if sig is None:
                continue
            
            # Compute score for this batch using signature's targets
            batch_score = self.framework.optimizer.compute_composite_score(
                outcomes,
                sig.primary_targets,
                sig.secondary_targets,
            )
            
            benchmark_score = sig.composite_score
            improvement = (batch_score - benchmark_score) / benchmark_score if benchmark_score > 0 else 0
            
            results[sig_name] = {
                "current_score": batch_score,
                "benchmark_score": benchmark_score,
                "improvement": improvement,
                "exceeds_benchmark": improvement > self.improvement_threshold,
            }
            signature_scores[sig_name] = batch_score
        
        # Record performance
        perf = BatchPerformance(
            batch_id=batch_id,
            timestamp=datetime.now().isoformat(),
            params=params,
            outcomes=outcomes,
            signature_scores=signature_scores,
        )
        self._performance_history.append(perf)
        
        return results
    
    def get_performance_history(self) -> list[BatchPerformance]:
        """Get all recorded batch performances."""
        return self._performance_history.copy()
    
    def get_recent_performances(self, n: int = 10) -> list[BatchPerformance]:
        """Get the N most recent batch performances."""
        return self._performance_history[-n:]


# ---------------------------------------------------------------------------
# Benchmark Validator
# ---------------------------------------------------------------------------

class BenchmarkValidator:
    """
    Validates improvements before updating benchmarks.
    
    Requires multiple consecutive improvements to confirm that a
    new configuration consistently outperforms the current benchmark.
    """
    
    def __init__(
        self,
        required_confirmations: int = 3,
        confirmation_window: int = 5,  # Look at last N batches
        tolerance: float = 0.01,  # 1% tolerance for "similar" performance
    ):
        self.required_confirmations = required_confirmations
        self.confirmation_window = confirmation_window
        self.tolerance = tolerance
        
        # Track pending improvements: signature_name -> list of (batch_id, params, score)
        self._pending_improvements: dict[str, deque] = {}
    
    def record_improvement(
        self,
        signature_name: str,
        batch_id: str,
        params: dict[str, float],
        outcomes: dict[str, float],
        score: float,
    ) -> bool:
        """
        Record a potential improvement and check if validated.
        
        Returns True if improvement is validated (enough confirmations).
        """
        if signature_name not in self._pending_improvements:
            self._pending_improvements[signature_name] = deque(maxlen=self.confirmation_window)
        
        queue = self._pending_improvements[signature_name]
        queue.append({
            "batch_id": batch_id,
            "params": params,
            "outcomes": outcomes,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Check if we have enough consistent improvements
        if len(queue) >= self.required_confirmations:
            # Check if recent scores are consistently high
            recent_scores = [item["score"] for item in queue]
            avg_score = np.mean(recent_scores[-self.required_confirmations:])
            min_score = np.min(recent_scores[-self.required_confirmations:])
            
            # All recent scores should be within tolerance of each other
            score_variance = np.std(recent_scores[-self.required_confirmations:]) / avg_score
            is_consistent = score_variance < self.tolerance
            
            return is_consistent
        
        return False
    
    def get_best_validated_config(
        self,
        signature_name: str,
    ) -> tuple[dict[str, float], dict[str, float], float] | None:
        """
        Get the best validated configuration for a signature.
        
        Returns (params, outcomes, score) or None if not validated.
        """
        if signature_name not in self._pending_improvements:
            return None
        
        queue = self._pending_improvements[signature_name]
        if len(queue) < self.required_confirmations:
            return None
        
        # Return the best from recent confirmations
        best = max(queue, key=lambda x: x["score"])
        return best["params"], best["outcomes"], best["score"]
    
    def clear_pending(self, signature_name: str):
        """Clear pending improvements for a signature after update."""
        if signature_name in self._pending_improvements:
            self._pending_improvements[signature_name].clear()
    
    def get_pending_count(self, signature_name: str) -> int:
        """Get number of pending improvements for a signature."""
        if signature_name not in self._pending_improvements:
            return 0
        return len(self._pending_improvements[signature_name])


# ---------------------------------------------------------------------------
# Learning History Manager
# ---------------------------------------------------------------------------

class LearningHistory:
    """
    Manages the history of all learning events.
    
    Provides persistence and querying capabilities.
    """
    
    def __init__(self, history_path: Path | str | None = None):
        self.history_path = Path(history_path) if history_path else LEARNING_HISTORY_PATH
        self._events: list[LearningEvent] = []
        self._load_history()
    
    def _load_history(self):
        """Load history from file."""
        if self.history_path.exists():
            try:
                with open(self.history_path, "r") as f:
                    data = json.load(f)
                self._events = [LearningEvent.from_dict(e) for e in data.get("events", [])]
            except Exception as e:
                print(f"Warning: Could not load learning history: {e}")
    
    def _save_history(self):
        """Save history to file."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": [e.to_dict() for e in self._events],
            "last_updated": datetime.now().isoformat(),
            "total_events": len(self._events),
        }
        with open(self.history_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def add_event(self, event: LearningEvent):
        """Add a learning event."""
        self._events.append(event)
        self._save_history()
    
    def get_events(
        self,
        event_type: LearningEventType | None = None,
        signature_name: str | None = None,
        limit: int | None = None,
    ) -> list[LearningEvent]:
        """Query events with optional filters."""
        events = self._events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if signature_name:
            events = [e for e in events if e.signature_name == signature_name]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_benchmark_updates(self, signature_name: str | None = None) -> list[LearningEvent]:
        """Get all benchmark update events."""
        return self.get_events(
            event_type=LearningEventType.BENCHMARK_UPDATED,
            signature_name=signature_name,
        )
    
    def get_summary(self) -> dict:
        """Get summary statistics of learning history."""
        updates = self.get_events(event_type=LearningEventType.BENCHMARK_UPDATED)
        improvements = self.get_events(event_type=LearningEventType.IMPROVEMENT_DETECTED)
        
        return {
            "total_events": len(self._events),
            "benchmark_updates": len(updates),
            "improvements_detected": len(improvements),
            "signatures_updated": len(set(e.signature_name for e in updates)),
            "avg_improvement_pct": np.mean([e.improvement_pct for e in updates if e.improvement_pct]) if updates else 0,
        }
    
    def clear(self):
        """Clear all history."""
        self._events = []
        self._save_history()


# ---------------------------------------------------------------------------
# Continuous Learning Engine
# ---------------------------------------------------------------------------

class ContinuousLearningEngine:
    """
    Main orchestrator for continuous learning simulation.
    
    Simulates receiving batch data, evaluates performance, validates
    improvements, and updates Golden Signature benchmarks.
    """
    
    def __init__(
        self,
        framework: GoldenSignatureFramework | None = None,
        predictor_fn: Callable[[dict], dict] | None = None,
        improvement_threshold: float = 0.02,
        required_confirmations: int = 3,
        auto_update: bool = True,
    ):
        self.predictor_fn = predictor_fn or default_predictor
        self.framework = framework or get_framework(self.predictor_fn)
        self.auto_update = auto_update
        
        self.monitor = PerformanceMonitor(
            framework=self.framework,
            improvement_threshold=improvement_threshold,
        )
        self.validator = BenchmarkValidator(
            required_confirmations=required_confirmations,
        )
        self.history = LearningHistory()
        
        self._callbacks: list[Callable[[LearningEvent], None]] = []
        self._batch_count = 0
        self._updates_count = 0
    
    def add_callback(self, callback: Callable[[LearningEvent], None]):
        """Add a callback to be notified of learning events."""
        self._callbacks.append(callback)
    
    def _emit_event(self, event: LearningEvent):
        """Emit a learning event to history and callbacks."""
        self.history.add_event(event)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"Warning: Callback error: {e}")
    
    def process_batch(
        self,
        batch_id: str,
        params: dict[str, float],
        actual_outcomes: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Process a single batch and check for improvements.
        
        Parameters:
        -----------
        batch_id : str
            Unique identifier for the batch
        params : dict
            Process parameters used for this batch
        actual_outcomes : dict, optional
            Actual measured outcomes. If None, will predict using predictor_fn.
            
        Returns:
        --------
        dict with processing results including any updates made
        """
        self._batch_count += 1
        
        # Get outcomes (predict if not provided)
        if actual_outcomes is None:
            # Add sensor defaults if missing
            full_params = params.copy()
            for key, value in SENSOR_DEFAULTS.items():
                if key not in full_params:
                    full_params[key] = value
            
            predictions = self.predictor_fn(full_params)
            outcomes = {
                "energy": predictions.get("Energy_kWh", 0),
                "quality": predictions.get("Dissolution_Rate", 85),
                "carbon": predictions.get("Energy_kWh", 0) * CARBON_FACTOR,
            }
            # Add derived metrics
            outcomes["yield"] = self._estimate_yield(params, outcomes)
            outcomes["throughput"] = self._estimate_throughput(params)
            outcomes["stability"] = self._estimate_stability(params)
        else:
            outcomes = actual_outcomes
        
        # Evaluate against all signatures
        evaluations = self.monitor.evaluate_batch(batch_id, params, outcomes)
        
        # Record batch processed event
        self._emit_event(LearningEvent(
            event_type=LearningEventType.BATCH_PROCESSED,
            timestamp=datetime.now().isoformat(),
            signature_name="all",
            batch_id=batch_id,
            details={"outcomes": outcomes, "evaluations_count": len(evaluations)},
        ))
        
        # Check for improvements
        updates_made = []
        for sig_name, eval_result in evaluations.items():
            if eval_result["exceeds_benchmark"]:
                # Record improvement detected
                self._emit_event(LearningEvent(
                    event_type=LearningEventType.IMPROVEMENT_DETECTED,
                    timestamp=datetime.now().isoformat(),
                    signature_name=sig_name,
                    batch_id=batch_id,
                    old_score=eval_result["benchmark_score"],
                    new_score=eval_result["current_score"],
                    improvement_pct=eval_result["improvement"] * 100,
                ))
                
                # Record for validation
                is_validated = self.validator.record_improvement(
                    signature_name=sig_name,
                    batch_id=batch_id,
                    params=params,
                    outcomes=outcomes,
                    score=eval_result["current_score"],
                )
                
                if is_validated:
                    self._emit_event(LearningEvent(
                        event_type=LearningEventType.IMPROVEMENT_VALIDATED,
                        timestamp=datetime.now().isoformat(),
                        signature_name=sig_name,
                        batch_id=batch_id,
                        new_score=eval_result["current_score"],
                        details={"confirmations": self.validator.required_confirmations},
                    ))
                    
                    # Auto-update if enabled
                    if self.auto_update:
                        update_result = self._update_benchmark(sig_name)
                        if update_result:
                            updates_made.append(sig_name)
        
        return {
            "batch_id": batch_id,
            "outcomes": outcomes,
            "evaluations": evaluations,
            "updates_made": updates_made,
            "batch_count": self._batch_count,
        }
    
    def _update_benchmark(self, signature_name: str) -> bool:
        """Update a Golden Signature benchmark with validated improvement."""
        best_config = self.validator.get_best_validated_config(signature_name)
        if best_config is None:
            return False
        
        params, outcomes, new_score = best_config
        sig = self.framework.get_signature(signature_name)
        if sig is None:
            return False
        
        old_score = sig.composite_score
        
        # Update the signature
        updated = self.framework.update_signature_if_better(
            name=signature_name,
            new_params=params,
            new_outcomes=outcomes,
        )
        
        if updated:
            self._updates_count += 1
            self._emit_event(LearningEvent(
                event_type=LearningEventType.BENCHMARK_UPDATED,
                timestamp=datetime.now().isoformat(),
                signature_name=signature_name,
                batch_id="validation",
                old_score=old_score,
                new_score=new_score,
                improvement_pct=((new_score - old_score) / old_score) * 100,
                details={
                    "new_params": params,
                    "new_outcomes": outcomes,
                    "update_count": self._updates_count,
                },
            ))
            
            # Clear pending validations
            self.validator.clear_pending(signature_name)
        
        return updated
    
    def _estimate_yield(self, params: dict, outcomes: dict) -> float:
        """Estimate production yield."""
        quality = outcomes.get("quality", 85)
        base_yield = 85.0
        quality_factor = (quality - 85) / 15 * 5
        moisture = params.get("Moisture_Content", 2.25)
        moisture_factor = -abs(moisture - 2.25) * 2
        return np.clip(base_yield + quality_factor + moisture_factor, 80, 99)
    
    def _estimate_throughput(self, params: dict) -> float:
        """Estimate throughput."""
        machine_speed = params.get("Machine_Speed", 55)
        return 150 + (machine_speed - 40) * 5
    
    def _estimate_stability(self, params: dict) -> float:
        """Estimate process stability."""
        stability = 90.0
        for param, (low, high) in PARAM_RANGES.items():
            if param in params:
                val = params[param]
                mid = (low + high) / 2
                range_span = (high - low) / 2
                deviation = abs(val - mid) / range_span
                stability -= deviation * 2
        return np.clip(stability, 70, 99)
    
    def simulate_from_dataset(
        self,
        dataset_path: str | Path | None = None,
        delay: float = 0.0,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> dict:
        """
        Simulate continuous learning from an existing dataset.
        
        Treats each batch in the dataset as a new "incoming" batch,
        processing them sequentially to simulate real-time learning.
        
        Parameters:
        -----------
        dataset_path : str | Path, optional
            Path to dataset CSV. Uses processed_dataset.csv if not specified.
        delay : float
            Delay between processing batches (seconds)
        progress_callback : callable, optional
            Called after each batch with (current, total, result)
            
        Returns:
        --------
        dict with simulation summary
        """
        if dataset_path is None:
            dataset_path = OUTPUTS_DIR / "processed_dataset.csv"
        
        df = pd.read_csv(dataset_path)
        total_batches = len(df)
        
        results = []
        for idx, row in df.iterrows():
            batch_id = row["Batch_ID"]
            
            # Extract parameters
            params = {
                "Granulation_Time": row["Granulation_Time"],
                "Binder_Amount": row["Binder_Amount"],
                "Drying_Temp": row["Drying_Temp"],
                "Drying_Time": row["Drying_Time"],
                "Compression_Force": row["Compression_Force"],
                "Machine_Speed": row["Machine_Speed"],
                "Lubricant_Conc": row["Lubricant_Conc"],
                "Moisture_Content": row["Moisture_Content"],
                "avg_power": row.get("avg_power", SENSOR_DEFAULTS["avg_power"]),
                "max_power": row.get("max_power", SENSOR_DEFAULTS["max_power"]),
                "power_std": row.get("power_std", SENSOR_DEFAULTS["power_std"]),
                "avg_temperature": row.get("avg_temperature", SENSOR_DEFAULTS["avg_temperature"]),
                "max_temperature": row.get("max_temperature", SENSOR_DEFAULTS["max_temperature"]),
                "avg_vibration": row.get("avg_vibration", SENSOR_DEFAULTS["avg_vibration"]),
            }
            
            # Use actual outcomes from dataset
            actual_outcomes = {
                "energy": row["Energy_kWh"],
                "quality": row["Dissolution_Rate"],
                "carbon": row["Energy_kWh"] * CARBON_FACTOR,
                "Energy_kWh": row["Energy_kWh"],
                "Dissolution_Rate": row["Dissolution_Rate"],
            }
            actual_outcomes["yield"] = self._estimate_yield(params, actual_outcomes)
            actual_outcomes["throughput"] = self._estimate_throughput(params)
            actual_outcomes["stability"] = self._estimate_stability(params)
            
            # Process batch
            result = self.process_batch(batch_id, params, actual_outcomes)
            results.append(result)
            
            if progress_callback:
                progress_callback(idx + 1, total_batches, result)
            
            if delay > 0:
                time.sleep(delay)
        
        # Generate summary
        summary = {
            "total_batches": total_batches,
            "total_updates": self._updates_count,
            "improvements_detected": len(self.history.get_events(
                event_type=LearningEventType.IMPROVEMENT_DETECTED
            )),
            "improvements_validated": len(self.history.get_events(
                event_type=LearningEventType.IMPROVEMENT_VALIDATED
            )),
            "signatures_updated": list(set(
                e.signature_name for e in self.history.get_benchmark_updates()
            )),
            "learning_history": self.history.get_summary(),
        }
        
        return summary
    
    def stream_simulation(
        self,
        dataset_path: str | Path | None = None,
        delay: float = 0.1,
    ) -> Generator[dict, None, None]:
        """
        Generator that yields results for each batch processed.
        
        Useful for real-time dashboard updates.
        """
        if dataset_path is None:
            dataset_path = OUTPUTS_DIR / "processed_dataset.csv"
        
        df = pd.read_csv(dataset_path)
        
        for idx, row in df.iterrows():
            batch_id = row["Batch_ID"]
            
            params = {
                "Granulation_Time": row["Granulation_Time"],
                "Binder_Amount": row["Binder_Amount"],
                "Drying_Temp": row["Drying_Temp"],
                "Drying_Time": row["Drying_Time"],
                "Compression_Force": row["Compression_Force"],
                "Machine_Speed": row["Machine_Speed"],
                "Lubricant_Conc": row["Lubricant_Conc"],
                "Moisture_Content": row["Moisture_Content"],
                "avg_power": row.get("avg_power", SENSOR_DEFAULTS["avg_power"]),
                "max_power": row.get("max_power", SENSOR_DEFAULTS["max_power"]),
                "power_std": row.get("power_std", SENSOR_DEFAULTS["power_std"]),
                "avg_temperature": row.get("avg_temperature", SENSOR_DEFAULTS["avg_temperature"]),
                "max_temperature": row.get("max_temperature", SENSOR_DEFAULTS["max_temperature"]),
                "avg_vibration": row.get("avg_vibration", SENSOR_DEFAULTS["avg_vibration"]),
            }
            
            actual_outcomes = {
                "energy": row["Energy_kWh"],
                "quality": row["Dissolution_Rate"],
                "carbon": row["Energy_kWh"] * CARBON_FACTOR,
                "Energy_kWh": row["Energy_kWh"],
                "Dissolution_Rate": row["Dissolution_Rate"],
            }
            actual_outcomes["yield"] = self._estimate_yield(params, actual_outcomes)
            actual_outcomes["throughput"] = self._estimate_throughput(params)
            actual_outcomes["stability"] = self._estimate_stability(params)
            
            result = self.process_batch(batch_id, params, actual_outcomes)
            result["progress"] = (idx + 1) / len(df) * 100
            result["total_batches"] = len(df)
            
            yield result
            
            if delay > 0:
                time.sleep(delay)
    
    def get_status(self) -> dict:
        """Get current learning engine status."""
        return {
            "batches_processed": self._batch_count,
            "benchmarks_updated": self._updates_count,
            "signatures_count": len(self.framework.list_signatures()),
            "pending_validations": {
                sig: self.validator.get_pending_count(sig)
                for sig in self.framework.list_signatures()
            },
            "history_summary": self.history.get_summary(),
        }
    
    def reset(self):
        """Reset the learning engine state (keeps signatures)."""
        self._batch_count = 0
        self._updates_count = 0
        self.validator = BenchmarkValidator(
            required_confirmations=self.validator.required_confirmations,
        )
        self.monitor = PerformanceMonitor(
            framework=self.framework,
            improvement_threshold=self.monitor.improvement_threshold,
        )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def run_learning_simulation(
    delay: float = 0.0,
    improvement_threshold: float = 0.02,
    required_confirmations: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Run a complete learning simulation on the existing dataset.
    
    Parameters:
    -----------
    delay : float
        Delay between batches (seconds)
    improvement_threshold : float
        Minimum improvement % to trigger detection
    required_confirmations : int
        Number of confirmations before updating benchmark
    verbose : bool
        Print progress updates
        
    Returns:
    --------
    dict with simulation summary
    """
    engine = ContinuousLearningEngine(
        improvement_threshold=improvement_threshold,
        required_confirmations=required_confirmations,
    )
    
    def progress_callback(current, total, result):
        if verbose:
            updates = result.get("updates_made", [])
            status = f"Batch {current}/{total}: {result['batch_id']}"
            if updates:
                status += f" | UPDATED: {', '.join(updates)}"
            print(status)
    
    summary = engine.simulate_from_dataset(
        delay=delay,
        progress_callback=progress_callback if verbose else None,
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Total batches processed: {summary['total_batches']}")
        print(f"Improvements detected: {summary['improvements_detected']}")
        print(f"Improvements validated: {summary['improvements_validated']}")
        print(f"Benchmarks updated: {summary['total_updates']}")
        print(f"Signatures updated: {summary['signatures_updated']}")
    
    return summary


# Global engine instance
_engine_instance: ContinuousLearningEngine | None = None


def get_learning_engine(
    reset: bool = False,
    **kwargs,
) -> ContinuousLearningEngine:
    """Get or create the global learning engine instance."""
    global _engine_instance
    if _engine_instance is None or reset:
        _engine_instance = ContinuousLearningEngine(**kwargs)
    return _engine_instance


if __name__ == "__main__":
    # Demo: Run learning simulation
    print("=" * 60)
    print("Continuous Learning Simulation Demo")
    print("=" * 60)
    
    summary = run_learning_simulation(
        delay=0.0,
        improvement_threshold=0.02,
        required_confirmations=3,
        verbose=True,
    )
