"""
Golden Signature Framework for Pharmaceutical Manufacturing Optimization
========================================================================

A Golden Signature is a set of optimized process parameters for a given
multi-objective target combination. This module provides:

1. Flexible target selection (any combination of primary + secondary targets)
2. Multi-objective optimization using Pareto-efficient solutions
3. Constraint handling for regulatory and process limits
4. Benchmark storage and comparison

Supported Targets:
- Energy Consumption (minimize)
- Dissolution Rate / Quality (maximize)
- Carbon Emissions (minimize)
- Yield (maximize)
- Production Throughput (maximize)
- Process Stability (maximize)

Example target combinations:
- Best yield with lowest energy consumption
- Optimal quality with best yield scenarios  
- Maximum performance with minimal environmental impact
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal
from scipy.stats.qmc import LatinHypercube
from enum import Enum
import copy


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
GOLDEN_BENCHMARKS_PATH = OUTPUTS_DIR / "golden_benchmarks.json"


class OptimizationDirection(Enum):
    """Direction of optimization for each target."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class TargetDefinition:
    """Definition of an optimization target."""
    name: str
    display_name: str
    direction: OptimizationDirection
    min_value: float
    max_value: float
    unit: str
    description: str
    
    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range where 1 is better."""
        norm = (value - self.min_value) / (self.max_value - self.min_value + 1e-10)
        norm = np.clip(norm, 0, 1)
        if self.direction == OptimizationDirection.MINIMIZE:
            return 1 - norm  # Lower is better, so invert
        return norm  # Higher is better


# Available optimization targets
AVAILABLE_TARGETS = {
    "energy": TargetDefinition(
        name="energy",
        display_name="Energy Consumption",
        direction=OptimizationDirection.MINIMIZE,
        min_value=50.0,
        max_value=150.0,
        unit="kWh",
        description="Total energy consumed per batch"
    ),
    "quality": TargetDefinition(
        name="quality",
        display_name="Quality (Dissolution Rate)",
        direction=OptimizationDirection.MAXIMIZE,
        min_value=70.0,
        max_value=100.0,
        unit="%",
        description="Tablet dissolution rate - higher is better quality"
    ),
    "carbon": TargetDefinition(
        name="carbon",
        display_name="Carbon Emissions",
        direction=OptimizationDirection.MINIMIZE,
        min_value=35.0,
        max_value=105.0,
        unit="kg CO₂",
        description="Carbon footprint per batch"
    ),
    "yield": TargetDefinition(
        name="yield",
        display_name="Production Yield",
        direction=OptimizationDirection.MAXIMIZE,
        min_value=80.0,
        max_value=100.0,
        unit="%",
        description="Percentage of good tablets produced"
    ),
    "throughput": TargetDefinition(
        name="throughput",
        display_name="Throughput",
        direction=OptimizationDirection.MAXIMIZE,
        min_value=100.0,
        max_value=500.0,
        unit="tablets/min",
        description="Production rate"
    ),
    "stability": TargetDefinition(
        name="stability",
        display_name="Process Stability",
        direction=OptimizationDirection.MAXIMIZE,
        min_value=0.0,
        max_value=100.0,
        unit="%",
        description="Consistency of production process"
    ),
}


@dataclass
class ProcessConstraint:
    """A constraint on process parameters or outcomes."""
    name: str
    parameter: str
    operator: Literal[">=", "<=", "==", ">", "<"]
    value: float
    is_hard: bool = True  # Hard constraint must be satisfied; soft is penalized
    penalty_weight: float = 100.0  # Penalty for violating soft constraints


# Regulatory and process constraints
DEFAULT_CONSTRAINTS = [
    ProcessConstraint("Min Quality", "quality", ">=", 85.0, is_hard=True),
    ProcessConstraint("Max Energy", "energy", "<=", 120.0, is_hard=False, penalty_weight=50.0),
    ProcessConstraint("Max Carbon", "carbon", "<=", 90.0, is_hard=False, penalty_weight=30.0),
]


# Controllable process parameter ranges (matched to actual production data)
PARAM_RANGES = {
    "Granulation_Time": (8, 30),      # Actual: 9-27
    "Binder_Amount": (5, 14),          # Actual: 5.8-13.5
    "Drying_Temp": (40, 75),           # Actual: 42-73
    "Drying_Time": (15, 50),           # Actual: 15-48
    "Compression_Force": (4, 20),      # Actual: 4.5-18
    "Machine_Speed": (90, 300),        # Actual: 92-280
    "Lubricant_Conc": (0.3, 3.0),      # Actual: 0.4-2.8
    "Moisture_Content": (0.1, 4.0),    # Actual: 0.2-3.6
}

# Sensor feature defaults (from low-energy batches)
SENSOR_DEFAULTS = {
    "avg_power": 21.0,
    "max_power": 57.5,
    "power_std": 15.9,
    "avg_temperature": 35.2,
    "max_temperature": 66.5,
    "avg_vibration": 2.8,
}

CARBON_FACTOR = 0.7  # kg CO2 per kWh


# ---------------------------------------------------------------------------
# Golden Signature Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TargetConfig:
    """Configuration for a single target in the optimization."""
    target_name: str
    priority: Literal["primary", "secondary"]
    weight: float = 1.0
    constraint_min: float | None = None
    constraint_max: float | None = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TargetConfig":
        return cls(**data)


@dataclass
class GoldenSignature:
    """
    A Golden Signature represents an optimized configuration for a 
    specific multi-objective target combination.
    """
    name: str
    description: str
    primary_targets: list[TargetConfig]
    secondary_targets: list[TargetConfig]
    optimal_params: dict[str, float]
    predicted_outcomes: dict[str, float]
    pareto_rank: int = 1  # Rank in Pareto frontier (1 = best)
    composite_score: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    update_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "primary_targets": [t.to_dict() for t in self.primary_targets],
            "secondary_targets": [t.to_dict() for t in self.secondary_targets],
            "optimal_params": self.optimal_params,
            "predicted_outcomes": self.predicted_outcomes,
            "pareto_rank": self.pareto_rank,
            "composite_score": self.composite_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "update_count": self.update_count,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GoldenSignature":
        return cls(
            name=data["name"],
            description=data["description"],
            primary_targets=[TargetConfig.from_dict(t) for t in data["primary_targets"]],
            secondary_targets=[TargetConfig.from_dict(t) for t in data["secondary_targets"]],
            optimal_params=data["optimal_params"],
            predicted_outcomes=data["predicted_outcomes"],
            pareto_rank=data.get("pareto_rank", 1),
            composite_score=data.get("composite_score", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            update_count=data.get("update_count", 0),
        )
    
    def get_target_summary(self) -> str:
        """Get a human-readable summary of targets."""
        primary = ", ".join([AVAILABLE_TARGETS[t.target_name].display_name 
                           for t in self.primary_targets])
        secondary = ", ".join([AVAILABLE_TARGETS[t.target_name].display_name 
                             for t in self.secondary_targets]) if self.secondary_targets else "None"
        return f"Primary: {primary} | Secondary: {secondary}"


# ---------------------------------------------------------------------------
# Multi-Objective Optimization Engine
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization engine using Pareto-based approach.
    
    Supports finding optimal parameter combinations for any set of targets
    while respecting process constraints.
    """
    
    def __init__(
        self,
        predictor_fn: Callable[[dict], dict] | None = None,
        param_ranges: dict[str, tuple[float, float]] | None = None,
        constraints: list[ProcessConstraint] | None = None,
    ):
        self.predictor_fn = predictor_fn
        self.param_ranges = param_ranges or PARAM_RANGES
        self.constraints = constraints or DEFAULT_CONSTRAINTS
        self._cached_predictions: dict = {}
    
    def set_predictor(self, predictor_fn: Callable[[dict], dict]):
        """Set the prediction function for outcomes."""
        self.predictor_fn = predictor_fn
    
    def generate_candidates(
        self,
        n_samples: int = 5000,
        seed: int | None = None,
        signature_name: str | None = None,
    ) -> pd.DataFrame:
        """Generate candidate parameter configurations using Latin Hypercube Sampling.
        
        Parameters:
        -----------
        n_samples : int
            Number of candidate configurations to generate
        seed : int, optional
            Random seed. If None and signature_name provided, uses signature-based seed
        signature_name : str, optional
            Name of signature (used to generate unique seed for reproducible but varied results)
        """
        param_names = list(self.param_ranges.keys())
        n_dims = len(param_names)
        
        # Generate unique but reproducible seed per signature
        if seed is None:
            if signature_name:
                seed = hash(signature_name) % (2**31)  # Signature-specific seed
            else:
                seed = 42  # Default fallback
        
        sampler = LatinHypercube(d=n_dims, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        
        scaled_samples = np.zeros_like(unit_samples)
        for i, name in enumerate(param_names):
            low, high = self.param_ranges[name]
            scaled_samples[:, i] = low + unit_samples[:, i] * (high - low)
        
        df = pd.DataFrame(scaled_samples, columns=param_names)
        
        # Add sensor defaults
        for col, value in SENSOR_DEFAULTS.items():
            df[col] = value
        
        return df
    
    def predict_outcomes(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Predict all target outcomes for candidate configurations."""
        if self.predictor_fn is None:
            raise ValueError("No predictor function set. Call set_predictor() first.")
        
        results = []
        for _, row in candidates.iterrows():
            params = row.to_dict()
            
            # Get base predictions
            predictions = self.predictor_fn(params)
            
            # Add derived metrics
            predictions["carbon"] = predictions.get("Energy_kWh", 0) * CARBON_FACTOR
            predictions["energy"] = predictions.get("Energy_kWh", 0)
            predictions["quality"] = predictions.get("Dissolution_Rate", 85)
            
            # Estimate yield based on quality and process parameters
            predictions["yield"] = self._estimate_yield(params, predictions)
            
            # Estimate throughput from machine speed
            predictions["throughput"] = self._estimate_throughput(params)
            
            # Estimate stability from process variance
            predictions["stability"] = self._estimate_stability(params)
            
            results.append({**params, **predictions})
        
        return pd.DataFrame(results)
    
    def _estimate_yield(self, params: dict, predictions: dict) -> float:
        """Estimate production yield based on quality and process parameters."""
        quality = predictions.get("quality", 85)
        
        # Yield is influenced by:
        # - Quality (higher dissolution = better yield)
        # - Moisture content (optimal around 2-2.5%)
        # - Compression force (optimal around 14-16 kN)
        
        base_yield = 85.0
        quality_factor = (quality - 85) / 15 * 5  # +/- 5% based on quality
        
        moisture = params.get("Moisture_Content", 2.25)
        moisture_optimal = 2.25
        moisture_factor = -abs(moisture - moisture_optimal) * 2  # Penalty for deviation
        
        compression = params.get("Compression_Force", 15)
        compression_optimal = 15
        compression_factor = -abs(compression - compression_optimal) * 0.5
        
        yield_estimate = base_yield + quality_factor + moisture_factor + compression_factor
        return np.clip(yield_estimate, 80, 99)
    
    def _estimate_throughput(self, params: dict) -> float:
        """Estimate throughput from machine parameters."""
        machine_speed = params.get("Machine_Speed", 55)
        # Throughput scales with machine speed
        return 150 + (machine_speed - 40) * 5  # 150-300 tablets/min range
    
    def _estimate_stability(self, params: dict) -> float:
        """Estimate process stability from parameter ranges."""
        # Stability is higher when parameters are closer to optimal ranges
        stability = 90.0
        
        # Penalize extreme values
        for param, (low, high) in self.param_ranges.items():
            if param in params:
                val = params[param]
                mid = (low + high) / 2
                range_span = (high - low) / 2
                deviation = abs(val - mid) / range_span
                stability -= deviation * 2  # Small penalty for deviation
        
        return np.clip(stability, 70, 99)
    
    def check_constraints(
        self,
        outcomes: dict,
        constraints: list[ProcessConstraint] | None = None,
    ) -> tuple[bool, float]:
        """
        Check if outcomes satisfy constraints.
        
        Returns:
            (is_feasible, penalty_score)
            - is_feasible: True if all hard constraints are satisfied
            - penalty_score: Sum of penalties for soft constraint violations
        """
        constraints = constraints or self.constraints
        is_feasible = True
        penalty = 0.0
        
        for constraint in constraints:
            value = outcomes.get(constraint.parameter, 0)
            
            satisfied = True
            if constraint.operator == ">=":
                satisfied = value >= constraint.value
            elif constraint.operator == "<=":
                satisfied = value <= constraint.value
            elif constraint.operator == ">":
                satisfied = value > constraint.value
            elif constraint.operator == "<":
                satisfied = value < constraint.value
            elif constraint.operator == "==":
                satisfied = abs(value - constraint.value) < 1e-6
            
            if not satisfied:
                if constraint.is_hard:
                    is_feasible = False
                else:
                    # Calculate violation magnitude
                    if constraint.operator in [">=", ">"]:
                        violation = constraint.value - value
                    else:
                        violation = value - constraint.value
                    penalty += constraint.penalty_weight * max(0, violation)
        
        return is_feasible, penalty
    
    def compute_pareto_frontier(
        self,
        df: pd.DataFrame,
        targets: list[TargetConfig],
    ) -> pd.DataFrame:
        """
        Compute Pareto-optimal solutions for the given targets.
        
        A solution is Pareto-optimal if no other solution dominates it
        (i.e., is better in all objectives).
        """
        # Get normalized scores for each target
        n = len(df)
        scores = np.zeros((n, len(targets)))
        
        for i, target in enumerate(targets):
            target_def = AVAILABLE_TARGETS[target.target_name]
            values = df[target.target_name].values
            scores[:, i] = np.array([target_def.normalize(v) for v in values])
            scores[:, i] *= target.weight  # Apply weight
        
        # Find Pareto frontier using non-dominated sorting
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not pareto_mask[i]:
                continue
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                # Check if j dominates i
                if np.all(scores[j] >= scores[i]) and np.any(scores[j] > scores[i]):
                    pareto_mask[i] = False
                    break
        
        pareto_df = df[pareto_mask].copy()
        pareto_df["pareto_rank"] = 1
        
        return pareto_df
    
    def compute_composite_score(
        self,
        outcomes: dict,
        primary_targets: list[TargetConfig],
        secondary_targets: list[TargetConfig],
    ) -> float:
        """Compute weighted composite score for a configuration."""
        score = 0.0
        total_weight = 0.0
        
        # Primary targets have 2x weight multiplier
        for target in primary_targets:
            target_def = AVAILABLE_TARGETS[target.target_name]
            value = outcomes.get(target.target_name, 0)
            norm_value = target_def.normalize(value)
            weight = target.weight * 2.0  # Primary weight multiplier
            score += norm_value * weight
            total_weight += weight
        
        # Secondary targets
        for target in secondary_targets:
            target_def = AVAILABLE_TARGETS[target.target_name]
            value = outcomes.get(target.target_name, 0)
            norm_value = target_def.normalize(value)
            score += norm_value * target.weight
            total_weight += target.weight
        
        return (score / total_weight) * 100 if total_weight > 0 else 0.0
    
    def optimize(
        self,
        primary_targets: list[TargetConfig],
        secondary_targets: list[TargetConfig] | None = None,
        n_samples: int = 5000,
        constraints: list[ProcessConstraint] | None = None,
        return_pareto: bool = True,
        top_k: int = 10,
        signature_name: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Run multi-objective optimization.
        
        Parameters:
        -----------
        primary_targets : list[TargetConfig]
            Primary optimization objectives (highest priority)
        secondary_targets : list[TargetConfig], optional
            Secondary objectives to consider
        n_samples : int
            Number of candidate configurations to generate
        constraints : list[ProcessConstraint], optional
            Additional constraints beyond defaults
        return_pareto : bool
            Whether to compute and return Pareto frontier
        top_k : int
            Number of top solutions to return
        signature_name : str, optional
            Name of signature (used for reproducible but varied candidate generation)
            
        Returns:
        --------
        (top_solutions_df, pareto_df)
        """
        secondary_targets = secondary_targets or []
        all_targets = primary_targets + secondary_targets
        
        # Merge constraints
        all_constraints = list(self.constraints)
        if constraints:
            all_constraints.extend(constraints)
        
        # Generate candidates with signature-specific seed for diversity
        candidates = self.generate_candidates(n_samples=n_samples, signature_name=signature_name)
        
        # Predict outcomes
        results = self.predict_outcomes(candidates)
        
        # Filter by hard constraints and calculate penalties
        feasible_mask = []
        penalties = []
        
        for _, row in results.iterrows():
            outcomes = row.to_dict()
            is_feasible, penalty = self.check_constraints(outcomes, all_constraints)
            feasible_mask.append(is_feasible)
            penalties.append(penalty)
        
        results["constraint_penalty"] = penalties
        feasible_df = results[feasible_mask].copy()
        
        if len(feasible_df) == 0:
            # Relax constraints if no feasible solutions
            print("Warning: No feasible solutions found. Relaxing hard constraints.")
            feasible_df = results.nsmallest(n_samples // 10, "constraint_penalty")
        
        # Compute composite scores
        composite_scores = []
        for _, row in feasible_df.iterrows():
            outcomes = row.to_dict()
            score = self.compute_composite_score(outcomes, primary_targets, secondary_targets)
            # Penalize based on constraint violations
            score -= row.get("constraint_penalty", 0) * 0.1
            composite_scores.append(score)
        
        feasible_df["composite_score"] = composite_scores
        
        # Get Pareto frontier if requested
        pareto_df = None
        if return_pareto and len(all_targets) > 1:
            pareto_df = self.compute_pareto_frontier(feasible_df, all_targets)
            pareto_df = pareto_df.sort_values("composite_score", ascending=False)
        
        # Get top solutions
        top_solutions = feasible_df.nlargest(top_k, "composite_score")
        
        return top_solutions, pareto_df


# ---------------------------------------------------------------------------
# Golden Signature Framework
# ---------------------------------------------------------------------------

class GoldenSignatureFramework:
    """
    Framework for managing Golden Signatures - optimized configurations
    for various multi-objective target combinations.
    
    Provides:
    - Creating new Golden Signatures for custom target combinations
    - Storing and retrieving benchmark configurations
    - Comparing signatures and tracking improvements
    """
    
    # Predefined signature templates
    PREDEFINED_SIGNATURES = {
        "Best Yield - Lowest Energy": {
            "description": "Maximize production yield while minimizing energy consumption",
            "primary": [
                TargetConfig("yield", "primary", weight=1.0),
                TargetConfig("energy", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("quality", "secondary", weight=0.5),
            ],
        },
        "Optimal Quality - Best Yield": {
            "description": "Achieve highest quality with maximum yield",
            "primary": [
                TargetConfig("quality", "primary", weight=1.0),
                TargetConfig("yield", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("energy", "secondary", weight=0.3),
            ],
        },
        "Max Performance - Min Environmental Impact": {
            "description": "Maximum throughput and quality with minimal carbon footprint",
            "primary": [
                TargetConfig("throughput", "primary", weight=1.0),
                TargetConfig("carbon", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("quality", "secondary", weight=0.5),
                TargetConfig("stability", "secondary", weight=0.3),
            ],
        },
        "Energy Champion": {
            "description": "Minimize energy consumption while meeting quality thresholds",
            "primary": [
                TargetConfig("energy", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("quality", "secondary", weight=0.5, constraint_min=85.0),
                TargetConfig("yield", "secondary", weight=0.3),
            ],
        },
        "Quality Excellence": {
            "description": "Maximize dissolution rate and product quality",
            "primary": [
                TargetConfig("quality", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("stability", "secondary", weight=0.5),
                TargetConfig("energy", "secondary", weight=0.3),
            ],
        },
        "Sustainability Focus": {
            "description": "Minimize environmental impact while maintaining quality",
            "primary": [
                TargetConfig("carbon", "primary", weight=1.0),
                TargetConfig("energy", "primary", weight=0.8),
            ],
            "secondary": [
                TargetConfig("quality", "secondary", weight=0.5, constraint_min=85.0),
            ],
        },
        "Balanced Excellence": {
            "description": "Balanced optimization across all key metrics",
            "primary": [
                TargetConfig("quality", "primary", weight=1.0),
                TargetConfig("energy", "primary", weight=1.0),
                TargetConfig("yield", "primary", weight=1.0),
            ],
            "secondary": [
                TargetConfig("carbon", "secondary", weight=0.5),
            ],
        },
        "High Throughput Production": {
            "description": "Maximize production capacity with acceptable quality",
            "primary": [
                TargetConfig("throughput", "primary", weight=1.0),
                TargetConfig("yield", "primary", weight=0.8),
            ],
            "secondary": [
                TargetConfig("quality", "secondary", weight=0.5, constraint_min=85.0),
                TargetConfig("energy", "secondary", weight=0.3),
            ],
        },
    }
    
    def __init__(
        self,
        predictor_fn: Callable[[dict], dict] | None = None,
        benchmarks_path: Path | str | None = None,
    ):
        self.benchmarks_path = Path(benchmarks_path) if benchmarks_path else GOLDEN_BENCHMARKS_PATH
        self.optimizer = MultiObjectiveOptimizer(predictor_fn=predictor_fn)
        self._signatures: dict[str, GoldenSignature] = {}
        self._original_signatures: dict[str, dict] = {}  # Store original for reset
        self._load_benchmarks()
        # Fill missing predictions if predictor is available
        if predictor_fn is not None:
            self._fill_missing_predictions()
        # Store original state after initial load
        self._store_original_state()
    
    def _store_original_state(self):
        """Store current signatures as original state for reset capability."""
        import copy
        self._original_signatures = {
            name: sig.to_dict() for name, sig in self._signatures.items()
        }
    
    def restore_original_benchmarks(self):
        """Restore benchmarks to their original state (before any updates)."""
        self._signatures = {
            name: GoldenSignature.from_dict(data) 
            for name, data in self._original_signatures.items()
        }
        # Re-fill predictions if predictor available
        if self.optimizer.predictor_fn is not None:
            self._fill_missing_predictions()
    
    def set_predictor(self, predictor_fn: Callable[[dict], dict]):
        """Set the prediction function."""
        self.optimizer.set_predictor(predictor_fn)
        # Recompute missing predictions for loaded signatures
        self._fill_missing_predictions()
    
    def _estimate_yield(self, params: dict, quality: float) -> float:
        """Estimate production yield based on parameters and quality."""
        base_yield = 85.0
        quality_factor = (quality - 85) / 15 * 5
        moisture = params.get("Moisture_Content", 2.25)
        moisture_factor = -abs(moisture - 2.25) * 2
        return float(np.clip(base_yield + quality_factor + moisture_factor, 80, 99))
    
    def _estimate_throughput(self, params: dict) -> float:
        """Estimate throughput based on parameters."""
        machine_speed = params.get("Machine_Speed", 55)
        return 150 + (machine_speed - 40) * 5
    
    def _estimate_stability(self, params: dict) -> float:
        """Estimate process stability based on parameters."""
        stability = 90.0
        for param, (low, high) in PARAM_RANGES.items():
            if param in params:
                val = params[param]
                mid = (low + high) / 2
                range_size = (high - low) / 2
                deviation = abs(val - mid) / range_size
                stability -= deviation * 2
        return float(np.clip(stability, 70, 99))
    
    def _fill_missing_predictions(self):
        """Fill in missing predicted outcomes for signatures using the predictor."""
        if self.optimizer.predictor_fn is None:
            return
        
        updated = False
        for name, sig in self._signatures.items():
            # Check if predictions are missing
            has_energy = "energy" in sig.predicted_outcomes or "Energy_kWh" in sig.predicted_outcomes
            has_quality = "quality" in sig.predicted_outcomes or "Dissolution_Rate" in sig.predicted_outcomes
            has_yield = "yield" in sig.predicted_outcomes
            has_throughput = "throughput" in sig.predicted_outcomes
            has_stability = "stability" in sig.predicted_outcomes
            
            if not has_energy or not has_quality or not has_yield or not has_throughput or not has_stability:
                try:
                    # Build params with sensor defaults
                    params = sig.optimal_params.copy()
                    for col, value in SENSOR_DEFAULTS.items():
                        if col not in params:
                            params[col] = value
                    
                    # Get core predictions from model
                    preds = self.optimizer.predictor_fn(params)
                    
                    # Update core outcomes
                    energy = preds.get("Energy_kWh", sig.predicted_outcomes.get("energy", 90))
                    quality = preds.get("Dissolution_Rate", sig.predicted_outcomes.get("quality", 85))
                    
                    sig.predicted_outcomes["Energy_kWh"] = energy
                    sig.predicted_outcomes["Dissolution_Rate"] = quality
                    sig.predicted_outcomes["energy"] = energy
                    sig.predicted_outcomes["quality"] = quality
                    sig.predicted_outcomes["carbon"] = energy * CARBON_FACTOR
                    
                    # Add derived outcomes estimates
                    sig.predicted_outcomes["yield"] = self._estimate_yield(params, quality)
                    sig.predicted_outcomes["throughput"] = self._estimate_throughput(params)
                    sig.predicted_outcomes["stability"] = self._estimate_stability(params)
                    
                    # Recalculate composite score with all targets
                    sig.composite_score = self.optimizer.compute_composite_score(
                        sig.predicted_outcomes,
                        sig.primary_targets,
                        sig.secondary_targets,
                    )
                    
                    updated = True
                except Exception as e:
                    print(f"Warning: Could not compute predictions for '{name}': {e}")
        
        if updated:
            self._save_benchmarks()
    
    def _load_benchmarks(self):
        """Load existing benchmarks from file."""
        if self.benchmarks_path.exists():
            try:
                with open(self.benchmarks_path, "r") as f:
                    data = json.load(f)
                
                # Handle both old and new format
                signatures_data = data.get("golden_signatures", data.get("benchmarks", {}))
                
                for name, sig_data in signatures_data.items():
                    # Convert old format to new if needed
                    if "primary_targets" not in sig_data:
                        sig_data = self._convert_old_format(name, sig_data)
                    self._signatures[name] = GoldenSignature.from_dict(sig_data)
            except Exception as e:
                print(f"Warning: Could not load benchmarks: {e}")
    
    def _convert_old_format(self, name: str, old_data: dict) -> dict:
        """Convert old benchmark format to new GoldenSignature format."""
        weights = old_data.get("weights", {"quality": 0.4, "energy": 0.4, "yield": 0.2})
        
        # Convert weights to target configs
        primary_targets = []
        secondary_targets = []
        
        for target_name, weight in weights.items():
            if weight >= 0.4:
                primary_targets.append(TargetConfig(target_name, "primary", weight).to_dict())
            else:
                secondary_targets.append(TargetConfig(target_name, "secondary", weight).to_dict())
        
        return {
            "name": name,
            "description": old_data.get("description", f"Converted from legacy format"),
            "primary_targets": primary_targets,
            "secondary_targets": secondary_targets,
            "optimal_params": old_data.get("best_config", {}),
            "predicted_outcomes": {
                "composite_score": old_data.get("best_score", 0),
            },
            "pareto_rank": 1,
            "composite_score": old_data.get("best_score", 0),
            "created_at": old_data.get("created_at", datetime.now().isoformat()),
            "updated_at": old_data.get("updated_at", datetime.now().isoformat()),
            "update_count": old_data.get("update_count", 0),
        }
    
    def _save_benchmarks(self):
        """Save benchmarks to file."""
        self.benchmarks_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "golden_signatures": {
                name: sig.to_dict() for name, sig in self._signatures.items()
            },
            "metadata": {
                "version": "2.0",
                "last_updated": datetime.now().isoformat(),
                "available_targets": list(AVAILABLE_TARGETS.keys()),
            }
        }
        
        with open(self.benchmarks_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_available_targets(self) -> dict[str, TargetDefinition]:
        """Get all available optimization targets."""
        return AVAILABLE_TARGETS
    
    def get_predefined_templates(self) -> dict[str, dict]:
        """Get predefined signature templates."""
        return self.PREDEFINED_SIGNATURES
    
    def get_signature(self, name: str) -> GoldenSignature | None:
        """Get a signature by name."""
        return self._signatures.get(name)
    
    def list_signatures(self) -> list[str]:
        """List all stored signature names."""
        return list(self._signatures.keys())
    
    def get_all_signatures(self) -> dict[str, GoldenSignature]:
        """Get all stored signatures."""
        return self._signatures.copy()
    
    def create_signature(
        self,
        name: str,
        primary_targets: list[TargetConfig] | list[str],
        secondary_targets: list[TargetConfig] | list[str] | None = None,
        description: str | None = None,
        constraints: list[ProcessConstraint] | None = None,
        n_samples: int = 5000,
    ) -> GoldenSignature:
        """
        Create a new Golden Signature for the specified target combination.
        
        Parameters:
        -----------
        name : str
            Name for this signature
        primary_targets : list
            Primary targets (list of TargetConfig or target names as strings)
        secondary_targets : list, optional
            Secondary targets
        description : str, optional
            Human-readable description
        constraints : list[ProcessConstraint], optional
            Additional constraints
        n_samples : int
            Number of optimization samples
            
        Returns:
        --------
        GoldenSignature with optimized parameters
        """
        # Convert string targets to TargetConfig if needed
        if primary_targets and isinstance(primary_targets[0], str):
            primary_targets = [
                TargetConfig(t, "primary", weight=1.0) for t in primary_targets
            ]
        
        if secondary_targets:
            if isinstance(secondary_targets[0], str):
                secondary_targets = [
                    TargetConfig(t, "secondary", weight=0.5) for t in secondary_targets
                ]
        else:
            secondary_targets = []
        
        # Generate description if not provided
        if description is None:
            primary_names = [AVAILABLE_TARGETS[t.target_name].display_name 
                          for t in primary_targets]
            secondary_names = [AVAILABLE_TARGETS[t.target_name].display_name 
                             for t in secondary_targets]
            description = f"Optimize for {' and '.join(primary_names)}"
            if secondary_names:
                description += f" with {', '.join(secondary_names)} as secondary"
        
        # Run optimization with signature-specific seed for diversity
        top_solutions, pareto_df = self.optimizer.optimize(
            primary_targets=primary_targets,
            secondary_targets=secondary_targets,
            n_samples=n_samples,
            constraints=constraints,
            signature_name=name,
        )
        
        if len(top_solutions) == 0:
            raise ValueError("Optimization failed to find feasible solutions")
        
        # Get best solution
        best = top_solutions.iloc[0]
        
        # Extract optimal parameters
        optimal_params = {
            param: float(best[param]) for param in PARAM_RANGES.keys()
        }
        
        # Extract predicted outcomes
        predicted_outcomes = {
            target: float(best.get(target, 0))
            for target in AVAILABLE_TARGETS.keys()
            if target in best
        }
        predicted_outcomes["Energy_kWh"] = float(best.get("energy", 0))
        predicted_outcomes["Dissolution_Rate"] = float(best.get("quality", 85))
        predicted_outcomes["Carbon_kg"] = float(best.get("carbon", 0))
        
        # Create signature
        signature = GoldenSignature(
            name=name,
            description=description,
            primary_targets=primary_targets,
            secondary_targets=secondary_targets,
            optimal_params=optimal_params,
            predicted_outcomes=predicted_outcomes,
            pareto_rank=1,
            composite_score=float(best["composite_score"]),
        )
        
        # Store and save
        self._signatures[name] = signature
        self._save_benchmarks()
        
        return signature
    
    def create_from_template(
        self,
        template_name: str,
        custom_name: str | None = None,
        n_samples: int = 5000,
    ) -> GoldenSignature:
        """Create a signature from a predefined template."""
        if template_name not in self.PREDEFINED_SIGNATURES:
            available = list(self.PREDEFINED_SIGNATURES.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")
        
        template = self.PREDEFINED_SIGNATURES[template_name]
        name = custom_name or template_name
        
        return self.create_signature(
            name=name,
            primary_targets=template["primary"],
            secondary_targets=template.get("secondary", []),
            description=template["description"],
            n_samples=n_samples,
        )
    
    def create_custom_signature(
        self,
        name: str,
        primary_target_names: list[str],
        primary_weights: list[float] | None = None,
        secondary_target_names: list[str] | None = None,
        secondary_weights: list[float] | None = None,
        description: str | None = None,
        n_samples: int = 5000,
    ) -> GoldenSignature:
        """
        Create a custom signature with user-specified targets and weights.
        
        Parameters:
        -----------
        name : str
            Name for this signature
        primary_target_names : list[str]
            Names of primary targets (e.g., ["energy", "quality"])
        primary_weights : list[float], optional
            Weights for primary targets (default: equal weights)
        secondary_target_names : list[str], optional
            Names of secondary targets
        secondary_weights : list[float], optional
            Weights for secondary targets (default: 0.5 each)
        description : str, optional
            Human-readable description
        n_samples : int
            Number of optimization samples
        """
        # Validate targets
        for t in primary_target_names:
            if t not in AVAILABLE_TARGETS:
                raise ValueError(f"Unknown target '{t}'. Available: {list(AVAILABLE_TARGETS.keys())}")
        
        if secondary_target_names:
            for t in secondary_target_names:
                if t not in AVAILABLE_TARGETS:
                    raise ValueError(f"Unknown target '{t}'. Available: {list(AVAILABLE_TARGETS.keys())}")
        
        # Build target configs
        primary_weights = primary_weights or [1.0] * len(primary_target_names)
        primary_targets = [
            TargetConfig(name, "primary", weight=w)
            for name, w in zip(primary_target_names, primary_weights)
        ]
        
        secondary_targets = []
        if secondary_target_names:
            secondary_weights = secondary_weights or [0.5] * len(secondary_target_names)
            secondary_targets = [
                TargetConfig(name, "secondary", weight=w)
                for name, w in zip(secondary_target_names, secondary_weights)
            ]
        
        return self.create_signature(
            name=name,
            primary_targets=primary_targets,
            secondary_targets=secondary_targets,
            description=description,
            n_samples=n_samples,
        )
    
    def compare_signatures(
        self,
        signature_names: list[str],
    ) -> pd.DataFrame:
        """Compare multiple signatures side by side."""
        rows = []
        
        for name in signature_names:
            sig = self._signatures.get(name)
            if sig is None:
                continue
            
            row = {"Signature": name}
            row["Description"] = sig.description
            row["Composite Score"] = sig.composite_score
            
            # Add target outcomes
            for target_name, target_def in AVAILABLE_TARGETS.items():
                value = sig.predicted_outcomes.get(target_name, None)
                if value is not None:
                    row[target_def.display_name] = f"{value:.2f} {target_def.unit}"
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def delete_signature(self, name: str) -> bool:
        """Delete a signature by name."""
        if name in self._signatures:
            del self._signatures[name]
            self._save_benchmarks()
            return True
        return False
    
    def update_signature_if_better(
        self,
        name: str,
        new_params: dict[str, float],
        new_outcomes: dict[str, float],
    ) -> bool:
        """
        Update a signature if the new configuration is better.
        
        This is used by the continuous learning module.
        
        Returns:
        --------
        True if signature was updated, False otherwise
        """
        sig = self._signatures.get(name)
        if sig is None:
            return False
        
        # Compute score for new configuration
        new_score = self.optimizer.compute_composite_score(
            new_outcomes,
            sig.primary_targets,
            sig.secondary_targets,
        )
        
        if new_score > sig.composite_score:
            sig.optimal_params = new_params
            sig.predicted_outcomes = new_outcomes
            sig.composite_score = new_score
            sig.updated_at = datetime.now().isoformat()
            sig.update_count += 1
            self._save_benchmarks()
            return True
        
        return False


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def create_golden_signature(
    name: str,
    primary_targets: list[str],
    secondary_targets: list[str] | None = None,
    predictor_fn: Callable[[dict], dict] | None = None,
) -> GoldenSignature:
    """
    Convenience function to create a Golden Signature.
    
    Example:
    --------
    >>> signature = create_golden_signature(
    ...     name="Energy-Quality Balance",
    ...     primary_targets=["energy", "quality"],
    ...     secondary_targets=["yield"]
    ... )
    """
    framework = GoldenSignatureFramework(predictor_fn=predictor_fn)
    return framework.create_custom_signature(
        name=name,
        primary_target_names=primary_targets,
        secondary_target_names=secondary_targets,
    )


def get_optimal_params_for_targets(
    primary_targets: list[str],
    secondary_targets: list[str] | None = None,
    predictor_fn: Callable[[dict], dict] | None = None,
    n_samples: int = 5000,
) -> dict[str, float]:
    """
    Get optimal parameters for a given target combination without storing.
    
    Returns:
    --------
    dict with optimal parameter values
    """
    framework = GoldenSignatureFramework(predictor_fn=predictor_fn)
    
    primary_configs = [TargetConfig(t, "primary", 1.0) for t in primary_targets]
    secondary_configs = [TargetConfig(t, "secondary", 0.5) for t in (secondary_targets or [])]
    
    # Generate pseudo signature name from targets for seed diversity
    sig_name = "_".join(primary_targets)
    
    top_solutions, _ = framework.optimizer.optimize(
        primary_targets=primary_configs,
        secondary_targets=secondary_configs,
        n_samples=n_samples,
        signature_name=sig_name,
    )
    
    best = top_solutions.iloc[0]
    return {param: float(best[param]) for param in PARAM_RANGES.keys()}


# ---------------------------------------------------------------------------
# Module-level framework instance
# ---------------------------------------------------------------------------

_framework_instance: GoldenSignatureFramework | None = None


def get_framework(predictor_fn: Callable[[dict], dict] | None = None) -> GoldenSignatureFramework:
    """Get or create the global framework instance."""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = GoldenSignatureFramework(predictor_fn=predictor_fn)
    elif predictor_fn is not None:
        _framework_instance.set_predictor(predictor_fn)
    return _framework_instance


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("Golden Signature Framework Demo")
    print("=" * 60)
    
    # Create a simple mock predictor for demo
    def mock_predictor(params: dict) -> dict:
        # Simple mock predictions
        energy = 70 + params.get("Drying_Time", 25) * 0.8 + params.get("Granulation_Time", 60) * 0.3
        quality = 80 + params.get("Binder_Amount", 8) * 1.2 - params.get("Machine_Speed", 55) * 0.05
        return {
            "Energy_kWh": energy,
            "Dissolution_Rate": min(quality, 99),
        }
    
    # Initialize framework
    framework = GoldenSignatureFramework(predictor_fn=mock_predictor)
    
    print("\n1. Available Targets:")
    for name, target in framework.get_available_targets().items():
        print(f"   - {target.display_name} ({target.direction.value})")
    
    print("\n2. Predefined Templates:")
    for name in framework.get_predefined_templates():
        print(f"   - {name}")
    
    print("\n3. Creating custom signature: 'Demo - Energy + Quality'")
    sig = framework.create_custom_signature(
        name="Demo - Energy + Quality",
        primary_target_names=["energy", "quality"],
        secondary_target_names=["yield"],
        n_samples=1000,
    )
    
    print(f"\n   Signature: {sig.name}")
    print(f"   Description: {sig.description}")
    print(f"   Composite Score: {sig.composite_score:.2f}")
    print(f"   Optimal Parameters:")
    for param, value in sig.optimal_params.items():
        print(f"      {param}: {value:.2f}")
    
    print("\n4. Stored Signatures:")
    for name in framework.list_signatures():
        print(f"   - {name}")
