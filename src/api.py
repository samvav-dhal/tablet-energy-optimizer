"""
FastAPI service for manufacturing energy optimization.

Provides prediction endpoints for energy consumption and dissolution rate
using trained XGBoost models, plus Golden Signature Framework endpoints
for multi-objective optimization.

Run with:
    uvicorn src.api:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .golden_signature import (
    GoldenSignatureFramework,
    TargetConfig,
    ProcessConstraint,
    AVAILABLE_TARGETS,
    get_framework,
)
from .train_models import add_engineered_features, ENGINEERED_FEATURES
from .adaptive_optimizer import (
    AdaptiveOptimizationOrchestrator,
    AdaptiveConfigManager,
    AdaptiveSession,
    AdaptiveEvent,
    AdaptiveEventType,
    AlertSeverity,
    ParameterAdjustment,
    TrajectoryPoint,
    get_orchestrator,
    create_orchestrator,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENERGY_MODEL_PATH = os.path.join(BASE_DIR, "models", "energy_model.pkl")
DISSOLUTION_MODEL_PATH = os.path.join(BASE_DIR, "models", "dissolution_model.pkl")

CARBON_FACTOR = 0.7  # kg CO2 per kWh


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """Input features for energy and dissolution prediction."""

    Granulation_Time: float = Field(..., ge=0, description="Granulation time (minutes)")
    Binder_Amount: float = Field(..., ge=0, description="Binder amount (kg)")
    Drying_Temp: float = Field(..., ge=0, description="Drying temperature (°C)")
    Drying_Time: float = Field(..., ge=0, description="Drying time (minutes)")
    Compression_Force: float = Field(..., ge=0, description="Compression force (kN)")
    Machine_Speed: float = Field(..., ge=0, description="Machine speed (RPM)")
    Lubricant_Conc: float = Field(..., ge=0, description="Lubricant concentration (%)")
    Moisture_Content: float = Field(..., ge=0, description="Moisture content (%)")
    avg_power: float = Field(..., ge=0, description="Average power consumption (kW)")
    max_power: float = Field(..., ge=0, description="Maximum power consumption (kW)")
    power_std: float = Field(..., ge=0, description="Power consumption std deviation")
    avg_temperature: float = Field(..., description="Average temperature (°C)")
    max_temperature: float = Field(..., description="Maximum temperature (°C)")
    avg_vibration: float = Field(..., ge=0, description="Average vibration (mm/s)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Granulation_Time": 60,
                    "Binder_Amount": 8.0,
                    "Drying_Temp": 65,
                    "Drying_Time": 25,
                    "Compression_Force": 15,
                    "Machine_Speed": 55,
                    "Lubricant_Conc": 1.0,
                    "Moisture_Content": 2.5,
                    "avg_power": 23.0,
                    "max_power": 59.0,
                    "power_std": 16.3,
                    "avg_temperature": 35.0,
                    "max_temperature": 68.0,
                    "avg_vibration": 3.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Prediction results with energy, dissolution, and carbon emissions."""

    Energy_kWh: float = Field(..., description="Predicted energy consumption (kWh)")
    Dissolution_Rate: float = Field(..., description="Predicted dissolution rate (%)")
    Carbon_kg: float = Field(..., description="Estimated carbon emissions (kg CO2)")


# Golden Signature Request/Response Models

class TargetConfigRequest(BaseModel):
    """Configuration for a single optimization target."""
    target_name: str = Field(..., description="Target name (energy, quality, carbon, yield, throughput, stability)")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="Weight for this target")
    constraint_min: Optional[float] = Field(default=None, description="Minimum constraint value")
    constraint_max: Optional[float] = Field(default=None, description="Maximum constraint value")


class CreateSignatureRequest(BaseModel):
    """Request to create a new Golden Signature."""
    name: str = Field(..., description="Name for the signature")
    primary_targets: list[str] = Field(..., description="List of primary target names")
    primary_weights: Optional[list[float]] = Field(default=None, description="Weights for primary targets")
    secondary_targets: Optional[list[str]] = Field(default=None, description="List of secondary target names")
    secondary_weights: Optional[list[float]] = Field(default=None, description="Weights for secondary targets")
    description: Optional[str] = Field(default=None, description="Human-readable description")
    n_samples: int = Field(default=5000, ge=100, le=50000, description="Number of optimization samples")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Best Yield - Lowest Energy",
                    "primary_targets": ["yield", "energy"],
                    "secondary_targets": ["quality"],
                    "description": "Maximize yield while minimizing energy",
                    "n_samples": 5000
                }
            ]
        }
    }


class SignatureResponse(BaseModel):
    """Response containing a Golden Signature."""
    name: str
    description: str
    primary_targets: list[dict]
    secondary_targets: list[dict]
    optimal_params: dict[str, float]
    predicted_outcomes: dict[str, float]
    composite_score: float
    pareto_rank: int
    created_at: str
    updated_at: str
    update_count: int


class TargetDefinitionResponse(BaseModel):
    """Information about an available target."""
    name: str
    display_name: str
    direction: str
    min_value: float
    max_value: float
    unit: str
    description: str


class TemplateResponse(BaseModel):
    """Information about a predefined signature template."""
    name: str
    description: str
    primary_targets: list[str]
    secondary_targets: list[str]


# ---------------------------------------------------------------------------
# Model storage
# ---------------------------------------------------------------------------

class ModelStore:
    """Container for loaded ML models and Golden Signature framework."""

    energy_model = None
    dissolution_model = None
    gs_framework = None
    adaptive_orchestrator = None


def make_predictor_fn():
    """Create a predictor function that uses the loaded models."""
    import pandas as pd
    
    def predictor(params: dict) -> dict:
        if ModelStore.energy_model is None or ModelStore.dissolution_model is None:
            raise RuntimeError("Models not loaded")
        
        # Build feature array with engineered features
        base_features = [
            "Granulation_Time", "Binder_Amount", "Drying_Temp", "Drying_Time",
            "Compression_Force", "Machine_Speed", "Lubricant_Conc", "Moisture_Content",
            "avg_power", "max_power", "power_std",
            "avg_temperature", "max_temperature", "avg_vibration"
        ]
        
        # Create DataFrame for feature engineering
        df = pd.DataFrame([params])
        df = add_engineered_features(df)
        
        feature_order = base_features + ENGINEERED_FEATURES
        X = df[feature_order].values
        
        energy_kwh = float(ModelStore.energy_model.predict(X)[0])
        dissolution_rate = float(ModelStore.dissolution_model.predict(X)[0])
        
        return {
            "Energy_kWh": energy_kwh,
            "Dissolution_Rate": dissolution_rate,
        }
    
    return predictor


# ---------------------------------------------------------------------------
# Lifespan handler (model loading)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup, cleanup on shutdown."""
    # Startup: load models
    print("Loading ML models...")
    try:
        ModelStore.energy_model = joblib.load(ENERGY_MODEL_PATH)
        ModelStore.dissolution_model = joblib.load(DISSOLUTION_MODEL_PATH)
        print(f"  Loaded energy model from '{ENERGY_MODEL_PATH}'")
        print(f"  Loaded dissolution model from '{DISSOLUTION_MODEL_PATH}'")
        
        # Initialize Golden Signature Framework
        print("Initializing Golden Signature Framework...")
        predictor_fn = make_predictor_fn()
        ModelStore.gs_framework = get_framework(predictor_fn)
        print(f"  Golden Signature Framework initialized with {len(ModelStore.gs_framework.list_signatures())} signatures")
        
        # Initialize Adaptive Optimization Orchestrator
        print("Initializing Adaptive Optimization Orchestrator...")
        ModelStore.adaptive_orchestrator = get_orchestrator()
        print("  Adaptive Optimization Orchestrator initialized")
        
    except FileNotFoundError as e:
        print(f"ERROR: Model file not found: {e}")
        raise

    yield

    # Shutdown: cleanup (optional)
    print("Shutting down API...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Manufacturing Energy Optimization API",
    description="""
Predict energy consumption, dissolution rate, and carbon emissions for pharmaceutical tablet manufacturing.

## Features
- **Predictions**: Get energy, quality, and carbon predictions for process parameters
- **Golden Signature Framework**: Multi-objective optimization with customizable target combinations

## Golden Signature Target Options
- `energy`: Energy Consumption (minimize)
- `quality`: Dissolution Rate / Quality (maximize)
- `carbon`: Carbon Emissions (minimize)
- `yield`: Production Yield (maximize)
- `throughput`: Production Throughput (maximize)
- `stability`: Process Stability (maximize)
""",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Manufacturing Energy Optimization API"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict energy consumption and dissolution rate for given process parameters.

    Returns predicted Energy_kWh, Dissolution_Rate, and Carbon_kg emissions.
    """
    if ModelStore.energy_model is None or ModelStore.dissolution_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Build feature array in correct order
    features = np.array([[
        request.Granulation_Time,
        request.Binder_Amount,
        request.Drying_Temp,
        request.Drying_Time,
        request.Compression_Force,
        request.Machine_Speed,
        request.Lubricant_Conc,
        request.Moisture_Content,
        request.avg_power,
        request.max_power,
        request.power_std,
        request.avg_temperature,
        request.max_temperature,
        request.avg_vibration,
    ]])

    # Predict
    energy_kwh = float(ModelStore.energy_model.predict(features)[0])
    dissolution_rate = float(ModelStore.dissolution_model.predict(features)[0])
    carbon_kg = energy_kwh * CARBON_FACTOR

    return PredictionResponse(
        Energy_kWh=round(energy_kwh, 4),
        Dissolution_Rate=round(dissolution_rate, 4),
        Carbon_kg=round(carbon_kg, 4),
    )


# ---------------------------------------------------------------------------
# Golden Signature Endpoints
# ---------------------------------------------------------------------------

@app.get("/golden-signature/targets", response_model=list[TargetDefinitionResponse])
async def get_available_targets():
    """Get all available optimization targets.
    
    Returns information about each target including its direction (minimize/maximize),
    typical value ranges, and description.
    """
    targets = []
    for name, target in AVAILABLE_TARGETS.items():
        targets.append(TargetDefinitionResponse(
            name=target.name,
            display_name=target.display_name,
            direction=target.direction.value,
            min_value=target.min_value,
            max_value=target.max_value,
            unit=target.unit,
            description=target.description,
        ))
    return targets


@app.get("/golden-signature/templates", response_model=list[TemplateResponse])
async def get_signature_templates():
    """Get predefined signature templates.
    
    These templates provide common target combinations that can be used
    to quickly create Golden Signatures.
    """
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    templates = []
    for name, template in ModelStore.gs_framework.get_predefined_templates().items():
        templates.append(TemplateResponse(
            name=name,
            description=template["description"],
            primary_targets=[t.target_name if hasattr(t, 'target_name') else t for t in template["primary"]],
            secondary_targets=[t.target_name if hasattr(t, 'target_name') else t for t in template.get("secondary", [])],
        ))
    return templates


@app.get("/golden-signature/list", response_model=list[str])
async def list_signatures():
    """List all stored Golden Signature names."""
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    return ModelStore.gs_framework.list_signatures()


@app.get("/golden-signature/{name}", response_model=SignatureResponse)
async def get_signature(name: str):
    """Get a specific Golden Signature by name."""
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    sig = ModelStore.gs_framework.get_signature(name)
    if sig is None:
        raise HTTPException(status_code=404, detail=f"Signature '{name}' not found")
    
    return SignatureResponse(
        name=sig.name,
        description=sig.description,
        primary_targets=[t.to_dict() for t in sig.primary_targets],
        secondary_targets=[t.to_dict() for t in sig.secondary_targets],
        optimal_params=sig.optimal_params,
        predicted_outcomes=sig.predicted_outcomes,
        composite_score=sig.composite_score,
        pareto_rank=sig.pareto_rank,
        created_at=sig.created_at,
        updated_at=sig.updated_at,
        update_count=sig.update_count,
    )


@app.post("/golden-signature/create", response_model=SignatureResponse)
async def create_signature(request: CreateSignatureRequest):
    """Create a new Golden Signature with custom target combination.
    
    This runs multi-objective optimization to find the best process parameters
    for the specified combination of primary and secondary targets.
    
    **Primary targets** are given highest priority in the optimization.
    **Secondary targets** are considered but with lower weight.
    
    Example target combinations:
    - Best yield with lowest energy: primary=["yield", "energy"]
    - Optimal quality with best yield: primary=["quality", "yield"]
    - Max performance, min environmental impact: primary=["throughput", "carbon"]
    """
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    # Validate targets
    valid_targets = list(AVAILABLE_TARGETS.keys())
    for target in request.primary_targets:
        if target not in valid_targets:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid target '{target}'. Valid targets: {valid_targets}"
            )
    
    if request.secondary_targets:
        for target in request.secondary_targets:
            if target not in valid_targets:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid target '{target}'. Valid targets: {valid_targets}"
                )
    
    try:
        sig = ModelStore.gs_framework.create_custom_signature(
            name=request.name,
            primary_target_names=request.primary_targets,
            primary_weights=request.primary_weights,
            secondary_target_names=request.secondary_targets,
            secondary_weights=request.secondary_weights,
            description=request.description,
            n_samples=request.n_samples,
        )
        
        return SignatureResponse(
            name=sig.name,
            description=sig.description,
            primary_targets=[t.to_dict() for t in sig.primary_targets],
            secondary_targets=[t.to_dict() for t in sig.secondary_targets],
            optimal_params=sig.optimal_params,
            predicted_outcomes=sig.predicted_outcomes,
            composite_score=sig.composite_score,
            pareto_rank=sig.pareto_rank,
            created_at=sig.created_at,
            updated_at=sig.updated_at,
            update_count=sig.update_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.post("/golden-signature/from-template/{template_name}", response_model=SignatureResponse)
async def create_from_template(template_name: str, custom_name: Optional[str] = None):
    """Create a Golden Signature from a predefined template.
    
    Available templates:
    - "Best Yield - Lowest Energy"
    - "Optimal Quality - Best Yield"
    - "Max Performance - Min Environmental Impact"
    - "Energy Champion"
    - "Quality Excellence"
    - "Sustainability Focus"
    - "Balanced Excellence"
    - "High Throughput Production"
    """
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    templates = ModelStore.gs_framework.get_predefined_templates()
    if template_name not in templates:
        raise HTTPException(
            status_code=404, 
            detail=f"Template '{template_name}' not found. Available: {list(templates.keys())}"
        )
    
    try:
        sig = ModelStore.gs_framework.create_from_template(
            template_name=template_name,
            custom_name=custom_name,
            n_samples=5000,
        )
        
        return SignatureResponse(
            name=sig.name,
            description=sig.description,
            primary_targets=[t.to_dict() for t in sig.primary_targets],
            secondary_targets=[t.to_dict() for t in sig.secondary_targets],
            optimal_params=sig.optimal_params,
            predicted_outcomes=sig.predicted_outcomes,
            composite_score=sig.composite_score,
            pareto_rank=sig.pareto_rank,
            created_at=sig.created_at,
            updated_at=sig.updated_at,
            update_count=sig.update_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create signature: {str(e)}")


@app.delete("/golden-signature/{name}")
async def delete_signature(name: str):
    """Delete a Golden Signature by name."""
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    if ModelStore.gs_framework.delete_signature(name):
        return {"message": f"Signature '{name}' deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail=f"Signature '{name}' not found")


@app.get("/golden-signature/compare", response_model=list[dict])
async def compare_signatures(names: str):
    """Compare multiple Golden Signatures side by side.
    
    Pass signature names as comma-separated string.
    Example: /golden-signature/compare?names=Signature1,Signature2
    """
    if ModelStore.gs_framework is None:
        raise HTTPException(status_code=503, detail="Golden Signature Framework not initialized")
    
    name_list = [n.strip() for n in names.split(",")]
    comparison_df = ModelStore.gs_framework.compare_signatures(name_list)
    
    return comparison_df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Adaptive Optimization Pydantic Models
# ---------------------------------------------------------------------------

class AdaptiveSessionRequest(BaseModel):
    """Request to start an adaptive optimization session."""
    batch_id: str = Field(..., description="Unique batch identifier")
    signature_name: str = Field(..., description="Golden Signature name to track against")
    initial_params: dict[str, float] = Field(..., description="Initial process parameters")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "batch_id": "T001",
                    "signature_name": "Energy Champion",
                    "initial_params": {
                        "Granulation_Time": 60,
                        "Binder_Amount": 8.0,
                        "Drying_Temp": 65,
                        "Drying_Time": 25,
                        "Compression_Force": 15,
                        "Machine_Speed": 55,
                        "Lubricant_Conc": 1.0,
                        "Moisture_Content": 2.5
                    }
                }
            ]
        }
    }


class SensorReadingRequest(BaseModel):
    """Sensor reading for adaptive processing."""
    power_kw: float = Field(..., description="Power consumption (kW)")
    temperature_c: float = Field(..., description="Temperature (°C)")
    vibration_mm_s: float = Field(..., description="Vibration (mm/s)")
    time_minutes: int = Field(..., ge=0, description="Time in batch (minutes)")
    phase: str = Field(..., description="Current manufacturing phase")
    features: dict[str, float] = Field(..., description="Complete feature set for prediction")


class PhaseCompletionRequest(BaseModel):
    """Request to mark a phase as complete."""
    phase: str = Field(..., description="Phase name that completed")
    actual_energy: float = Field(..., ge=0, description="Actual energy consumed in phase (kWh)")


class EndSessionRequest(BaseModel):
    """Request to end an adaptive session."""
    final_outcomes: Optional[dict[str, float]] = Field(
        default=None, 
        description="Final actual outcomes (energy, quality, carbon)"
    )


class AdaptiveConfigUpdateRequest(BaseModel):
    """Request to update adaptive configuration."""
    drift_threshold_pct: Optional[float] = Field(default=None, ge=0, le=50)
    anomaly_z_threshold: Optional[float] = Field(default=None, ge=1, le=10)
    adjustment_limit_pct: Optional[float] = Field(default=None, ge=1, le=50)
    auto_apply_mode: Optional[bool] = Field(default=None)


# ---------------------------------------------------------------------------
# Adaptive Optimization Endpoints
# ---------------------------------------------------------------------------

@app.post("/adaptive/start-session")
async def start_adaptive_session(request: AdaptiveSessionRequest):
    """Start a new adaptive optimization session for a batch.
    
    This initiates real-time monitoring of the batch against the specified
    Golden Signature, enabling drift detection, anomaly alerts, and
    parameter adjustment recommendations.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    try:
        session = ModelStore.adaptive_orchestrator.start_session(
            batch_id=request.batch_id,
            signature_name=request.signature_name,
            initial_params=request.initial_params,
        )
        return {
            "status": "session_started",
            "batch_id": session.batch_id,
            "signature_name": session.signature_name,
            "target_outcomes": session.target_outcomes,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")


@app.post("/adaptive/process-reading")
async def process_sensor_reading(request: SensorReadingRequest):
    """Process a single sensor reading through the adaptive pipeline.
    
    Returns drift metrics, anomaly alerts, and parameter adjustment
    recommendations if the batch is deviating from targets.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    result = ModelStore.adaptive_orchestrator.process_reading(
        power_kw=request.power_kw,
        temperature_c=request.temperature_c,
        vibration_mm_s=request.vibration_mm_s,
        time_minutes=request.time_minutes,
        phase=request.phase,
        features=request.features,
    )
    
    if result.get("status") == "no_active_session":
        raise HTTPException(status_code=400, detail="No active session. Start a session first.")
    
    return result


@app.get("/adaptive/status")
async def get_adaptive_status():
    """Get current status of the adaptive optimization orchestrator.
    
    Returns session state, cumulative metrics, and pending adjustments.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    return ModelStore.adaptive_orchestrator.get_status()


@app.get("/adaptive/trajectory")
async def get_batch_trajectory():
    """Get the trajectory data for the current batch.
    
    Returns time-series data of energy consumption, quality estimates,
    and comparison against targets.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    trajectory = ModelStore.adaptive_orchestrator.get_trajectory()
    if not trajectory:
        raise HTTPException(status_code=404, detail="No active session or no trajectory data")
    
    return {"trajectory": trajectory}


@app.get("/adaptive/scenarios")
async def get_projected_scenarios():
    """Get pessimistic, realistic, and optimistic projections for the current batch.
    
    Useful for understanding potential outcomes and planning interventions.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    scenarios = ModelStore.adaptive_orchestrator.get_scenarios()
    if scenarios.get("status") == "no_active_session":
        raise HTTPException(status_code=400, detail="No active session")
    
    return scenarios


@app.post("/adaptive/apply-adjustment/{index}")
async def apply_adjustment(index: int = 0):
    """Apply a pending parameter adjustment recommendation.
    
    Parameters
    ----------
    index : int
        Index of adjustment in pending list (0 = best recommendation)
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    result = ModelStore.adaptive_orchestrator.apply_adjustment(index)
    
    if result["status"] == "no_active_session":
        raise HTTPException(status_code=400, detail="No active session")
    if result["status"] == "no_pending_adjustments":
        raise HTTPException(status_code=404, detail="No pending adjustments to apply")
    if result["status"] == "invalid_index":
        raise HTTPException(status_code=400, detail="Invalid adjustment index")
    if result["status"] == "cannot_apply_mid_batch":
        raise HTTPException(
            status_code=400, 
            detail=f"Parameter '{result['parameter']}' cannot be adjusted mid-batch"
        )
    
    return result


@app.post("/adaptive/reject-adjustment/{index}")
async def reject_adjustment(index: int = 0):
    """Reject a pending parameter adjustment recommendation.
    
    Parameters
    ----------
    index : int
        Index of adjustment in pending list
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    result = ModelStore.adaptive_orchestrator.reject_adjustment(index)
    
    if result["status"] == "no_pending_adjustments":
        raise HTTPException(status_code=404, detail="No pending adjustments")
    if result["status"] == "invalid_index":
        raise HTTPException(status_code=400, detail="Invalid adjustment index")
    
    return result


@app.post("/adaptive/complete-phase")
async def complete_phase(request: PhaseCompletionRequest):
    """Mark a manufacturing phase as complete and update energy budget.
    
    This triggers constraint rebalancing for remaining phases if
    the phase was over or under budget.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    return ModelStore.adaptive_orchestrator.complete_phase(
        phase=request.phase,
        actual_energy=request.actual_energy,
    )


@app.post("/adaptive/end-session")
async def end_adaptive_session(request: EndSessionRequest):
    """End the current adaptive optimization session.
    
    Optionally provide final actual outcomes to feed into the
    continuous learning system.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    result = ModelStore.adaptive_orchestrator.end_session(
        final_outcomes=request.final_outcomes
    )
    
    if result["status"] == "no_active_session":
        raise HTTPException(status_code=400, detail="No active session to end")
    
    return result


@app.get("/adaptive/history")
async def get_adaptation_history(
    event_type: Optional[str] = None,
    batch_id: Optional[str] = None,
    limit: int = 50,
):
    """Get adaptation history events.
    
    Parameters
    ----------
    event_type : str, optional
        Filter by event type (drift_detected, anomaly_detected, adjustment_applied, etc.)
    batch_id : str, optional
        Filter by batch ID
    limit : int
        Maximum number of events to return
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    event_type_enum = None
    if event_type:
        try:
            event_type_enum = AdaptiveEventType(event_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid event type. Valid types: {[e.value for e in AdaptiveEventType]}"
            )
    
    events = ModelStore.adaptive_orchestrator.history.get_events(
        event_type=event_type_enum,
        batch_id=batch_id,
        limit=limit,
    )
    
    return {
        "events": [e.to_dict() for e in events],
        "total": len(events),
    }


@app.get("/adaptive/history/summary")
async def get_history_summary():
    """Get summary statistics of adaptation history."""
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    return ModelStore.adaptive_orchestrator.history.get_summary()


@app.get("/adaptive/config")
async def get_adaptive_config():
    """Get current adaptive optimization configuration."""
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    return ModelStore.adaptive_orchestrator.config.config


@app.put("/adaptive/config")
async def update_adaptive_config(request: AdaptiveConfigUpdateRequest):
    """Update adaptive optimization configuration.
    
    Changes take effect immediately for the current and future sessions.
    """
    if ModelStore.adaptive_orchestrator is None:
        raise HTTPException(status_code=503, detail="Adaptive Orchestrator not initialized")
    
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    if updates:
        ModelStore.adaptive_orchestrator.config.update(updates)
        ModelStore.adaptive_orchestrator.config.save()
    
    return {"status": "updated", "config": ModelStore.adaptive_orchestrator.config.config}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
