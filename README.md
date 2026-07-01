# Tablet Energy Optimizer

A machine-learning system for optimizing energy consumption in pharmaceutical tablet manufacturing. It combines predictive modeling, multi-objective optimization, a Golden Signature Framework, and a real-time Streamlit dashboard to help manufacturers reduce energy use while maintaining product quality and regulatory compliance.

## Features

- **Golden Signature Framework** — Define Pareto-optimal process parameter sets for any combination of targets (energy, quality, yield, carbon, throughput)
- **Multi-objective Optimization** — Latin Hypercube sampling + Pareto frontier analysis to balance competing objectives
- **Continuous Learning Engine** — Automatically updates Golden Signatures when production consistently outperforms current benchmarks
- **Adaptive Optimizer** — Real-time parameter recommendations that adapt to incoming sensor data
- **Streamlit Dashboard** — Interactive visualization of optimization results, per-batch performance, learning trends, and regulatory compliance
- **FastAPI Service** — REST endpoints for energy prediction and Golden Signature recommendations
- **Regulatory Compliance Tracking** — Enforces FDA/EMA thresholds (≥85% dissolution rate, ≤120 kWh/batch) and sustainability targets

## Project Structure

```
tablet-energy-optimizer/
├── main.py                     # Data pipeline entry point
├── requirements.txt
├── data/
│   └── production_data.csv     # Raw production data
├── outputs/                    # Generated artifacts
│   ├── processed_dataset.csv
│   ├── optimization_results.csv
│   ├── pareto_frontier.csv
│   ├── golden_benchmarks.json
│   ├── adaptive_config.json
│   ├── learning_history.json
│   └── model_metrics.json
├── models/                     # Trained ML models (joblib)
├── src/
│   ├── data_pipeline.py        # Feature engineering & dataset assembly
│   ├── train_models.py         # XGBoost model training
│   ├── predictor.py            # Inference wrapper
│   ├── optimization.py         # Energy optimization (Latin Hypercube search)
│   ├── golden_signature.py     # Golden Signature Framework & Pareto optimizer
│   ├── adaptive_optimizer.py   # Real-time adaptive optimization
│   ├── continuous_learning.py  # Self-improving benchmark system
│   ├── realtime_stream.py      # Sensor data simulation & streaming
│   └── api.py                  # FastAPI REST service
└── dashboard/
    └── app.py                  # Streamlit dashboard
```

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd tablet-energy-optimizer

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the Dashboard (recommended starting point)

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Requires pre-generated outputs in the `outputs/` directory (included in the repo).

### 2. Run the Data Pipeline

Processes raw Excel production data into the feature dataset used for model training.

```bash
python main.py
```

> **Note:** Expects `data/production_data.xlsx` and `data/batch_data.xlsx`. The included `production_data.csv` is the processed output.

### 3. Start the REST API

```bash
uvicorn src.api:app --reload
```

API docs available at `http://localhost:8000/docs`.

## Dashboard Sections

| Section | Description |
|---|---|
| **Golden Signature Profiles** | Select optimization profiles (Energy Optimizer, TriBalance, Green Impact, YieldMax, Quality Leader) and view recommended parameters |
| **Pareto Frontier** | Interactive scatter of Pareto-optimal solutions trading off energy vs. quality |
| **Per-Batch Performance** | Historical batch energy vs. target with anomaly flagging |
| **Continuous Learning Trends** | Track benchmark improvement over time |
| **Live Sensor Simulation** | Real-time adaptive parameter recommendations |
| **Regulatory Compliance** | Audit trail of compliance status against FDA/EMA and sustainability limits |

## Optimization Profiles

| Profile | Quality Weight | Energy Weight | Yield Weight |
|---|---|---|---|
| Energy Optimizer | 0.20 | 0.70 | 0.10 |
| TriBalance | 0.40 | 0.40 | 0.20 |
| Green Impact Mode | 0.25 | 0.50 | 0.25 |
| YieldMax | 0.20 | 0.20 | 0.60 |
| Quality Leader | 0.70 | 0.15 | 0.15 |

## Regulatory Limits

| Metric | Limit |
|---|---|
| Max energy per batch | 120 kWh |
| Min dissolution rate | 85% |
| Max carbon per batch | 90 kg CO₂ |
| Max annual carbon | 150 t CO₂ |
| Energy reduction target | 15% YoY |

