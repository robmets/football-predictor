# ⚽ Football Match Predictor

A data science project for predicting football match outcomes using historical data,
statistical modelling, machine learning, and Monte Carlo simulation.

## Supported Leagues
- Bundesliga (BL1)
- Premier League (PL)
- Champions League (CL)
- La Liga, Serie A, Ligue 1

## Architecture
```
Data Collection → Feature Engineering → Modelling → Simulation → Output
```

## Setup

```bash
# 1. Clone & enter
git clone https://github.com/YOUR_USERNAME/football-predictor.git
cd football-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp config/.env.example config/.env
# → Add your API keys in config/.env

# 5. Initialize the database
python scripts/init_db.py

# 6. Fetch first data
python scripts/fetch_data.py --league BL1 --seasons 5
```

## Branch Strategy
| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code only |
| `develop` | Integration branch for features |
| `feature/data-collector` | Data collection modules |
| `feature/feature-engineering` | Feature extraction |
| `feature/poisson-model` | Poisson goal model |
| `feature/ml-model` | XGBoost / Random Forest |
| `feature/monte-carlo` | Simulation engine |
| `feature/dashboard` | Streamlit UI |

## Project Structure
```
football-predictor/
├── config/             # Config files & API keys (never commit .env!)
├── data/
│   ├── raw/            # Raw API responses (never commit)
│   ├── processed/      # Cleaned, feature-engineered data
│   └── external/       # Weather, odds, external sources
├── src/
│   ├── collectors/     # Data fetching modules
│   ├── features/       # Feature engineering
│   ├── models/         # Statistical & ML models
│   ├── simulation/     # Monte Carlo engine
│   └── utils/          # Shared helpers (db, logging, config)
├── dashboard/          # Streamlit app
├── notebooks/          # Exploratory analysis (Jupyter)
├── tests/              # Unit & integration tests
└── scripts/            # CLI entry points
```

## Usage

```bash
# Predict a specific match
python scripts/predict.py --home "Bayern München" --away "Borussia Dortmund" --league BL1

# Run full pipeline
python scripts/pipeline.py --league BL1 --matchday next

# Launch dashboard
streamlit run dashboard/app.py
```

## Tech Stack
- **Data**: `pandas`, `numpy`, `requests`, `soccerdata`
- **ML**: `scikit-learn`, `xgboost`
- **Stats**: `scipy`
- **DB**: `SQLite` (dev) / `PostgreSQL` (prod)
- **UI**: `streamlit`
- **Testing**: `pytest`
