# ENS Challenge 60 - Stock Auction Volume Prediction

## Challenge Overview

Predict the natural logarithm of closing auction volume (as a fraction of total daily volume) for 900 stocks across ~350 test days.

- **Metric**: RMSE (Root Mean Squared Error)
- **Benchmark**: 0.4742 RMSE
- **Data**: ~720K training samples (800 days × 900 stocks)

## Project Structure

```
ENS60/
├── main.ipynb               # Main ML pipeline notebook
├── AGENT.md                 # Agent guidelines and must-haves
├── IDEAS.md                 # Techniques and ideas documentation
├── ENS-Challenge-60-ML-Techniques-Guide.md  # Comprehensive techniques reference
├── configs/
│   └── config.yaml          # Hyperparameters and paths
├── data/
│   ├── input_training.csv.gz    # Training features
│   ├── output_training_*.csv    # Training targets
│   └── input_test.csv.gz        # Test features
├── outputs/
│   └── submission_*.csv     # Submission files
├── tests/
│   ├── test_data_loader_alignment.py
│   └── test_validation_split_integrity.py
└── utils/
    ├── data_loader.py       # Data loading and merging
    ├── preprocessing.py     # NaN handling, scaling
    ├── feature_engineering.py  # Feature creation
    ├── validation.py        # Time series CV
    ├── models.py            # Model definitions
    ├── ensemble.py          # Stacking, blending
    ├── mlflow_utils.py      # Experiment tracking
    └── visualization.py     # Plots
```

## Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Pipeline

Open `main.ipynb` in Jupyter and run all cells.

## Key Features

### Data
- **126 features**: 61 intraday returns, 61 volume fractions, pid, day, LS, NLV
- **NLV**: Most predictive feature (explains ~62% of variance)
- **Target**: log(auction_volume / total_daily_volume)

### Models
1. **NLV Baseline**: Linear regression on NLV only
2. **Two-Stage Model**: Linear + LightGBM on residuals (CFM benchmark approach)
3. **LightGBM**: Full feature model with categorical support
4. **Ensemble**: Stacking with Ridge meta-learner

### Validation
- **TimeSeriesSplit**: Always train on past, validate on future
- **5-fold CV**: Temporal cross-validation

## Performance Targets

| Model | Expected RMSE | R² |
|-------|--------------|-----|
| NLV-only baseline | ~0.29 | ~0.62 |
| Two-stage (Linear + LGB) | ~0.35-0.40 | ~0.70-0.75 |
| LightGBM full features | ~0.35-0.40 | ~0.70-0.75 |
| Ensemble (winning) | ~0.30-0.35 | ~0.80+ |

## Critical Notes

⚠️ **Data Alignment**: Target file IDs start at 1070752, not 0. Merge by position, not ID.

⚠️ **Time Series CV**: Never use random K-Fold - causes temporal leakage.

⚠️ **Categorical Features**: Use `pid` as categorical, not numeric.

See `AGENT.md` for detailed guidelines.

## References

- [Challenge Page](https://challengedata.ens.fr/challenges/60)
- Winner: Franck Zibi (2021) - RNN + LightGBM + CatBoost + Stacking

## License

This project is for educational purposes as part of the ENS Data Challenge.