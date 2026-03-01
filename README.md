# Regime-Aware Stock Direction Prediction Framework

## Overview
This project investigates whether stock return direction can be predicted using technical
indicators, market-level signals, and structural regime information under strict temporal
validation.

The objective was not simply to maximize performance, but to design and evaluate
model under realistic non-stationary market conditions.

The framework evolves through three modeling steps:
* **Model 1**: Technical indicators only.
* **Model 2**: Technical + index-level features + horizon experimentation.
* **Model 3**: Regime-aware modeling with structured walk-forward validation and
hyperparameter tuning.

All experiments were conducted using expanding-window walk-forward validation to
preserve temporal integrity.

## Methodology

1. **Feature Engineering**
   * Stock-level technical indicators (SMA distance, MACD, Bollinger Bands, rolling volatility, momentum deltas).
   * Market-level context using Nifty 50 and Nifty Next 50 indicators
   * Structural regime encoding via:
     * Trend regime (200-day SMA state).
     * Volatility regime (relative to training median).
     * Interaction term (`Combined_Regime`).
   Intermediate regime features were excluded to prevent feature inflation.
   
2. **Validation Framework**
   * Expanding-window walk-forward validation (2019-2024).
   * Strict separation of:
     * Architecture selection
     * Hyperparameter tuning
     * Final holdout evaluation (2025).
   No leakage from future data.

3. **Model Comparison**
    <br/>
   Evaluated:
   * Random Forest Classifier.
   * XGBoost Classifier.
   * Ensemble Voting Classifier.
   
   Selection based on:
   * Mean ROC-AUC across folds.
   * Cross-year stability (standard deviation).
   * Precision-coverage tradeoff.
   
   XGBoost demonstrated the strongest ranking consistency.

## Key Results

**Walk-Forward (2019 - 2024)**
* Mean ROC-AUC ≈ 0.54-0.55
* Std ≈ 0.02-0.03
* Precision @ 0.6 ≈ 0.60

Indicates moderate but persistent ranking ability.

**Final Holdout (2025)**
* ROC-AUC ≈ 0.507
* Precision @ 0.6 ≈ 0.55
* Coverage ≈ 13%

| Phase                    | ROC-AUC   | Notes                                      |
|--------------------------|-----------|--------------------------------------------|
| Walk-Forward (2019–2024) | 0.54–0.55 | Stable cross-year ranking                  |
| Holdout (2025)           | 0.507     | Performance degradation under regime shift |

Performance degraded under distributional shift in 2025.

## Diagnostic Findings

Analysis revealed:
* Class distribution shift between training and 2025.
* Structural redistribution of regime states.
* Compression of predicted probability distribution.
* Threshold tuning unable to recover separation.

Conclusion:
```
The predictive signal is regime-conditional and sensitive to non-stationarity.
```

## Technical Highlights
* Expanding-window validation.
* Fold-level stability analysis.
* Hyperparameter grid search (108 configurations).
* Regime interaction modeling.
* Distribution shift diagnostics.
* Threshold sensitivity analysis.

## Takeaway
This project demonstrates disciplined experimental design, temporal validation rigor,
structured model selection, and diagnostic analysis under real-world market non-stationarity.