# V1 Model Performance Report: Limitations & Drawbacks
## Current Architecture:
* Input Features: Single-stock technical indicators (RSI, SMA, BB, etc.) only.
* Missing Context: No general market data (Nifty 50 trend, Sector performance).
* Best Model: Voting Classifier.

![V1 Model Performance](v1-metrics.png)

## The "Coin Flip" Baseline (Low Signal-to-Noise Ratio)
The most glaring issue is the baseline performance. At the standard classification threshold of 0.50, none of the models can reliably distinguish a winning trade from a losing one.

### Evidence:

* Random Forest Precision: 50.24%

* XG-Boost Precision: 50.32%

* Voting Precision: 50.36%

### Implication: 
The current feature set (Technical Indicators alone) contains too much "noise." The model detects a "bullish pattern" (e.g., RSI > 60), but without knowing if the market is crashing, that pattern is essentially random (50/50).

## Extreme "Opportunity Cost" (Recall Collapse)
To reach a profitable precision (e.g., >60%), the model must abandon 98.7% of all trading opportunities.

### Evidence:

* Voting Classifier: To improve precision from 50% (at 0.50 threshold) to 63.75% (at 0.60 threshold), the trade volume shrinks from 12,956 trades to only 160 trades.


### Implication: 
The model is "afraid." It can only find refuge in extreme outliers. A good model should be able to find trading opportunities in the 0.53-0.57 probability range, but the current model cannot (Precision remains ~50-53%). We are leaving money on the table because the model is not equipped with enough context to "trust" signals.

## Poor Probability Calibration (The "0.65" Drop)
Higher confidence should always translate to higher accuracy, but the data reveals a discontinuity at the ends.

### Evidence:
* Voting Classifier: Precision reaches a maximum of 63.75% (at threshold 0.60) but then falls to 59.18% at the higher threshold of 0.65.

### Implication: 
This suggests overfitting to particular noise. The trading opportunities the current model is "most confident" in (0.65+) are actually less likely to be true than those at 0.60. This is a very bad sign that the model is mistaking noise for signal (e.g., "Always buy if RSI is exactly 71.2").

## Single-Stock Bias (Systematic Risk Ignorance)
The model looks at all stocks in isolation. It believes that if Tata Motors has a "Golden Cross," it will necessarily go up, even if the Nifty 50 is down 2%.


The Flaw: Technical Analysis is most effective when done in tandem with the market trend ("A rising tide lifts all boats"). The reason the model achieves only 50% precision is that it swims against the tide 50% of the time by ignoring the data in the Index.
