import numpy as np

def evaluate_model_confidence(name, model, x_test, y_test):
    """
    Checks if higher confidence = higher precision.
    """
    print(f"\n🔍 DEEP DIVE: PROBABILITY ANALYSIS FOR {name.upper()}")
    probs = model.predict_proba(x_test)[:, 1]

    thresholds = [0.50, 0.53, 0.55, 0.57, 0.60, 0.65]
    metrics    = []
    for t in thresholds:

        high_conf_preds = (probs > t).astype(int)

        if np.sum(high_conf_preds) > 0:

            mask           = probs > t
            filtered_y     = y_test[mask]
            filtered_preds = high_conf_preds[mask]


            win_rate = np.mean(filtered_y)
            metrics.append([t, win_rate])
            print(f"Threshold > {t:.2f} | Trades: {len(filtered_y)} | ✅ Precision: {win_rate:.4f}")
        else:
            print(f"Threshold > {t:.2f} | Trades: 0")
    return metrics