import os

import pandas as pd
import numpy as np
from sklearn.metrics import (
  confusion_matrix,
  classification_report,
  roc_auc_score,
  recall_score,
  f1_score,
  precision_score
)

from config.Config import config


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


            win_rate = np.mean(filtered_preds == filtered_y)
            metrics.append([name, t, len(filtered_y), win_rate, len(filtered_y) / len(x_test)])
            print(f"Threshold > {t:.2f} | Trades: {len(filtered_y)} | ✅ Precision: {win_rate:.4f}")
        else:
            print(f"Threshold > {t:.2f} | Trades: 0")
    return metrics

def evaluate_core_metrics(name, model, x_test, y_test):

    y_pred        = model.predict(x_test)
    y_prob        = model.predict_proba(x_test)[:, 1]
    y_pred_custom = (y_prob > 0.6)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_custom, labels=[0, 1])

    true_negative, false_positive, false_negative, true_positive = cm.ravel()

    precision = true_positive / ( true_positive + false_positive )  # Of all the models positive predictions, how many are actually positive.
    recall    = true_positive / ( true_positive + false_negative )  # Of all positive outcomes, what proportion are correctly classifed as positive.
    f1        = 2 * ((precision * recall) / (precision + recall))
    roc_auc   = roc_auc_score(y_test, y_prob)

    return {
        "model_name": name,
        "precision" : precision,
        "recall"    : recall,
        "f1_score"  : f1,
        "roc_auc"   : roc_auc
    }

if __name__ == "__main__":
    from joblib import load
    from pathlib import Path
    from src.data import data
    PARENT_DIR = Path(__file__).parent.parent
    test_data = data.test_data
    x_test  = test_data.drop(columns=['Target'])
    y_test  = test_data['Target']
    results = []
    threshold_metrics    = []
    threshold_metrics_df = pd.DataFrame()
    reports_path         = PARENT_DIR / config['data']['reports_path'] / config.current_model_version
    os.makedirs(reports_path, exist_ok=True)
    for model_name in [f"random_forest_model_{config.current_model_version}.pkl", f"xgb_classifier_model_{config.current_model_version}.pkl", f"voting_classifier_{config.current_model_version}.pkl"]:
        PARENT_DIR = Path(__file__).parent.parent
        MODELS_DIR = PARENT_DIR / "models" / config.current_model_version / model_name
        model      = load(MODELS_DIR)
        results.append(evaluate_core_metrics(model_name.split(".")[0], model, x_test, y_test))
        threshold_metrics_df = pd.concat([threshold_metrics_df, pd.DataFrame(evaluate_model_confidence(model_name.split(".")[0], model, x_test, y_test), columns=['Model', 'Threshold', 'Number of Trades', 'Precision', 'Coverage'])])

    # threshold_metrics_df.set_index('Model', inplace=True)

    print(pd.DataFrame(results).set_index('model_name'))
    print(threshold_metrics_df)

    report_name = reports_path / f'{config.current_model_version}_performance_metrics.csv'
    threshold_metrics_df.pivot(index='Threshold', columns='Model',  values=['Precision', 'Coverage']).to_csv(report_name)