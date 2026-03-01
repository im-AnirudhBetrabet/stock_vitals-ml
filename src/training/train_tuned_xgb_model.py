import os

from xgboost import XGBClassifier
from src.data import data
from src.evaluate_model_confidence import evaluate_core_metrics
from joblib import dump
from config.Config import config
from pathlib import Path

from src.validators.walk_forward_validator import PARENT_DIR


def train_xgb_model():
    # 1. Get training and testing data
    training_data = data.training_data.copy()
    testing_data  = data.test_data.copy()

    # 2. Calculating volatility
    train_training_index_median_vol  = training_data['NSEI_20d_vol'].dropna().median()
    # 3.1 Determine the volatility regime for the training data.
    training_data['Vol_Regime']      = (training_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
    # 3.2 Determine the combined regime for the training data
    training_data['Combined_Regime'] = training_data['Trend_Regime'] * 2 + training_data['Vol_Regime']

    # 4. Determine the volatility and combined regimes for the test data
    testing_data['Vol_Regime']      = (testing_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
    testing_data['Combined_Regime'] = testing_data['Trend_Regime'] * 2 + testing_data['Vol_Regime']

    # 5. Dropping un-necessary features
    x_train = training_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
    y_train = training_data['Target']

    x_test = testing_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
    y_test = testing_data['Target']

    # 6. Calculating class imbalance
    negative_cases = (y_train == 0).sum()
    positive_cases = (y_train == 1).sum()

    if positive_cases == 0:
        raise ValueError("No positive cases found in training data.")

    scale_val = negative_cases / positive_cases

    xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        subsample=1.0,
        colsample_bytree=0.8,
        scale_pos_weight=scale_val,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb.fit(x_train, y_train)

    metrics = evaluate_core_metrics(name="XGBoost Classifier", model=xgb, x_test=x_test, y_test=y_test, )
    print(metrics)

    model_dir = Path(__file__).parent.parent.parent / 'models' / config.current_model_version
    os.makedirs(model_dir, exist_ok=True)
    dump(xgb, model_dir / f"xgb_classifier_model_{config.current_model_version}.pkl")


if __name__ == "__main__":
    train_xgb_model()
