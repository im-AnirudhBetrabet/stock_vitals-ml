from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from pathlib import Path
from config.Config import config
from src.data import data
from src.evaluate_model_confidence import evaluate_model_confidence
import joblib

PARENT_DIR         = Path(__file__).parent.parent
PROCESSED_DATA_DIR = PARENT_DIR / config['data']['processed_path']
MODELS_DIR         = PARENT_DIR / "models"


def train_random_forest_model():
    print("Starting training...")

    train_df = data.training_data
    test_df  = data.test_data

    # Separating the feature and targets
    x_train = train_df.drop(columns=['Target'])
    y_train = train_df['Target']

    x_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']

    print(f"Training Random Forest Classifier model on {len(x_train)} rows, Testing on {len(x_test)} rows.")

    # Train
    random_forest_model = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42, max_features='sqrt', class_weight='balanced_subsample')
    random_forest_model.fit(x_train, y_train)

    # Test
    preds     = random_forest_model.predict(x_test)
    precision = precision_score(y_test, preds)
    evaluate_model_confidence("Random Forest Classifier", random_forest_model, x_test, y_test)
    print("\n" + "=" * 30)
    print(f"🎯Random Forest Classifier MODEL PRECISION: {precision:.4f}")
    print("=" * 30)

    joblib.dump(random_forest_model, MODELS_DIR / "random_forest_model.pkl")

def train_xgboost_model():
    print("Starting training...")

    train_df = data.training_data
    test_df  = data.test_data

    # Separating the feature and targets
    x_train = train_df.drop(columns=['Target'])
    y_train = train_df['Target']

    x_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']

    negative_cases = (y_train == 0).sum()
    positive_cases = (y_train == 1).sum()

    scale_val = negative_cases / positive_cases

    print(f"Negative: {negative_cases}, Positive: {positive_cases}")
    print(f"Suggested scale_pos_weight: {scale_val}")
    print(f"Training XGBoost classifier model on {len(x_train)} rows, Testing on {len(x_test)} rows.")

    xgb_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42, eval_metric='logloss', scale_pos_weight=scale_val)
    print("\n🚀 Training XGBoost...")
    xgb_classifier.fit(x_train, y_train)
    preds = xgb_classifier.predict(x_test)
    precision = precision_score(y_test, preds)
    evaluate_model_confidence("XG Boost Classifier", xgb_classifier, x_test, y_test)
    print("\n" + "=" * 30)
    print(f"🎯 XG Boost MODEL PRECISION: {precision:.4f}")
    print("=" * 30)

    joblib.dump(xgb_classifier, MODELS_DIR / "xgb_classifier_model.pkl")


def main():
    train_random_forest_model()
    train_xgboost_model()


if __name__ == "__main__":
    main()

