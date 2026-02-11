from joblib import load, dump
from sklearn.ensemble import VotingClassifier
from pathlib import Path
from src.data import data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.evaluate_model_confidence import evaluate_model_confidence

# -- Path Setup
PARENT_DIR = Path(__file__).parent.parent
MODELS_DIR = PARENT_DIR / "models"

def get_random_classifier_model():
    print("Loading Random forest classifier model..")
    file_path = MODELS_DIR / "random_forest_model.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"Random forest classifier was not found in {MODELS_DIR}")
    model = load(file_path)
    print("Random forest classifier model loaded.")
    return model

def get_xgboost_classifier_model():
    print("Loading XG Boost classifier model..")
    file_path = MODELS_DIR / "xgb_classifier_model.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"XG-Boost classifier was not found in {MODELS_DIR}")
    model = load(file_path)
    print("XG Boost classifier model loaded.")
    return model

def train_ensemble_model():
    # Loading the models
    rfc_model = get_random_classifier_model()
    xgb_model = get_xgboost_classifier_model()

    # Gathering the train and test data
    train_df = data.training_data
    test_df  = data.test_data

    x_train = train_df.drop(columns=['Target'])
    y_train = train_df['Target']

    x_test = test_df.drop(columns=['Target'])
    y_test = test_df['Target']

    print("\n Training Voting classifier model..")
    voting_classifier = VotingClassifier(
        estimators=[("rfc", rfc_model), ("xgb", xgb_model)],
        n_jobs=-1,
        voting="soft"
    )
    voting_classifier.fit(x_train, y_train)
    dump(voting_classifier, MODELS_DIR/ "voting_classifier.pkl")

    rfc_metrics = evaluate_model_confidence("Random Forest Classifier", rfc_model        , x_test, y_test)
    xgb_metrics = evaluate_model_confidence("XG Boost Classifier"     , xgb_model        , x_test, y_test)
    vc_metrics  = evaluate_model_confidence("Voting Classifier"       , voting_classifier, x_test, y_test)

    df_rfc = pd.DataFrame(rfc_metrics, columns=['Threshold', 'Precision'])
    df_xgb = pd.DataFrame(xgb_metrics, columns=['Threshold', 'Precision'])
    df_vc  = pd.DataFrame(vc_metrics , columns=['Threshold', 'Precision'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_rfc, x='Threshold', y='Precision', label='Random Forest'  , marker='o', linestyle='--')
    sns.lineplot(data=df_xgb, x='Threshold', y='Precision', label='XG Boost'       , marker='s', linestyle='--')
    sns.lineplot(data=df_vc , x='Threshold', y='Precision', label='Voting Ensemble', marker='^', linewidth=3)

    plt.axhline(y=0.60, color='red', linestyle=':', label='60% Profit Zone')

    plt.title("Model Precision vs. Confidence Threshold", fontsize=14)
    plt.ylabel("Precision (Win Rate)", fontsize=12)
    plt.xlabel("Confidence Threshold", fontsize=12)
    plt.legend()
    plt.savefig(PARENT_DIR / "metrics.png")
    plt.show()

def main():
    train_ensemble_model()

if __name__ == "__main__":
    main()