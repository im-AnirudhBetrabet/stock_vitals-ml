# from src.data import data
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from typing import Dict
from src.evaluate_model_confidence import evaluate_core_metrics
import logging
import pandas as pd
from config.Config import config
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__file__)
PARENT_DIR = Path(__file__).parent.parent.parent
REPORT_DIR = PARENT_DIR / config['data']['reports_path'] / "walk_forward_validation" / config.current_model_version

def build_random_forest():
    """
    Creates an instance of the Random Forest classifier.
    """
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=10, n_jobs=-1, random_state=42, max_features='sqrt', class_weight='balanced_subsample')

def build_xg_boost(scale_val: float):
    """
    Creates an instance of the XG Boost classifier.
    """
    return XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, n_jobs=-1, random_state=42, eval_metric='logloss', scale_pos_weight=scale_val)


def build_voting_classifier(rfc_model, xgb_model):
    """
    Creates an instance of the Ensemble voting classifier
    """
    return VotingClassifier(
        estimators=[("rfc", rfc_model), ("xgb", xgb_model)],
        n_jobs=-1,
        voting="soft"
    )


def walk_forward_validation(stock_data, start_year, end_year) -> list[Dict]:
    """
    Performs a walk-forward validation of the various models.
    """
    results: list = []
    logger.info("*" * 30)

    for test_year in range(start_year, end_year + 1):

        logger.info(f">> Evaluating for training year {start_year} - {end_year} & test year {test_year}")
        train_data = stock_data[stock_data.index.year <  test_year]
        test_data  = stock_data[stock_data.index.year == test_year]

        train_data = train_data.copy()
        test_data  = test_data.copy()

        if len(test_data) == 0:
            continue


        train_training_index_median_vol = train_data['NSEI_20d_vol'].dropna().median()
        train_data['Vol_Regime']        = (train_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
        train_data['Combined_Regime']   = train_data['Trend_Regime'] * 2 + train_data['Vol_Regime']

        test_data['Vol_Regime']        = (test_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
        test_data['Combined_Regime']   = test_data['Trend_Regime'] * 2 + test_data['Vol_Regime']



        x_train = train_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
        y_train = train_data['Target']

        x_test = test_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
        y_test = test_data['Target']
        print(f"Train Features: {x_train.columns}")
        print(f"Test Features: {x_test.columns}")
        negative_cases = (y_train == 0).sum()
        positive_cases = (y_train == 1).sum()
        if positive_cases == 0:
            continue
        scale_val = negative_cases / positive_cases
        logger.info(">>> Evaluating Random Forest Model")
        rf = build_random_forest()
        rf.fit(x_train, y_train)
        metrics = evaluate_core_metrics(f"Random_forest", rf, x_test, y_test)
        metrics['test_year'] = test_year
        results.append(metrics)

        logger.info(">>> Evaluating XG Boost Model")
        xgb = build_xg_boost(scale_val)
        xgb.fit(x_train, y_train)
        metrics = evaluate_core_metrics(f"XGBoost", xgb, x_test, y_test)
        metrics['test_year'] = test_year
        results.append(metrics)

        logger.info(">>> Evaluating Voting Classifier Model")
        vc = build_voting_classifier(rf, xgb)
        vc.fit(x_train, y_train)
        metrics = evaluate_core_metrics(f"Voting_Classifier", vc, x_test, y_test)
        metrics['test_year'] = test_year
        results.append(metrics)
    logger.info("*" * 30)
    return results

def check_metrics():
    report_path = PARENT_DIR / config['data']['reports_path'] / "walk_forward_validation" / "v2"

    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")


    versions = ['v2', 'v2.1', 'v2.2']
    colors = sns.color_palette("Set1", len(versions))
    markers = ['o', 's', 'D']  # Circle, Square, Diamond

    all_stats = {}
    for version in versions:
        path = report_path / version / f"{version}_walk_forward_validation.csv"
        df = pd.read_csv(path)
        all_stats[version] = df.groupby("model_name")['roc_auc'].agg(['mean', 'std'])

    model_order = all_stats['v2'].sort_values('mean', ascending=False).index
    x_axis = range(len(model_order))

    # 2. ADDING Offset X slightly so error bars don't overlap
    offsets = [-0.1, 0, 0.1]

    for i, version in enumerate(versions):
        stats = all_stats[version].reindex(model_order)

        # Position with offset
        pos = [x + offsets[i] for x in x_axis]

        plt.errorbar(
            x=pos,
            y=stats['mean'],
            yerr=stats['std'],
            label=f"Version {version}",
            fmt=markers[i],
            linestyle='-',
            capsize=6,
            color=colors[i],
            markersize=9,
            linewidth=2,
            elinewidth=2,
            alpha=0.9
        )

    # Re-label X axis correctly after using jitter
    plt.xticks(x_axis, model_order, rotation=25, ha='right')

    plt.title('Model Comparison: ROC-AUC Stability', fontsize=15, fontweight='bold')
    plt.ylabel('Mean ROC-AUC', fontsize=12)
    plt.xlabel('Model Name', fontsize=12)
    plt.legend(title="Validation Version", loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(report_path / "walk_forward_validation.png")



if __name__ == "__main__":
    # start_year = 2018
    # results    = walk_forward_validation(data.all_data, 2019, 2025)
    # results_df = pd.DataFrame(results)
    #
    # os.makedirs(REPORT_DIR, exist_ok=True)
    file_name  = REPORT_DIR / f"{config.current_model_version}_walk_forward_validation.csv"
    # results_df.to_csv(file_name)
    results_df=pd.read_csv(file_name)
    print(results_df.groupby('model_name')['roc_auc'].mean())
    print(results_df.groupby('model_name')['roc_auc'].std())
    models = results_df['model_name'].unique()
    colors = sns.color_palette(n_colors=len(models))
    ticks = ["^", "o", "s"]
    metrics_to_plot = ["roc_auc", "precision", "recall", "f1_score"]
    for metric in metrics_to_plot:
        fig = plt.figure(figsize=(6, 6))
        for idx, model in enumerate(models):
            temp_df = results_df[results_df["model_name"] == model][["model_name", "test_year", metric]]
            model_color = colors[idx]
            sns.lineplot(data=temp_df, x='test_year', y=metric, label=model, marker=ticks[idx], color=model_color)
            plt.axhline(y=temp_df[metric].mean(), label=model, linestyle='--', color=model_color)
        plt.xlabel("Test year")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(REPORT_DIR / f"{config.current_model_version}_{metric}.png")
    check_metrics()