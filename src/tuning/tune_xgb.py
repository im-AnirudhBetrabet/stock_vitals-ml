from typing import List, Dict, Any
from xgboost import XGBClassifier
# from src.data import data
import pandas as pd
from src.evaluate_model_confidence import evaluate_core_metrics
import seaborn as sns
import matplotlib.pyplot as plt


def get_param_grid() -> List[Dict[str, Any]]:
    """
    Returns a grid of XGBoost parameters.
    """
    print(">> Building Hyperparameter Grid...")
    max_depth_options     = [3, 4, 5]
    learning_rate_options = [0.03, 0.05, 0.1]
    n_estimator_options   = [150, 250, 350]
    subsample_options     = [0.8, 1.0]
    colsample_options     = [0.8, 1.0]

    param_grid = []
    for depth in max_depth_options:
        for lr in learning_rate_options:
            for n_est in n_estimator_options:
                for subsample in subsample_options:
                    for colsample in colsample_options:
                        param_grid.append({
                            "max_depth"       : depth,
                            "learning_rate"   : lr,
                            "n_estimators"    : n_est,
                            "subsample"       : subsample,
                            "colsample_bytree": colsample
                        })
    print(">> Hyperparameter Grid built.")
    return param_grid

def build_xgb_model(params: Dict[str, Any], scale_val: float):
    max_depth, learning_rate, n_estimators, subsample, colsample_bytree = params.values()
    return XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_val,
        random_state=42,
        eval_metric="logloss"
    )

def evaluate_config(stock_data: pd.DataFrame, params: Dict[str, Any], metrics_collector: List) -> Dict[str, float]:
    overall_metrics = []
    for test_year in range(2019, 2025):
        print(f"Running test for year {test_year}")
        # 1. split training and test data based on test year
        train_data = stock_data[stock_data.index.year  < test_year]
        test_data  = stock_data[stock_data.index.year == test_year]
        train_data = train_data.copy()
        test_data  = test_data.copy()

        # 2. Handling edge case when the test data might not be available
        if len(test_data) == 0:
            continue

        # 3. Calculate the median 20-Day volatility of Nifty 50 from the training data.
        train_training_index_median_vol = train_data['NSEI_20d_vol'].dropna().median()
        # 3.1 Determine the volatility regime for the training data.
        train_data['Vol_Regime'] = (train_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
        # 3.2 Determine the combined regime for the training data
        train_data['Combined_Regime'] = train_data['Trend_Regime'] * 2 + train_data['Vol_Regime']

        # 4. Determine the volatility and combined regimes for the test data
        test_data['Vol_Regime'] = (test_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
        test_data['Combined_Regime'] = test_data['Trend_Regime'] * 2 + test_data['Vol_Regime']

        # 5. Dropping un-necessary features
        x_train = train_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
        y_train = train_data['Target']

        x_test = test_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
        y_test = test_data['Target']

        # 6. Calculating class imbalance
        negative_cases = (y_train == 0).sum()
        positive_cases = (y_train == 1).sum()

        if positive_cases == 0:
            continue
        scale_val = negative_cases / positive_cases

        #7. Building the model with the current params
        xgb = build_xgb_model(params, scale_val)
        xgb.fit(x_train, y_train)

        # 8. Determining the core metrics at 0.6 threshold
        metrics = evaluate_core_metrics(name="XG Boost Classifier", model=xgb, x_test=x_test, y_test=y_test)
        metrics.update(params)
        metrics['Test_Year'] = test_year
        metrics_collector.append(metrics)
        overall_metrics.append(metrics)

    metrics_df      = pd.DataFrame(overall_metrics)
    summary_metrics = {
        "mean_precision": metrics_df['precision'].mean(),
        "mean_recall"   : metrics_df['recall'].mean(),
        "mean_roc_auc"  : metrics_df['roc_auc'].mean(),
        "mean_f1_score" : metrics_df['f1_score'].mean(),
        "mean_coverage" : metrics_df['coverage'].mean(),
        "std_precision" : metrics_df['precision'].std(),
        "std_recall"    : metrics_df['recall'].std(),
        "std_roc_auc"   : metrics_df['roc_auc'].std(),
        "std_f1_score"  : metrics_df['f1_score'].std(),
        "std_coverage"  : metrics_df['coverage'].std(),
    }
    summary_metrics.update(params)
    return summary_metrics


if __name__ == "__main__":
    # param_list        = get_param_grid()
    # all_stock_data    = data.all_data
    # metrics_collector = []
    # summary_collector = []
    # for param in param_list:
    #     summary = evaluate_config(all_stock_data, param, metrics_collector)
    #     summary_collector.append(summary)
    #
    # pd.DataFrame(metrics_collector).to_csv("xgb_tuning_metrics.csv")
    # pd.DataFrame(summary_collector).to_csv("xbg_training_summary_metrics.csv")

    summary_df = pd.read_csv('xbg_training_summary_metrics.csv')


    filtered_df = summary_df[(summary_df['mean_roc_auc'] >= 0.548) &
                             (summary_df['std_roc_auc'] <= 0.03)].copy()

    print(f"Number of models meeting the criteria: {len(filtered_df)}")

    # Set seaborn theme
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: Highlighting the Sweet Spot ---
    plt.figure(figsize=(10, 6))

    # Plot all points in light grey
    sns.scatterplot(
        data=summary_df,
        x='mean_roc_auc',
        y='std_roc_auc',
        color='lightgrey',
        alpha=0.6,
        label='All Models'
    )

    # Plot the filtered points in bright blue
    sns.scatterplot(
        data=filtered_df,
        x='mean_roc_auc',
        y='std_roc_auc',
        color='blue',
        s=100,
        edgecolor='black',
        label='Filtered Models (AUC >= 0.54 & Std <= 0.03)'
    )

    # Add reference lines for your thresholds
    plt.axvline(x=0.54, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=0.02, color='red', linestyle='--', alpha=0.5)

    plt.title('Mean ROC AUC vs. Standard Deviation of ROC AUC', fontsize=14)
    plt.xlabel('Mean ROC AUC (Higher is better)')
    plt.ylabel('Std Dev of ROC AUC (Lower is more stable)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('auc_sweet_spot.png')
    plt.show()

    # --- PLOT 2: Hyperparameters of the Filtered Models ---
    if len(filtered_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Learning Rate vs Max Depth for the filtered models
        sns.scatterplot(
            data=filtered_df,
            x='learning_rate',
            y='max_depth',
            hue='n_estimators',
            size='mean_roc_auc',
            sizes=(100, 300),
            palette='viridis',
            ax=axes[0]
        )
        axes[0].set_title('Hyperparameters of Top Stable Models')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Max Depth')

        # Plot the distribution of n_estimators for the filtered models
        sns.countplot(
            data=filtered_df,
            x='n_estimators',
            palette='Set2',
            ax=axes[1]
        )
        axes[1].set_title('Count of n_estimators in Top Models')

        plt.tight_layout()
        plt.savefig('top_models_hyperparameters.png')
        plt.show()
    else:
        print("No models met the specified criteria to generate the hyperparameter plot.")

