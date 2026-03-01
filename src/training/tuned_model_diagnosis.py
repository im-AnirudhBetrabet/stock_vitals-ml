from src.data import data
from joblib import load
from config.Config import config
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.evaluate_model_confidence import evaluate_core_metrics_at_varying_thresholds

if __name__ == "__main__":
    historical_data = data.all_data.copy()

    train_validation_set = historical_data[(historical_data.index.year >= 2019) & (historical_data.index.year <= 2024)]
    test_validation_set  = historical_data[historical_data.index.year == 2025]

    # Class imbalance analysis
    total_train_data = len(train_validation_set)
    total_test_data  = len(test_validation_set)

    train_pos_cases  = (train_validation_set['Target'] == 1).sum()
    train_neg_cases  = (train_validation_set['Target'] == 0).sum()

    test_pos_cases = (test_validation_set['Target'] == 1).sum()
    test_neg_cases = (test_validation_set['Target'] == 0).sum()

    train_pos_ratio = train_pos_cases / total_train_data
    train_neg_ratio = train_neg_cases / total_train_data

    test_pos_ratio = test_pos_cases / total_test_data
    test_neg_ratio = test_neg_cases / total_test_data

    print("==== 2019–2024 ====")
    print(f"Total samples: {total_train_data}")
    print(f"Bullish days: {train_pos_cases}")
    print(f"Bearish days: {train_neg_cases}")
    print(f"Positive ratio: {train_pos_ratio:.4f}")
    print()

    print("==== 2025 ====")
    print(f"Total samples: {total_test_data}")
    print(f"Bullish days: {test_pos_cases}")
    print(f"Bearish days: {test_neg_cases}")
    print(f"Positive ratio: {test_pos_ratio:.4f}")

    # Probability distribution analysis
    train_training_index_median_vol  = historical_data['NSEI_20d_vol'].dropna().median()
    # 3.1 Determine the volatility regime for the training data.
    historical_data['Vol_Regime']      = (historical_data['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
    # 3.2 Determine the combined regime for the training data
    historical_data['Combined_Regime'] = historical_data['Trend_Regime'] * 2 + historical_data['Vol_Regime']

    # 4. Determine the volatility and combined regimes for the test data
    test_validation_set['Vol_Regime']      = (test_validation_set['NSEI_20d_vol'] > train_training_index_median_vol).astype(int)
    test_validation_set['Combined_Regime'] = test_validation_set['Trend_Regime'] * 2 + test_validation_set['Vol_Regime']

    # 5. Dropping un-necessary features
    x_train = historical_data.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
    y_train = historical_data['Target']

    x_test = test_validation_set.drop(columns=['Target', 'Vol_Regime', 'Trend_Regime', 'NSEI_200SMA'])
    y_test = test_validation_set['Target']
    PARENT_DIR   = Path(__file__).parent.parent.parent
    model_dir    = PARENT_DIR / 'models' / config.current_model_version / "xgb_classifier_model_v3.pkl"
    reports_path = PARENT_DIR / config['data']['reports_path'] / config.current_model_version

    xgb        = load(model_dir)

    historical_data_probability  = xgb.predict_proba(x_train)[:, 1]
    test_data_probability        = xgb.predict_proba(x_test)[:, 1]

    plt.figure(figsize=(10, 6))

    plt.hist(historical_data_probability, bins=50, alpha=0.6, label="2019–2024")
    plt.hist(test_data_probability, bins=50, alpha=0.6, label="2025")

    plt.axvline(0.6, color="red", linestyle="--", label="Threshold 0.6")

    plt.title("Predicted Probability Distribution Shift")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(reports_path / "v3_model_probability_distribution.png")
    plt.show()

    regime_hist_counts = x_train['Combined_Regime'].value_counts(normalize=True)
    regime_test_counts = x_test['Combined_Regime'].value_counts(normalize=True)

    print("2019-2024 Regime Distribution")
    print(regime_hist_counts.sort_index(ascending=True))
    print("\n 2025 Regime Distribution")
    print(regime_test_counts.sort_index(ascending=True))

    df_compare = pd.DataFrame({
        '2019-2024 (Train)': regime_hist_counts,
        '2025 (Test)': regime_test_counts
    })

    # 2. Sort the index to ensure the regimes (0.0, 1.0, 2.0, 3.0) are in order
    df_compare = df_compare.sort_index()

    # 3. Create the grouped bar chart
    ax = df_compare.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'], edgecolor='black', width=0.7)

    # 4. Formatting the plot
    plt.title('Regime Distribution Comparison: 2019-2024 vs 2025', fontsize=14, pad=15)
    plt.ylabel('Proportion', fontsize=12)
    plt.xlabel('Combined Regime', fontsize=12)
    plt.xticks(rotation=0)  # Keeps the x-axis labels horizontal
    plt.legend(title='Time Period', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Adds horizontal gridlines for easier reading

    # Optional: Add the exact proportion numbers on top of each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 6),
                    textcoords='offset points',
                    fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.savefig(reports_path / "v3_model_regime_distribution.png")
    plt.show()

    metrics_at_varying_threshold = evaluate_core_metrics_at_varying_thresholds("XGBoost Classifier", model=xgb, x_test=x_test, y_test=y_test)
    threshold_metrics_df = pd.DataFrame(metrics_at_varying_threshold)

    threshold_metrics_df.to_csv(reports_path / f'{config.current_model_version}_performance_metrics.csv')

    models = threshold_metrics_df['Model'].unique()
    ticks = ["^", "o", "s"]
    for metric in threshold_metrics_df.columns[2:]:
        fig = plt.figure(figsize=(6, 6))
        for idx, model in enumerate(models):
            temp_df = threshold_metrics_df[threshold_metrics_df["Model"] == model][["Model", "Threshold", metric]]
            sns.lineplot(data=temp_df, x='Threshold', y=metric, label=model, marker=ticks[idx])
            plt.xlabel("Threshold")
            plt.ylabel(metric)
            plt.legend()
        plt.savefig(reports_path / f"{config.current_model_version}_{metric}.png")