"""Fast targeted run for one indicator series."""

from time_series_analysis_pipeline_v7 import (
    INPUT_DIR,
    OUTPUT_DIR,
    PipelineConfig,
    ResultsExporter,
    TimeSeriesAnalyzer,
    detect_date_column,
    detect_indicator_columns,
    load_excel_data,
)


TARGET_SERIES = "Depots Clientele_comptes chèques"


def main() -> None:
    df = load_excel_data(INPUT_DIR)
    date_col = detect_date_column(df)
    indicator_cols = detect_indicator_columns(df, date_col)

    if TARGET_SERIES not in indicator_cols:
        matches = [col for col in indicator_cols if "comptes chèques" in col.lower()]
        raise ValueError(
            f"Target series not found: {TARGET_SERIES}. "
            f"Close matches: {matches}"
        )

    config = PipelineConfig(
        horizon=30,
        holdout_min=30,
        holdout_max=60,
        max_lag=90,
        max_selected_lags=18,
        use_xgboost=True,
    )
    analyzer = TimeSeriesAnalyzer(
        df,
        date_col=date_col,
        indicator_cols=[TARGET_SERIES],
        config=config,
    )
    analyzer.analyze_all()

    exporter = ResultsExporter(analyzer, OUTPUT_DIR)
    exporter.create_summary_report()
    exporter.create_visualization_dashboard()
    exporter.create_individual_charts()

    print("Target series:", TARGET_SERIES)
    print("Model:", analyzer.models.get(TARGET_SERIES, {}).get("name"))
    print("Metrics:", analyzer.models.get(TARGET_SERIES, {}).get("metrics"))
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
