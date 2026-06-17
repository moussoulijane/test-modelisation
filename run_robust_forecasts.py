from pathlib import Path

from robust_forecast_engine import ForecastConfig, run_bundle


config = ForecastConfig(
    input_path=Path("in/reserve_in.xlsx"),
    output_dir=Path("out"),
    holdout_len=60,
    forecast_horizon=90,
    wf_folds=5,
    wf_horizon=30,
    min_gain_vs_baseline_pct=5.0,
)

run_bundle(
    name="depots_cheques_courants",
    targets=[
        "Depots Clientele_comptes chèques",
        "Depots Clientele_comptes courants",
    ],
    config=config,
    add_total=True,
)

run_bundle(
    name="credit_equipement",
    targets=["Credit Décaissement_crédits à lequipement"],
    config=config,
)
