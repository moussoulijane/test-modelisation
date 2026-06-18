from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.11",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_INTRO = """\
Ce notebook est construit comme outil de prise de decision, pas comme generateur de prediction magique.

Principe:
- on reserve les 60 derniers jours comme hold-out final;
- la selection des modeles se fait avant le hold-out, via walk-forward rolling origin;
- les modeles complexes sont compares a des baselines simples;
- si le gain vs meilleur baseline est < 5%, le baseline est retenu;
- la projection 90 jours inclut un intervalle empirique et un niveau de fiabilite.
"""


IMPORTS = """\
from pathlib import Path

import pandas as pd

from robust_forecast_engine import ForecastConfig, load_workbook, run_bundle
"""


def depots_notebook() -> dict:
    cells = [
        md("# Forecast robuste - Depots comptes cheques et comptes courants\n\n" + COMMON_INTRO),
        md("## Configuration"),
        code(
            IMPORTS
            + """

config = ForecastConfig(
    input_path=Path("in/reserve_in.xlsx"),
    output_dir=Path("out"),
    holdout_len=60,
    forecast_horizon=90,
    wf_folds=5,
    wf_horizon=30,
    min_gain_vs_baseline_pct=5.0,
    peak_quantile=0.80,
    peak_metric_weight=0.45,
    peak_weight_multiplier=4.0,
    max_mape_degradation_vs_baseline_pct=25.0,
)

targets = [
    "Depots Clientele_comptes chèques",
    "Depots Clientele_comptes courants",
]
"""
        ),
        md("## Inspection rapide du fichier source"),
        code(
            """\
df = load_workbook(config)
print(df.shape)
print(df.index.min(), "->", df.index.max())
display(df[targets].describe().T)
"""
        ),
        md("## Execution de la pipeline robuste"),
        code(
            """\
results = run_bundle(
    name="depots_cheques_courants",
    targets=targets,
    config=config,
    add_total=True,
)
"""
        ),
        md("## Synthese decisionnelle"),
        code(
            """\
summary_rows = []
for result in results:
    diag = result.diagnostics.set_index("metric")["value"]
    summary_rows.append({
        "serie": result.target,
        "modele_retenu": diag["final_model"],
        "modele_propose": diag["proposed_model"],
        "modele_scenario_pic": diag["peak_scenario_model"],
        "meilleur_baseline": diag["best_baseline"],
        "meilleur_baseline_mape": diag["best_mape_baseline"],
        "mape_holdout": diag["holdout_MAPE"],
        "peak_wmape_holdout": diag["holdout_peak_wMAPE"],
        "peak_capture": diag["holdout_peak_capture"],
        "score_decision": diag["holdout_score"],
        "selection_gain_vs_baseline_pct": diag["selection_gain_vs_best_baseline_pct"],
        "selection_mape_degradation_pct": diag["selection_mape_degradation_vs_best_baseline_pct"],
        "robustesse": diag["robustness_flag"],
        "recommendation": diag["decision_recommendation"],
    })
summary = pd.DataFrame(summary_rows)
display(summary)
"""
        ),
        md("## Projections 90 jours"),
        code(
            """\
for result in results:
    print("\\n", result.target)
    display(result.projection.head(15))
"""
        ),
        md("## Comparaison des modeles"),
        code(
            """\
for result in results:
    print("\\n", result.target)
    cols = [
        "model", "holdout_score", "holdout_MAPE", "holdout_peak_wMAPE",
        "holdout_peak_capture", "wf_selection_score", "wf_score_median", "wf_peak_wMAPE_median",
        "wf_peak_capture_mean", "ensemble_weight"
    ]
    display(result.candidates[cols].sort_values("holdout_score"))
"""
        ),
        md("## Audit multi-holdout sans fuite"),
        code(
            """\
for result in results:
    print("\\n", result.target)
    display(result.multi_holdout)
"""
        ),
    ]
    return notebook(cells)


def credit_notebook() -> dict:
    cells = [
        md("# Forecast robuste - Credit a l'equipement\n\n" + COMMON_INTRO),
        md("## Configuration"),
        code(
            IMPORTS
            + """

config = ForecastConfig(
    input_path=Path("in/reserve_in.xlsx"),
    output_dir=Path("out"),
    holdout_len=60,
    forecast_horizon=90,
    wf_folds=5,
    wf_horizon=30,
    min_gain_vs_baseline_pct=5.0,
    peak_quantile=0.80,
    peak_metric_weight=0.45,
    peak_weight_multiplier=4.0,
    max_mape_degradation_vs_baseline_pct=25.0,
)

targets = ["Credit Décaissement_crédits à lequipement"]
"""
        ),
        md("## Inspection rapide du fichier source"),
        code(
            """\
df = load_workbook(config)
print(df.shape)
print(df.index.min(), "->", df.index.max())
display(df[targets].describe().T)
"""
        ),
        md("## Execution de la pipeline robuste"),
        code(
            """\
results = run_bundle(
    name="credit_equipement",
    targets=targets,
    config=config,
    add_total=False,
)
"""
        ),
        md("## Synthese decisionnelle"),
        code(
            """\
summary_rows = []
for result in results:
    diag = result.diagnostics.set_index("metric")["value"]
    summary_rows.append({
        "serie": result.target,
        "modele_retenu": diag["final_model"],
        "modele_propose": diag["proposed_model"],
        "modele_scenario_pic": diag["peak_scenario_model"],
        "meilleur_baseline": diag["best_baseline"],
        "meilleur_baseline_mape": diag["best_mape_baseline"],
        "mape_holdout": diag["holdout_MAPE"],
        "peak_wmape_holdout": diag["holdout_peak_wMAPE"],
        "peak_capture": diag["holdout_peak_capture"],
        "score_decision": diag["holdout_score"],
        "selection_gain_vs_baseline_pct": diag["selection_gain_vs_best_baseline_pct"],
        "selection_mape_degradation_pct": diag["selection_mape_degradation_vs_best_baseline_pct"],
        "robustesse": diag["robustness_flag"],
        "recommendation": diag["decision_recommendation"],
    })
summary = pd.DataFrame(summary_rows)
display(summary)
"""
        ),
        md("## Projection 90 jours"),
        code(
            """\
for result in results:
    display(result.projection.head(20))
"""
        ),
        md("## Comparaison des modeles"),
        code(
            """\
for result in results:
    cols = [
        "model", "holdout_score", "holdout_MAPE", "holdout_peak_wMAPE",
        "holdout_peak_capture", "wf_selection_score", "wf_score_median", "wf_peak_wMAPE_median",
        "wf_peak_capture_mean", "ensemble_weight"
    ]
    display(result.candidates[cols].sort_values("holdout_score"))
"""
        ),
        md("## Audit multi-holdout sans fuite"),
        code(
            """\
for result in results:
    display(result.multi_holdout)
"""
        ),
    ]
    return notebook(cells)


def main() -> None:
    outputs = {
        "robust_depots_cheques_courants.ipynb": depots_notebook(),
        "robust_credit_equipement.ipynb": credit_notebook(),
    }
    for path, nb in outputs.items():
        Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
