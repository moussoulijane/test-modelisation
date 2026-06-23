# Pipeline Forecast Robuste

Cette copie est isolee du projet original. Les fichiers legacy sont conserves pour reference:

- `time_series_analysis_pipeline_v7_legacy.py`
- `run_legacy.ipynb`

La version active est:

- `time_series_analysis_pipeline_v7.py`
- `run.ipynb`
- `run_robust_pipeline.py`

## Execution

Depuis ce dossier:

```powershell
python run_robust_pipeline.py
```

Ou ouvrir `run.ipynb` et executer les cellules.

## Changements principaux

- Suppression de la fuite de donnees ML: Random Forest, Gradient Boosting et XGBoost sont entraines sur train uniquement.
- Plus de `asfreq("D")` et plus d'interpolation automatique des jours fermes.
- Generation future sur jours ouvres marocains, hors week-ends et jours feries civils fixes.
- Selection du meilleur modele par WMAPE peak-aware sur holdout, pas par R2.
- Comparaison systematique contre deux baselines: derniere valeur et mediane du meme jour de semaine.
- Selection automatique des lags sur la fenetre train uniquement.
- Ajout de XGBoost si la librairie `xgboost` est installee.
- Ajout de `KNN_ANALOG_PATH`, qui recherche des trajectoires historiques similaires.
- Detection des pics et ponderation plus forte des erreurs sur pics.
- Graphiques Plotly conserves, mais les residus sont maintenant des vrais residus holdout.

## Sorties

Le dossier `out/` contient:

- `analyse_series_chronologiques_robuste.xlsx`
- `classification_distribution.html`
- `model_accuracy.html`
- `correlation_matrix.html`
- `serie_*.html`

Les anciens fichiers presents dans `out/` viennent de la copie originale. Une nouvelle execution les mettra a jour ou ajoutera les sorties robustes.
