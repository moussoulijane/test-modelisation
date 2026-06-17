# Modélisation et projection — séries bancaires quotidiennes

Deux notebooks Jupyter autonomes pour projeter à 90 jours :

- **`modelisation_comptes_cheques.ipynb`** — série `Credit Décaissement_comptes chèques`
- **`modelisation_credits_equipement.ipynb`** — série `Credit Décaissement_crédits à lequipement`
- **`robust_depots_cheques_courants.ipynb`** — dépôts comptes chèques, comptes courants, et total chèques+courants
- **`robust_credit_equipement.ipynb`** — crédit à l'équipement

Chaque notebook : EDA, tests de stationnarité (ADF/KPSS), détection automatique des ruptures et outliers, SARIMAX avec exogènes calendaires, sélection d'ordre par walk-forward MAPE, diagnostic résidus, hold-out 60 j, panel multi-modèles, projection 90 j avec intervalles de confiance empiriques asymétriques, et export Excel.

La sélection finale combine désormais plusieurs candidats (SARIMAX full, SARIMAX post-rupture, ETS, Theta, Holt-damped, LightGBM si installé). Les poids de l'ensemble sont appris sur la MAPE médiane walk-forward du panel complet, puis validés sur le hold-out final. Si le modèle ne bat pas le meilleur baseline de plus de 5 %, le notebook bascule explicitement sur le baseline.

Pour les crédits équipement, un modèle intermittent deux étages est ajouté : probabilité de décaissement × montant conditionnel positif. Si `lightgbm` n'est pas installé ou si l'échantillon est trop court, il bascule automatiquement sur un lissage TSB simple.

## Structure

```
.
├── in/                                       # déposer ici le .xlsx source
├── out/                                      # graphiques + projection_robuste_*.xlsx générés
├── modelisation_comptes_cheques.ipynb
├── modelisation_credits_equipement.ipynb
├── robust_depots_cheques_courants.ipynb
├── robust_credit_equipement.ipynb
├── robust_forecast_engine.py                 # moteur commun pour les notebooks robustes
├── run_robust_forecasts.py                   # exécution batch des deux nouveaux notebooks
├── build_robust_notebooks.py                 # régénère les deux notebooks robustes
└── build_notebooks.py                        # générateur (pour reproduire les .ipynb)
```

## Utilisation

1. Déposer le fichier Excel source dans `in/`. Il doit contenir une colonne `date` et les colonnes `Credit Décaissement_comptes chèques` et `Credit Décaissement_crédits à lequipement`.
2. Ouvrir l'un des notebooks et exécuter de bout en bout (Restart & Run All).
3. Les sorties (graphiques + tableau de projection) apparaissent dans `out/`.

## Dépendances

```
numpy pandas matplotlib seaborn scipy statsmodels openpyxl
lightgbm  # optionnel, active les modèles ML et intermittent avancé
```

## Différences méthodologiques entre les deux séries

| Aspect | Comptes chèques | Crédits équipement |
|---|---|---|
| Transformation | log si toutes les valeurs sont positives | log désactivé automatiquement si >15% de zéros |
| Saisonnalité | hebdo ouvrée + fin de mois | trimestrielle + août |
| Calendar dummies | DOW + EOM | EOQ + août (DOW testé par AIC) |
| Détection ruptures | 1 rupture (nov 2025) | multi-ruptures via CUSUM segmenté |
| Baselines | naïf, RW + drift, naïf saisonnier | naïf, moyenne trimestrielle, naïf saisonnier |
| Sélection finale | ensemble inverse-MAPE walk-forward ou baseline si gain <5% | ensemble inverse-MAPE walk-forward + intermittent two-stage ou baseline si gain <5% |
| Fiabilité projection | toute la fenêtre 90 j | dégradation marquée au-delà de 30 j |

## Tests

```bash
python build_notebooks.py        # régénère les deux .ipynb à partir de la spec
python build_robust_notebooks.py # régénère les deux notebooks robustes
python run_robust_forecasts.py   # exécute la pipeline robuste sur in/reserve_in.xlsx
```
