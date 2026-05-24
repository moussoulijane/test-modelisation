# Modélisation et projection — séries bancaires quotidiennes

Deux notebooks Jupyter autonomes pour projeter à 90 jours :

- **`modelisation_comptes_cheques.ipynb`** — série `Credit Décaissement_comptes chèques`
- **`modelisation_credits_equipement.ipynb`** — série `Credit Décaissement_crédits à lequipement`

Chaque notebook : EDA, tests de stationnarité (ADF/KPSS), détection automatique des ruptures et outliers, ARIMAX avec exogènes calendaires, sélection d'ordre par walk-forward MAPE, diagnostic résidus, hold-out 60 j, projection 90 j avec intervalles de confiance empiriques asymétriques, et export Excel.

## Structure

```
.
├── in/                                       # déposer ici le .xlsx source
├── out/                                      # graphiques + projection_robuste_*.xlsx générés
├── modelisation_comptes_cheques.ipynb
├── modelisation_credits_equipement.ipynb
└── build_notebooks.py                        # générateur (pour reproduire les .ipynb)
```

## Utilisation

1. Déposer le fichier Excel source dans `in/`. Il doit contenir une colonne `date` et les colonnes `Credit Décaissement_comptes chèques` et `Credit Décaissement_crédits à lequipement`.
2. Ouvrir l'un des notebooks et exécuter de bout en bout (Restart & Run All).
3. Les sorties (graphiques + tableau de projection) apparaissent dans `out/`.

## Dépendances

```
numpy pandas matplotlib seaborn scipy statsmodels openpyxl
```

## Différences méthodologiques entre les deux séries

| Aspect | Comptes chèques | Crédits équipement |
|---|---|---|
| Transformation | log obligatoire | log si positive |
| Saisonnalité | hebdo + fin de mois | trimestrielle + août |
| Calendar dummies | DOW + EOM | EOQ + août (DOW testé par AIC) |
| Détection ruptures | 1 rupture (nov 2025) | multi-ruptures via CUSUM segmenté |
| Baseline alternatif | RW + drift | moyenne trimestrielle |
| Fiabilité projection | toute la fenêtre 90 j | dégradation marquée au-delà de 30 j |

## Tests

```bash
python build_notebooks.py        # régénère les deux .ipynb à partir de la spec
```
