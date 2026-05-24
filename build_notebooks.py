"""
Generates the two banking time-series modelling notebooks.

Run: python build_notebooks.py
Outputs: modelisation_comptes_cheques.ipynb, modelisation_credits_equipement.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True) or [""],
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True) or [""],
    }


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ============================================================================
# Shared code blocks
# ============================================================================

IMPORTS = '''import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Style
plt.rcParams.update({
    "figure.figsize": (12, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "font.family": "DejaVu Sans",
})

COLOR_HIST = "#1F4E78"
COLOR_PRED = "#C0504D"
COLOR_CI   = "#2E75B5"
COLOR_ACCENT = "#9E480E"

print("Imports OK")'''


CONFIG_TEMPLATE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT_DIR  = Path("in")
OUTPUT_DIR = Path("out")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Target series
TARGET_COL = "{target}"

# Modelling
USE_LOG          = True       # log-transform of the target
MAX_DIFF_ORDER   = 2          # max integration order tested
HOLDOUT_LEN      = 60         # length of the hold-out window (business days)
FORECAST_HORIZON = 90         # length of the future projection (business days)
ALPHA            = 0.05       # confidence-interval significance level
ARIMA_PMAX       = 3          # max AR order in the grid
ARIMA_QMAX       = 3          # max MA order in the grid
WF_FOLDS         = 3          # walk-forward folds for order selection
OUTLIER_MAD_INIT = 5.0        # threshold for the initial outlier scan
OUTLIER_MAD_ITER = 4.0        # threshold for the iterative outlier loop
OUTLIER_MAX_ITER = 4          # number of outlier-detection iterations
FIT_MAXITER      = 250        # SARIMAX fit iterations
RUPTURE_MIN_GAP  = 20         # minimum gap (days) between candidate breakpoints

print("Config OK")
print(f"Target column: {{TARGET_COL!r}}")'''


CHARGEMENT = '''# Locate the first Excel file in the input directory
xlsx_files = sorted(INPUT_DIR.glob("*.xlsx"))
if not xlsx_files:
    raise FileNotFoundError(
        f"No .xlsx file found in {INPUT_DIR.resolve()}. "
        "Drop the source workbook in the input directory and rerun this cell."
    )

src_path = xlsx_files[0]
print(f"Reading: {src_path.name}")

raw = pd.read_excel(src_path)
raw.columns = [c.strip() for c in raw.columns]

# Identify the date column robustly
date_candidates = [c for c in raw.columns if c.lower() in ("date", "dates", "jour", "day")]
if not date_candidates:
    # fallback: first column
    date_candidates = [raw.columns[0]]
date_col = date_candidates[0]

raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True)
raw = (raw.dropna(subset=[date_col])
          .sort_values(date_col)
          .drop_duplicates(subset=[date_col], keep="last")
          .set_index(date_col))

if TARGET_COL not in raw.columns:
    # fuzzy fallback: pick the closest match by normalised string
    norm = lambda s: "".join(ch.lower() for ch in s if ch.isalnum())
    target_norm = norm(TARGET_COL)
    matches = [c for c in raw.columns if norm(c) == target_norm]
    if not matches:
        raise KeyError(
            f"Target column {TARGET_COL!r} not found. "
            f"Available columns: {list(raw.columns)[:10]}..."
        )
    actual = matches[0]
    print(f"Using fuzzy match: {actual!r}")
else:
    actual = TARGET_COL

# Keep target only, drop NaNs, force business-day index
series = raw[actual].astype(float).dropna()
series.name = actual

# Build a complete business-day index and detect gaps
full_idx = pd.bdate_range(series.index.min(), series.index.max())
n_gaps = len(full_idx.difference(series.index))

print(f"Period       : {series.index.min().date()} → {series.index.max().date()}")
print(f"Observations : {len(series)}")
print(f"Business-day gaps detected : {n_gaps}")
print(f"Inferred frequency : {pd.infer_freq(series.index) or 'irregular (business days)'}")
print(series.describe().to_string())'''


EDA_GRID = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3×2 EDA grid
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
fig, axes = plt.subplots(3, 2, figsize=(14, 11))

# (0,0) Level + linear trend
y = series.values
x = np.arange(len(y))
slope, intercept, *_ = stats.linregress(x, y)
axes[0, 0].plot(series.index, y, color=COLOR_HIST, lw=0.9, label="Niveau")
axes[0, 0].plot(series.index, intercept + slope * x, color=COLOR_ACCENT,
                lw=1.3, ls="--", label=f"Tendance (a={slope:+.2f})")
axes[0, 0].set_title("Série brute + tendance linéaire")
axes[0, 0].legend()

# (0,1) log series if positive
if (series > 0).all():
    axes[0, 1].plot(series.index, np.log(y), color=COLOR_HIST, lw=0.9)
    axes[0, 1].set_title("log(série)")
else:
    axes[0, 1].text(0.5, 0.5, "Série non strictement positive — log impossible",
                    transform=axes[0, 1].transAxes, ha="center", va="center")
    axes[0, 1].set_title("log(série) indisponible")

# (1,0) First difference
diff = series.diff().dropna()
axes[1, 0].plot(diff.index, diff.values, color=COLOR_HIST, lw=0.7)
axes[1, 0].axhline(0, color="black", lw=0.5)
axes[1, 0].set_title("Différence première")

# (1,1) Distribution
sk = stats.skew(y, bias=False)
kt = stats.kurtosis(y, bias=False)
axes[1, 1].hist(y, bins=50, color=COLOR_HIST, alpha=0.85, edgecolor="white")
axes[1, 1].set_title(f"Distribution — skew={sk:+.2f}, kurt={kt:+.2f}")

# (2,0) ACF
acf_vals = acf(series.dropna(), nlags=40, fft=True)
axes[2, 0].stem(range(len(acf_vals)), acf_vals, linefmt=COLOR_HIST, markerfmt="o", basefmt=" ")
ci_acf = 1.96 / np.sqrt(len(series))
axes[2, 0].axhline( ci_acf, color="grey", ls="--", lw=0.7)
axes[2, 0].axhline(-ci_acf, color="grey", ls="--", lw=0.7)
axes[2, 0].set_title("ACF (40 lags)")

# (2,1) PACF
pacf_vals = pacf(series.dropna(), nlags=40, method="ywm")
axes[2, 1].stem(range(len(pacf_vals)), pacf_vals, linefmt=COLOR_HIST, markerfmt="o", basefmt=" ")
axes[2, 1].axhline( ci_acf, color="grey", ls="--", lw=0.7)
axes[2, 1].axhline(-ci_acf, color="grey", ls="--", lw=0.7)
axes[2, 1].set_title("PACF (40 lags)")

plt.tight_layout()
plt.show()

cv = series.std() / max(abs(series.mean()), 1e-9)
pct_zero = (series == 0).mean() * 100
print(f"mean   = {series.mean():12.4f}")
print(f"median = {series.median():12.4f}")
print(f"std    = {series.std():12.4f}")
print(f"CV     = {cv:12.4f}")
print(f"min    = {series.min():12.4f}")
print(f"max    = {series.max():12.4f}")
print(f"% zeros= {pct_zero:12.4f}%")'''


STATIONNARITE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Combined stationarity tests (ADF + KPSS), cross-verdict, integration order
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def stationarity_check(s: pd.Series, label: str) -> tuple[bool, dict]:
    s = s.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    adf_stationary  = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05

    if adf_stationary and kpss_stationary:
        verdict = "I(0) — stationnaire"
    elif (not adf_stationary) and (not kpss_stationary):
        verdict = "I(1) — différencier"
    elif adf_stationary and not kpss_stationary:
        verdict = "Tendance déterministe — détendancer"
    else:
        verdict = "Ambigu — prudence"

    print(f"[{label}]  ADF p={adf_p:.4f}  KPSS p={kpss_p:.4f}  →  {verdict}")
    return verdict.startswith("I(0)"), {"adf_p": adf_p, "kpss_p": kpss_p, "verdict": verdict}

target_for_tests = np.log(series) if USE_LOG and (series > 0).all() else series.copy()
print(f"Series tested : {'log-' if USE_LOG and (series > 0).all() else ''}{TARGET_COL}")
print("━" * 70)

current = target_for_tests.copy()
diff_order = 0
verdicts = []
for d in range(MAX_DIFF_ORDER + 1):
    is_stat, info = stationarity_check(current, f"d={d}")
    verdicts.append(info)
    if is_stat:
        diff_order = d
        break
    current = current.diff().dropna()
else:
    diff_order = MAX_DIFF_ORDER

print("━" * 70)
print(f"Ordre d'intégration retenu : d = {diff_order}")'''


# ============================================================================
# Notebook 1: Comptes chèques
# ============================================================================

INTERVENTIONS_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Robust outlier flagging (MAD on log-diff) + breakpoint detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Work on log-differences: a robust scale-free proxy for shocks
if USE_LOG and (series > 0).all():
    work = np.log(series)
else:
    work = series.copy()
log_diff = work.diff().dropna()

median = log_diff.median()
mad    = (log_diff - median).abs().median()
mad    = max(float(mad), 1e-8)
z_rob  = (log_diff - median) / (1.4826 * mad)

initial_outliers = log_diff.index[z_rob.abs() > OUTLIER_MAD_INIT]
print(f"Outliers initiaux (|z_rob| > {OUTLIER_MAD_INIT}) : {len(initial_outliers)}")

# Candidate breakpoint: biggest jump in the LATER half of the sample
cutoff = log_diff.index[len(log_diff) // 2]
late = log_diff.loc[cutoff:]
break_candidate = late.abs().idxmax()

# Compare pre- vs post-break level
pre  = series.loc[:break_candidate].iloc[:-1]
post = series.loc[break_candidate:]
if len(pre) >= 30 and len(post) >= 30:
    mean_pre  = pre.tail(60).mean()
    mean_post = post.head(60).mean()
    jump_abs  = mean_post - mean_pre
    jump_rel  = jump_abs / max(abs(mean_pre), 1e-9)
    print(f"Rupture candidate : {break_candidate.date()}")
    print(f"  Niveau moyen avant (60j) : {mean_pre:,.1f}")
    print(f"  Niveau moyen après (60j) : {mean_post:,.1f}")
    print(f"  Saut absolu              : {jump_abs:+,.1f}")
    print(f"  Saut relatif             : {jump_rel:+.2%}")
else:
    print("Échantillon trop court pour confirmer la rupture — on l'ignore.")
    break_candidate = None

# Visualise the diagnostic
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(series.index, series.values, color=COLOR_HIST, lw=0.9)
if break_candidate is not None:
    ax[0].axvline(break_candidate, color=COLOR_ACCENT, ls="--", lw=1.2,
                  label=f"Rupture {break_candidate.date()}")
    ax[0].legend()
ax[0].set_title("Niveau brut + rupture candidate")

ax[1].plot(log_diff.index, z_rob.values, color=COLOR_HIST, lw=0.7)
ax[1].scatter(initial_outliers, z_rob.loc[initial_outliers], color=COLOR_ACCENT,
              zorder=3, s=22, label=f"Outliers (n={len(initial_outliers)})")
ax[1].axhline( OUTLIER_MAD_INIT, color="grey", ls="--", lw=0.6)
ax[1].axhline(-OUTLIER_MAD_INIT, color="grey", ls="--", lw=0.6)
ax[1].set_title("MAD-robust z-score sur log-différence")
ax[1].legend()

plt.tight_layout()
plt.show()'''


EXOG_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exogenous regressors: DOW dummies + end-of-month + step + pulses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_exog(idx: pd.DatetimeIndex,
               break_date,
               pulse_dates: list,
               include_dow: bool = True,
               include_eom: bool = True,
               include_eoq: bool = False,
               include_august: bool = False) -> pd.DataFrame:
    exog = pd.DataFrame(index=idx)

    if include_dow:
        for k, name in enumerate(["dow_tue", "dow_wed", "dow_thu", "dow_fri",
                                  "dow_sat", "dow_sun"], start=1):
            exog[name] = (idx.dayofweek == k).astype(float)

    if include_eom:
        exog["eom"] = (idx.day >= 25).astype(float)

    if include_eoq:
        is_q_month = idx.month.isin([3, 6, 9, 12])
        exog["eoq"] = (is_q_month & (idx.day >= 20)).astype(float)

    if include_august:
        exog["august"] = (idx.month == 8).astype(float)

    if break_date is not None:
        exog["step_break"] = (idx >= break_date).astype(float)

    for i, d in enumerate(pulse_dates):
        exog[f"pulse_{i:02d}"] = (idx == pd.Timestamp(d)).astype(float)

    return exog

pulse_dates = list(initial_outliers)
exog_train = build_exog(series.index, break_candidate, pulse_dates,
                        include_dow=True, include_eom=True)

print(f"Exog columns ({exog_train.shape[1]}) : {list(exog_train.columns)[:8]} ...")
print(exog_train.tail(3))'''


ITER_OUTLIERS = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Iterative outlier loop on ARIMAX(1,1,1) residuals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fit_arimax(y, exog, order=(1, 1, 1), maxiter=FIT_MAXITER):
    try:
        model = SARIMAX(y, exog=exog, order=order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        return model.fit(disp=False, maxiter=maxiter)
    except Exception as e:
        print(f"  fit error ({order}): {e}")
        return None

y_target = np.log(series) if (USE_LOG and (series > 0).all()) else series.copy()

extra_pulses: list = []
current_exog = exog_train.copy()

for it in range(1, OUTLIER_MAX_ITER + 1):
    res = fit_arimax(y_target, current_exog, order=(1, 1, 1))
    if res is None:
        print(f"[iter {it}] échec du fit — arrêt")
        break

    resid = res.resid
    med   = resid.median()
    mad   = (resid - med).abs().median()
    mad   = max(float(mad), 1e-9)
    zres  = (resid - med) / (1.4826 * mad)

    new_outliers = resid.index[zres.abs() > OUTLIER_MAD_ITER]
    new_outliers = [d for d in new_outliers if d not in extra_pulses and d not in pulse_dates]

    print(f"[iter {it}]  resid_std={resid.std():.5f}  nouveaux outliers={len(new_outliers)}")

    if not new_outliers:
        print("Convergence — pas de nouveaux outliers.")
        break

    extra_pulses.extend(new_outliers)
    for i, d in enumerate(new_outliers):
        current_exog[f"pulse_extra_{it:02d}_{i:02d}"] = (
            current_exog.index == pd.Timestamp(d)
        ).astype(float)

print("━" * 70)
print(f"Total outliers pulses : {len(pulse_dates) + len(extra_pulses)}")
print(f"Exog final width      : {current_exog.shape[1]}")'''


WALK_FORWARD = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Walk-forward order selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def back_to_original(yhat_log, target_resid_var):
    """Log-normal bias correction when returning to the original scale."""
    if USE_LOG and (series > 0).all():
        return np.exp(yhat_log + 0.5 * target_resid_var)
    return yhat_log

def walk_forward_mape(order, y, exog, folds=WF_FOLDS, val_len=30):
    n_total = len(y)
    train_min = n_total - folds * val_len
    if train_min < 60:
        return np.nan
    mapes = []
    for k in range(folds):
        end_train = train_min + k * val_len
        y_tr = y.iloc[:end_train]
        y_va = y.iloc[end_train:end_train + val_len]
        x_tr = exog.iloc[:end_train]
        x_va = exog.iloc[end_train:end_train + val_len]
        res = fit_arimax(y_tr, x_tr, order=order)
        if res is None:
            return np.nan
        fc = res.get_forecast(steps=len(y_va), exog=x_va)
        yhat_log = fc.predicted_mean.values
        sigma2 = float(res.params.get("sigma2", res.scale)) if hasattr(res, "scale") else 1e-6
        yhat = back_to_original(yhat_log, sigma2)
        y_va_orig = (np.exp(y_va.values) if (USE_LOG and (series > 0).all()) else y_va.values)
        denom = np.where(np.abs(y_va_orig) < 1e-6, 1e-6, np.abs(y_va_orig))
        mapes.append(np.mean(np.abs(yhat - y_va_orig) / denom) * 100)
    return float(np.mean(mapes))

grid_results = []
for p in range(ARIMA_PMAX + 1):
    for q in range(ARIMA_QMAX + 1):
        order = (p, 1, q)
        mape = walk_forward_mape(order, y_target, current_exog)
        # quick AIC on the whole sample for reference
        res_full = fit_arimax(y_target, current_exog, order=order)
        aic = res_full.aic if res_full is not None else np.nan
        grid_results.append({"order": order, "mape_wf": mape, "aic": aic})
        print(f"  order={order}  mape_wf={mape:7.3f}%  aic={aic:10.2f}")

grid_df = pd.DataFrame(grid_results).sort_values("mape_wf").reset_index(drop=True)
print("━" * 70)
print("Top 5 ordres par MAPE walk-forward :")
print(grid_df.head(5).to_string(index=False))

best_order = grid_df.iloc[0]["order"]
print(f"\\nOrdre retenu : {best_order}")'''


FIT_DIAG = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final fit + residual diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final_res = fit_arimax(y_target, current_exog, order=best_order, maxiter=FIT_MAXITER + 100)
print(final_res.summary())

resid = final_res.resid.dropna()

# Tests
lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
jb_stat, jb_p, jb_skew, jb_kurt = stats.jarque_bera(resid.values)

print("\\nLjung-Box :")
print(lb.to_string())
print(f"\\nJarque-Bera : stat={jb_stat:.3f}, p={jb_p:.4g}")
print(f"  skew={jb_skew:+.3f}  kurt(excès)={jb_kurt:+.3f}")

# Plots: residual series, distribution, QQ, ACF, PACF
fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.25)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(resid.index, resid.values, color=COLOR_HIST, lw=0.7)
ax1.axhline(0, color="black", lw=0.5)
ax1.set_title("Résidus")

ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(resid.values, bins=50, color=COLOR_HIST, alpha=0.85, edgecolor="white")
ax2.set_title("Distribution des résidus")

ax3 = fig.add_subplot(gs[1, 1])
sm.qqplot(resid.values, line="s", ax=ax3, markerfacecolor=COLOR_HIST,
          markeredgecolor=COLOR_HIST, alpha=0.6)
ax3.set_title("QQ-plot (normale)")

ax4 = fig.add_subplot(gs[2, 0])
plot_acf(resid, lags=40, ax=ax4, color=COLOR_HIST)
ax4.set_title("ACF résidus")

ax5 = fig.add_subplot(gs[2, 1])
plot_pacf(resid, lags=40, ax=ax5, method="ywm", color=COLOR_HIST)
ax5.set_title("PACF résidus")

plt.show()'''


HOLDOUT_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hold-out evaluation vs naive + random-walk-with-drift
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_end = len(series) - HOLDOUT_LEN
y_train_log  = y_target.iloc[:train_end]
y_test_log   = y_target.iloc[train_end:]
exog_train_sub = current_exog.iloc[:train_end]
exog_test_sub  = current_exog.iloc[train_end:]

res_train = fit_arimax(y_train_log, exog_train_sub, order=best_order,
                       maxiter=FIT_MAXITER + 100)
fc = res_train.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)

sigma2_log = float(np.var(res_train.resid))
yhat_log = fc.predicted_mean.values
yhat = back_to_original(yhat_log, sigma2_log)

y_test_orig = (np.exp(y_test_log.values)
               if (USE_LOG and (series > 0).all()) else y_test_log.values)

# Metrics
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b):
    denom = np.where(np.abs(b) < 1e-6, 1e-6, np.abs(b))
    return float(np.mean(np.abs(a - b) / denom) * 100)

mae_model  = mae(yhat, y_test_orig)
rmse_model = rmse(yhat, y_test_orig)
mape_model = mape(yhat, y_test_orig)

# Naive (last value of train)
y_train_orig = series.iloc[:train_end].values
naive_pred   = np.full(HOLDOUT_LEN, y_train_orig[-1])
mae_naive  = mae(naive_pred, y_test_orig)
rmse_naive = rmse(naive_pred, y_test_orig)
mape_naive = mape(naive_pred, y_test_orig)

# Random walk with drift (in original scale)
drift = np.mean(np.diff(y_train_orig))
rwd_pred = y_train_orig[-1] + drift * np.arange(1, HOLDOUT_LEN + 1)
mae_rwd  = mae(rwd_pred, y_test_orig)
rmse_rwd = rmse(rwd_pred, y_test_orig)
mape_rwd = mape(rwd_pred, y_test_orig)

print("━" * 70)
print(f"{'Modèle':<28}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
print(f"{'ARIMAX' + str(best_order):<28}{mae_model:>10.2f}{rmse_model:>10.2f}{mape_model:>9.2f}%")
print(f"{'Naïf (last value)':<28}{mae_naive:>10.2f}{rmse_naive:>10.2f}{mape_naive:>9.2f}%")
print(f"{'Random walk + drift':<28}{mae_rwd:>10.2f}{rmse_rwd:>10.2f}{mape_rwd:>9.2f}%")
print("━" * 70)
gain_naive = (mape_naive - mape_model) / mape_naive * 100
gain_rwd   = (mape_rwd   - mape_model) / max(mape_rwd, 1e-6) * 100
print(f"Gain MAPE vs naïf       : {gain_naive:+.2f}%")
print(f"Gain MAPE vs RW + drift : {gain_rwd:+.2f}%")'''


PROJECTION_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Future projection with EMPIRICAL asymmetric confidence intervals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

future_idx = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1),
                            periods=FORECAST_HORIZON)

exog_future = build_exog(future_idx, break_candidate, [],
                         include_dow=True, include_eom=True)

# Align with the columns the model was fitted on (extra pulses are 0 in the future)
for col in current_exog.columns:
    if col not in exog_future.columns:
        exog_future[col] = 0.0
exog_future = exog_future[current_exog.columns]

# Final model already covers the full sample → use final_res
fc_future = final_res.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
yhat_log_future = fc_future.predicted_mean.values
se_log_future   = fc_future.se_mean.values

# Empirical asymmetric residual quantiles
resid_std = (resid - resid.mean()) / max(resid.std(), 1e-9)
q_lo = np.quantile(resid_std.values,       ALPHA / 2)   # ≈ -1.96 if normal
q_hi = np.quantile(resid_std.values, 1.0 - ALPHA / 2)

lo_log = yhat_log_future + q_lo * se_log_future
hi_log = yhat_log_future + q_hi * se_log_future

sigma2_log = float(np.var(resid))
yhat_future = back_to_original(yhat_log_future, sigma2_log)
lo_future   = back_to_original(lo_log,          sigma2_log)
hi_future   = back_to_original(hi_log,          sigma2_log)

projection = pd.DataFrame({
    "date"    : future_idx,
    "previs"  : yhat_future,
    "ic_bas"  : lo_future,
    "ic_haut" : hi_future,
    "day_of_week": future_idx.day_name(),
})
print(projection.head(15).to_string(index=False))
print(f"\\nProjection sur {FORECAST_HORIZON} jours ouvrés "
      f"({future_idx[0].date()} → {future_idx[-1].date()})")'''


PLOT_FINAL_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final visualisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

hist_window = 300
hist_tail = series.iloc[-hist_window:]
test_dates = series.index[train_end:]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(hist_tail.index, hist_tail.values, color=COLOR_HIST, lw=1.1,
        label="Historique (300 derniers j)")
ax.plot(test_dates, y_test_orig, color="black", lw=1.0, alpha=0.7,
        label="Réalisé hold-out")
ax.plot(test_dates, yhat, color=COLOR_ACCENT, lw=1.2, ls="--",
        label=f"Prédit hold-out (MAPE={mape_model:.2f}%)")
ax.plot(future_idx, yhat_future, color=COLOR_PRED, lw=1.4,
        label="Projection 90 j")
ax.fill_between(future_idx, lo_future, hi_future, color=COLOR_CI, alpha=0.20,
                label=f"IC {int((1 - ALPHA) * 100)}% (empirique)")

if break_candidate is not None:
    ax.axvline(break_candidate, color="grey", ls=":", lw=1.1)
    ax.text(break_candidate, ax.get_ylim()[1], f" rupture {break_candidate.date()}",
            color="grey", va="top", fontsize=9)

ax.set_title(f"Projection — {TARGET_COL}\\n"
             f"ARIMAX{best_order}  |  MAPE test = {mape_model:.2f}%  |  "
             f"Gain vs naïf = {gain_naive:+.1f}%")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()'''


EXPORT_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Excel export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

safe_name = (TARGET_COL.replace(" ", "_")
                       .replace("/", "_")
                       .replace("\\\\", "_")
                       .replace("'", ""))
out_path = OUTPUT_DIR / f"projection_robuste_{safe_name}.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    projection.to_excel(writer, sheet_name="projection", index=False)
    pd.DataFrame({
        "metric": ["MAPE_test", "MAE_test", "RMSE_test",
                   "MAPE_naive", "MAPE_rwd", "Gain_vs_naive_pct",
                   "ARIMA_order", "USE_LOG", "horizon_days",
                   "n_obs_total", "n_obs_train", "break_date"],
        "value":  [mape_model, mae_model, rmse_model,
                   mape_naive, mape_rwd, gain_naive,
                   str(best_order), USE_LOG, FORECAST_HORIZON,
                   len(series), train_end,
                   str(break_candidate.date()) if break_candidate is not None else ""],
    }).to_excel(writer, sheet_name="diagnostics", index=False)

print(f"✅ Projection exportée : {out_path.resolve()}")'''


# ============================================================================
# Notebook 2: Crédits équipement
# ============================================================================

INTERVENTIONS_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-breakpoint detection via CUSUM on log-returns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Equipment-loan disbursements come in lumps → segmented CUSUM may detect
# several level shifts. We scan iteratively and keep candidates that survive
# a minimum-gap constraint.

if USE_LOG and (series > 0).all():
    work = np.log(series)
else:
    work = series.copy()
log_diff = work.diff().dropna()

median = log_diff.median()
mad    = (log_diff - median).abs().median()
mad    = max(float(mad), 1e-8)
z_rob  = (log_diff - median) / (1.4826 * mad)
initial_outliers = log_diff.index[z_rob.abs() > OUTLIER_MAD_INIT]
print(f"Outliers initiaux (|z_rob| > {OUTLIER_MAD_INIT}) : {len(initial_outliers)}")

def cusum_breakpoints(s: pd.Series, max_breaks: int = 3, min_gap: int = RUPTURE_MIN_GAP):
    """Greedy CUSUM-style segmentation: at each step, pick the index that
    splits the residual variance the most, then recurse inside each segment."""
    s = s.dropna()
    n = len(s)
    if n < 60:
        return []

    def best_split(start: int, end: int):
        seg = s.iloc[start:end].values
        L = len(seg)
        if L < 30:
            return None, -np.inf
        cumsum = np.cumsum(seg - seg.mean())
        # max absolute deviation = strongest single-shift candidate
        k = int(np.argmax(np.abs(cumsum[5:-5]))) + 5
        # Variance gain vs unsplit
        var_full = np.var(seg)
        var_l = np.var(seg[:k]) if k > 5 else var_full
        var_r = np.var(seg[k:]) if (L - k) > 5 else var_full
        var_split = (k * var_l + (L - k) * var_r) / L
        gain = var_full - var_split
        return start + k, gain

    segments = [(0, n)]
    breaks = []
    while len(breaks) < max_breaks:
        candidates = []
        for (a, b) in segments:
            idx, gain = best_split(a, b)
            if idx is not None and gain > 0:
                candidates.append((idx, gain, a, b))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_gain, a, b = candidates[0]
        # enforce gap
        too_close = any(abs(best_idx - k) < min_gap for k in [seg[0] for seg in segments] + [seg[1] for seg in segments])
        if too_close:
            break
        breaks.append(best_idx)
        segments = sorted(set([
            *[(a2, b2) for (a2, b2) in segments if not (a2 == a and b2 == b)],
            (a, best_idx),
            (best_idx, b),
        ]))
    return sorted({s.index[k] for k in breaks})

break_dates = cusum_breakpoints(work, max_breaks=3)
print(f"Ruptures détectées (CUSUM segmenté) : {len(break_dates)}")
for d in break_dates:
    print(f"  - {d.date()}")

# Compute jump magnitudes for context
for d in break_dates:
    pre  = series.loc[:d].tail(40)
    post = series.loc[d:].head(40)
    if len(pre) >= 10 and len(post) >= 10:
        print(f"  {d.date()}  niveau avant={pre.mean():,.1f}  niveau après={post.mean():,.1f}  "
              f"saut={post.mean() - pre.mean():+,.1f}")

# Visualise
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax[0].plot(series.index, series.values, color=COLOR_HIST, lw=0.9)
for d in break_dates:
    ax[0].axvline(d, color=COLOR_ACCENT, ls="--", lw=1.0, alpha=0.7)
ax[0].set_title(f"Niveau brut + ruptures candidates ({len(break_dates)})")

ax[1].plot(log_diff.index, z_rob.values, color=COLOR_HIST, lw=0.7)
ax[1].scatter(initial_outliers, z_rob.loc[initial_outliers], color=COLOR_ACCENT,
              zorder=3, s=22, label=f"Outliers (n={len(initial_outliers)})")
ax[1].axhline( OUTLIER_MAD_INIT, color="grey", ls="--", lw=0.6)
ax[1].axhline(-OUTLIER_MAD_INIT, color="grey", ls="--", lw=0.6)
ax[1].set_title("MAD-robust z-score sur log-différence")
ax[1].legend()

plt.tight_layout()
plt.show()'''


EXOG_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exogenous regressors: end-of-quarter + August + step(s) + pulses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_exog(idx: pd.DatetimeIndex,
               break_dates: list,
               pulse_dates: list,
               include_dow: bool = False,
               include_eom: bool = False,
               include_eoq: bool = True,
               include_august: bool = True) -> pd.DataFrame:
    exog = pd.DataFrame(index=idx)

    if include_dow:
        for k, name in enumerate(["dow_tue", "dow_wed", "dow_thu", "dow_fri",
                                  "dow_sat", "dow_sun"], start=1):
            exog[name] = (idx.dayofweek == k).astype(float)

    if include_eom:
        exog["eom"] = (idx.day >= 25).astype(float)

    if include_eoq:
        is_q_month = idx.month.isin([3, 6, 9, 12])
        exog["eoq"] = (is_q_month & (idx.day >= 20)).astype(float)

    if include_august:
        exog["august"] = (idx.month == 8).astype(float)

    for i, d in enumerate(break_dates):
        exog[f"step_break_{i:02d}"] = (idx >= pd.Timestamp(d)).astype(float)

    for i, d in enumerate(pulse_dates):
        exog[f"pulse_{i:02d}"] = (idx == pd.Timestamp(d)).astype(float)

    return exog

pulse_dates = list(initial_outliers)
exog_train = build_exog(series.index, break_dates, pulse_dates,
                        include_dow=False, include_eoq=True, include_august=True)

# Quick test: are DOW dummies actually useful here?
exog_with_dow = build_exog(series.index, break_dates, pulse_dates,
                           include_dow=True, include_eoq=True, include_august=True)
y_target = np.log(series) if (USE_LOG and (series > 0).all()) else series.copy()

def aic_of(exog):
    try:
        m = SARIMAX(y_target, exog=exog, order=(1, 1, 1),
                    enforce_stationarity=False, enforce_invertibility=False)
        return m.fit(disp=False, maxiter=200).aic
    except Exception:
        return np.nan

aic_no_dow   = aic_of(exog_train)
aic_with_dow = aic_of(exog_with_dow)
print(f"AIC sans DOW : {aic_no_dow:.2f}")
print(f"AIC avec DOW : {aic_with_dow:.2f}")
if aic_with_dow + 6 < aic_no_dow:  # ≥ 6 points required to justify 6 extra params
    print("→ DOW retenu (gain AIC suffisant)")
    exog_train = exog_with_dow
else:
    print("→ DOW retiré (gain AIC insuffisant)")

print(f"\\nExog columns ({exog_train.shape[1]}) : {list(exog_train.columns)}")
print(exog_train.tail(3))'''


ITER_OUTLIERS_CE = ITER_OUTLIERS  # same iterative logic


HOLDOUT_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hold-out evaluation vs naive + quarterly-mean baseline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_end = len(series) - HOLDOUT_LEN
y_train_log    = y_target.iloc[:train_end]
y_test_log     = y_target.iloc[train_end:]
exog_train_sub = current_exog.iloc[:train_end]
exog_test_sub  = current_exog.iloc[train_end:]

res_train = fit_arimax(y_train_log, exog_train_sub, order=best_order,
                       maxiter=FIT_MAXITER + 100)
fc = res_train.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)

sigma2_log = float(np.var(res_train.resid))
yhat_log = fc.predicted_mean.values
yhat = back_to_original(yhat_log, sigma2_log)

y_test_orig = (np.exp(y_test_log.values)
               if (USE_LOG and (series > 0).all()) else y_test_log.values)

def mae(a, b):  return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mape(a, b):
    denom = np.where(np.abs(b) < 1e-6, 1e-6, np.abs(b))
    return float(np.mean(np.abs(a - b) / denom) * 100)

mae_model  = mae(yhat, y_test_orig)
rmse_model = rmse(yhat, y_test_orig)
mape_model = mape(yhat, y_test_orig)

# Naive baseline
y_train_orig = series.iloc[:train_end].values
naive_pred   = np.full(HOLDOUT_LEN, y_train_orig[-1])
mae_naive  = mae(naive_pred, y_test_orig)
rmse_naive = rmse(naive_pred, y_test_orig)
mape_naive = mape(naive_pred, y_test_orig)

# Quarterly-mean baseline: predict each test point with the train mean of
# observations belonging to the same calendar quarter.
test_idx = series.index[train_end:]
train_idx = series.index[:train_end]
train_q   = pd.Series(y_train_orig, index=train_idx).groupby(train_idx.quarter).mean()
qmean_pred = np.array([train_q.get(d.quarter, y_train_orig[-1]) for d in test_idx])
mae_qm  = mae(qmean_pred, y_test_orig)
rmse_qm = rmse(qmean_pred, y_test_orig)
mape_qm = mape(qmean_pred, y_test_orig)

print("━" * 70)
print(f"{'Modèle':<28}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}")
print(f"{'ARIMAX' + str(best_order):<28}{mae_model:>10.2f}{rmse_model:>10.2f}{mape_model:>9.2f}%")
print(f"{'Naïf (last value)':<28}{mae_naive:>10.2f}{rmse_naive:>10.2f}{mape_naive:>9.2f}%")
print(f"{'Quarterly mean':<28}{mae_qm:>10.2f}{rmse_qm:>10.2f}{mape_qm:>9.2f}%")
print("━" * 70)
gain_naive = (mape_naive - mape_model) / mape_naive * 100
gain_qm    = (mape_qm    - mape_model) / max(mape_qm, 1e-6) * 100
print(f"Gain MAPE vs naïf            : {gain_naive:+.2f}%")
print(f"Gain MAPE vs quarterly mean  : {gain_qm:+.2f}%")'''


PROJECTION_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Future projection with EMPIRICAL asymmetric confidence intervals
# Note: fiabilité limitée au-delà de ~30 jours vu le caractère lumpy de la série
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

future_idx = pd.bdate_range(series.index[-1] + pd.Timedelta(days=1),
                            periods=FORECAST_HORIZON)

include_dow_flag = any(c.startswith("dow_") for c in current_exog.columns)
exog_future = build_exog(future_idx, break_dates, [],
                         include_dow=include_dow_flag,
                         include_eoq=True, include_august=True)

# Align with the columns the model was fitted on
for col in current_exog.columns:
    if col not in exog_future.columns:
        exog_future[col] = 0.0
exog_future = exog_future[current_exog.columns]

fc_future = final_res.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
yhat_log_future = fc_future.predicted_mean.values
se_log_future   = fc_future.se_mean.values

resid_std = (resid - resid.mean()) / max(resid.std(), 1e-9)
q_lo = np.quantile(resid_std.values,       ALPHA / 2)
q_hi = np.quantile(resid_std.values, 1.0 - ALPHA / 2)

lo_log = yhat_log_future + q_lo * se_log_future
hi_log = yhat_log_future + q_hi * se_log_future

sigma2_log = float(np.var(resid))
yhat_future = back_to_original(yhat_log_future, sigma2_log)
lo_future   = back_to_original(lo_log,          sigma2_log)
hi_future   = back_to_original(hi_log,          sigma2_log)

projection = pd.DataFrame({
    "date"    : future_idx,
    "previs"  : yhat_future,
    "ic_bas"  : lo_future,
    "ic_haut" : hi_future,
    "day_of_week": future_idx.day_name(),
})
print(projection.head(15).to_string(index=False))
print(f"\\nProjection sur {FORECAST_HORIZON} jours ouvrés "
      f"({future_idx[0].date()} → {future_idx[-1].date()})")
print("⚠  Au-delà de ~30 jours la projection devient indicative : "
      "la série est dominée par des à-coups d'investissement non prévisibles.")'''


PLOT_FINAL_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final visualisation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

hist_window = 300
hist_tail = series.iloc[-hist_window:]
test_dates = series.index[train_end:]

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(hist_tail.index, hist_tail.values, color=COLOR_HIST, lw=1.1,
        label="Historique (300 derniers j)")
ax.plot(test_dates, y_test_orig, color="black", lw=1.0, alpha=0.7,
        label="Réalisé hold-out")
ax.plot(test_dates, yhat, color=COLOR_ACCENT, lw=1.2, ls="--",
        label=f"Prédit hold-out (MAPE={mape_model:.2f}%)")
ax.plot(future_idx, yhat_future, color=COLOR_PRED, lw=1.4,
        label="Projection 90 j")
ax.fill_between(future_idx, lo_future, hi_future, color=COLOR_CI, alpha=0.20,
                label=f"IC {int((1 - ALPHA) * 100)}% (empirique)")

for d in break_dates:
    ax.axvline(d, color="grey", ls=":", lw=1.0, alpha=0.8)

ax.set_title(f"Projection — {TARGET_COL}\\n"
             f"ARIMAX{best_order}  |  MAPE test = {mape_model:.2f}%  |  "
             f"Gain vs naïf = {gain_naive:+.1f}%")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()'''


EXPORT_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Excel export
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

safe_name = (TARGET_COL.replace(" ", "_")
                       .replace("/", "_")
                       .replace("\\\\", "_")
                       .replace("'", ""))
out_path = OUTPUT_DIR / f"projection_robuste_{safe_name}.xlsx"

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    projection.to_excel(writer, sheet_name="projection", index=False)
    pd.DataFrame({
        "metric": ["MAPE_test", "MAE_test", "RMSE_test",
                   "MAPE_naive", "MAPE_quarterly_mean", "Gain_vs_naive_pct",
                   "ARIMA_order", "USE_LOG", "horizon_days",
                   "n_obs_total", "n_obs_train", "n_breaks"],
        "value":  [mape_model, mae_model, rmse_model,
                   mape_naive, mape_qm, gain_naive,
                   str(best_order), USE_LOG, FORECAST_HORIZON,
                   len(series), train_end, len(break_dates)],
    }).to_excel(writer, sheet_name="diagnostics", index=False)
    if break_dates:
        pd.DataFrame({"break_date": [d.date() for d in break_dates]})\
          .to_excel(writer, sheet_name="breakpoints", index=False)

print(f"✅ Projection exportée : {out_path.resolve()}")'''


# ============================================================================
# Markdown cells (per notebook)
# ============================================================================

MD_INTRO_CC = '''# Modélisation et projection — Comptes chèques

Décaissements quotidiens sur comptes chèques. La série est très bruitée, hétéroscédastique, et présente une rupture de niveau en fin d'historique. La stratégie retenue : log-transformation, ARIMAX avec dummies de jour-de-semaine et fin-de-mois, step dummy pour la rupture, et intervalles de confiance construits à partir des quantiles empiriques des résidus standardisés plutôt qu'à partir d'une hypothèse gaussienne qui serait rejetée par Jarque-Bera.

Chaque section ci-dessous explique d'abord la raison de l'étape, puis la met en œuvre.'''

MD_CONFIG = '''## Imports et configuration

Tous les paramètres modifiables sont regroupés ici. Modifier ces constantes suffit à reconduire l'étude sur une autre fenêtre ou un autre seuil de détection sans avoir à fouiller le notebook.'''

MD_LOAD = '''## Chargement et inspection initiale

Lecture du premier fichier Excel trouvé dans `in/`, parsing de la colonne `date`, déduplication, mise en index temporel et détection des trous en jours ouvrés. Si la colonne cible exacte n'existe pas, une recherche tolérante aux espaces et casse est tentée.'''

MD_EDA = '''## EDA ciblé

Six diagnostics visuels en une planche : niveau + tendance linéaire, log de la série, différence première, distribution avec skew/kurt, ACF et PACF à 40 retards. Les statistiques univariées complètent le diagnostic.'''

MD_STAT = '''## Tests de stationnarité combinés

ADF rejette une racine unitaire ; KPSS rejette la stationnarité. Les deux tests sont complémentaires : un verdict croisé identifie sans ambiguïté le cas I(0), I(1), trend-stationnaire ou ambigu. L'ordre d'intégration `d` est ensuite déterminé par itération.'''

MD_INTERV_CC = '''## Détection automatique des interventions

MAD-robust z-score sur la log-différence pour repérer les chocs ponctuels. La rupture de niveau attendue (passage du palier ~3300 à ~4100 en novembre 2025) est cherchée comme le plus gros saut dans la seconde moitié de l'échantillon. On compare ensuite niveau moyen avant/après pour s'assurer que la rupture est réelle et non un simple outlier.'''

MD_EXOG_CC = '''## Construction des régresseurs exogènes

Le calendrier bancaire impose des dummies jour-de-semaine (effet vendredi vs lundi), un indicateur fin-de-mois (clôtures, virements de paie), un step de rupture, et des pulses pour chaque outlier détecté.'''

MD_ITER = '''## Détection itérative d'outliers résiduels

Après le premier fit, certains résidus restent extrêmes : on les pulse-out et on refit. Quatre itérations max, arrêt à la convergence. Ce nettoyage progressif est plus stable qu'un seuil unique appliqué d'entrée de jeu.'''

MD_WF = '''## Sélection d'ordre par walk-forward

L'AIC tend à surajuster les dummies (elles diminuent artificiellement le critère). On choisit donc l'ordre `(p, 1, q)` qui minimise le MAPE moyen sur trois plis walk-forward de 30 jours — la métrique qu'on cherche réellement à minimiser en projection.'''

MD_FIT = '''## Ajustement final et diagnostic résidus

Refit sur l'échantillon complet avec l'ordre retenu. Ljung-Box (autocorrélation résiduelle) et Jarque-Bera (normalité) qualifient la qualité du fit. La normalité est presque toujours rejetée — c'est précisément pour cela qu'on n'utilisera pas les IC gaussiens en projection.'''

MD_HOLD = '''## Évaluation hold-out

Refit sur les 60 derniers jours retirés, prédiction, métriques MAE / RMSE / MAPE. Comparaison à deux baselines : naïf (dernière valeur du train) et random-walk-with-drift. Si la complexité du modèle ne paie pas vs ces baselines, on doit s'interroger.'''

MD_PROJ_CC = '''## Projection 90 jours avec IC empiriques asymétriques

L'erreur-type de prévision SARIMAX est conservée, mais les quantiles 2.5 % / 97.5 % sont pris sur la distribution empirique des résidus standardisés, pas sur ±1.96. La correction `exp(μ + σ²/2)` rétablit l'espérance lors du retour à l'échelle d'origine.'''

MD_VIZ = '''## Visualisation finale et export

Plot principal et écriture du tableau de projection dans `out/`.'''


MD_INTRO_CE = '''# Modélisation et projection — Crédits à l'équipement

Décaissements quotidiens de crédits d'investissement aux entreprises. Le comportement diffère radicalement des comptes chèques : moins de bruit hebdomadaire, mais des à-coups marqués liés à des projets d'équipement, un creux estival en août et des pics en fin de trimestre. Plusieurs régimes de niveau sont possibles — on les détecte par segmentation CUSUM plutôt qu'en supposant une seule rupture.

Les intervalles de confiance seront naturellement plus larges qu'en comptes chèques, et la fiabilité de la projection se dégrade plus vite avec l'horizon.'''

MD_INTERV_CE = '''## Détection automatique des interventions

Plutôt qu'une rupture unique, on procède par segmentation greedy CUSUM : à chaque étape, on identifie l'index qui maximise le gain de variance après split. Une contrainte de gap minimum évite les ruptures collées les unes aux autres.'''

MD_EXOG_CE = '''## Construction des régresseurs exogènes

Les décisions d'équipement ne sont pas hebdomadaires, donc pas de DOW par défaut — on teste néanmoins leur utilité par AIC. Les dummies actives sont end-of-quarter (clôtures, budgets), août (creux estival), step(s) pour chaque rupture et pulses pour les outliers.'''

MD_PROJ_CE = '''## Projection 90 jours avec IC empiriques asymétriques

Même mécanique d'IC empiriques que pour les comptes chèques. Attention : la fiabilité se dégrade significativement au-delà de 30 jours sur cette série dominée par des à-coups d'investissement non récurrents.'''


# ============================================================================
# Assemble notebooks
# ============================================================================

def cc_cells() -> list[dict]:
    return [
        md(MD_INTRO_CC),
        md(MD_CONFIG),
        code(IMPORTS),
        code(CONFIG_TEMPLATE.format(target="Credit Décaissement_comptes chèques")),
        md(MD_LOAD),
        code(CHARGEMENT),
        md(MD_EDA),
        code(EDA_GRID),
        md(MD_STAT),
        code(STATIONNARITE),
        md(MD_INTERV_CC),
        code(INTERVENTIONS_CC),
        md(MD_EXOG_CC),
        code(EXOG_CC),
        md(MD_ITER),
        code(ITER_OUTLIERS),
        md(MD_WF),
        code(WALK_FORWARD),
        md(MD_FIT),
        code(FIT_DIAG),
        md(MD_HOLD),
        code(HOLDOUT_CC),
        md(MD_PROJ_CC),
        code(PROJECTION_CC),
        md(MD_VIZ),
        code(PLOT_FINAL_CC),
        code(EXPORT_CC),
    ]


def ce_cells() -> list[dict]:
    return [
        md(MD_INTRO_CE),
        md(MD_CONFIG),
        code(IMPORTS),
        code(CONFIG_TEMPLATE.format(target="Credit Décaissement_crédits à lequipement")),
        md(MD_LOAD),
        code(CHARGEMENT),
        md(MD_EDA),
        code(EDA_GRID),
        md(MD_STAT),
        code(STATIONNARITE),
        md(MD_INTERV_CE),
        code(INTERVENTIONS_CE),
        md(MD_EXOG_CE),
        code(EXOG_CE),
        md(MD_ITER),
        code(ITER_OUTLIERS_CE),
        md(MD_WF),
        code(WALK_FORWARD),
        md(MD_FIT),
        code(FIT_DIAG),
        md(MD_HOLD),
        code(HOLDOUT_CE),
        md(MD_PROJ_CE),
        code(PROJECTION_CE),
        md(MD_VIZ),
        code(PLOT_FINAL_CE),
        code(EXPORT_CE),
    ]


def main():
    here = Path(__file__).resolve().parent

    nb1 = make_notebook(cc_cells())
    nb2 = make_notebook(ce_cells())

    (here / "modelisation_comptes_cheques.ipynb").write_text(
        json.dumps(nb1, indent=1, ensure_ascii=False), encoding="utf-8")
    (here / "modelisation_credits_equipement.ipynb").write_text(
        json.dumps(nb2, indent=1, ensure_ascii=False), encoding="utf-8")

    print("✅ Notebooks generated:")
    print(f"  - {here / 'modelisation_comptes_cheques.ipynb'}")
    print(f"  - {here / 'modelisation_credits_equipement.ipynb'}")


if __name__ == "__main__":
    main()
