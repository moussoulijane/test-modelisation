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
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
WF_FOLDS         = 5          # walk-forward folds (multi-fold rolling origin)
SEASONAL_PERIOD  = 5          # seasonal period (5 = weekly on business days)
ZERO_PCT_NO_LOG  = 15.0       # if % zeros above this, force USE_LOG=False
MIN_GAIN_VS_NAIVE = 5.0       # min % MAPE gain over best baseline to retain SARIMAX
COVERAGE_TARGET  = 0.95       # target empirical coverage for IC
COVERAGE_TOL     = 0.10       # tolerance on coverage deviation before recalibration
OUTLIER_MAD_INIT = 5.0        # threshold for the initial outlier scan
OUTLIER_MAD_ITER = 4.0        # threshold for the iterative outlier loop
OUTLIER_MAX_ITER = 4          # number of outlier-detection iterations
FIT_MAXITER      = 250        # SARIMAX fit iterations
RUPTURE_MIN_GAP  = 20         # minimum gap (days) between candidate breakpoints
CHOW_PVALUE_MAX  = 0.05       # max p-value to retain a candidate break (Chow test)

print("Config OK")
print(f"Target column: {{TARGET_COL!r}}")'''


HELPERS_MA = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Moroccan business calendar (fixed civil holidays + Mon-Fri)
# Islamic mobile holidays (Aïd, Mawlid, Achoura) NOT handled in this MVP.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MA_FIXED_HOLIDAYS = [
    (1, 1),    # Nouvel an
    (1, 11),   # Manifeste de l'Indépendance
    (5, 1),    # Fête du Travail
    (7, 30),   # Fête du Trône
    (8, 14),   # Oued Ed-Dahab
    (8, 20),   # Révolution du Roi et du Peuple
    (8, 21),   # Fête de la Jeunesse
    (11, 6),   # Marche Verte
    (11, 18),  # Fête de l'Indépendance
]

def ma_is_holiday(ts) -> bool:
    ts = pd.Timestamp(ts)
    return (ts.month, ts.day) in MA_FIXED_HOLIDAYS

def ma_business_days(start, end=None, periods=None) -> pd.DatetimeIndex:
    """Moroccan business days: Mon-Fri minus fixed civil holidays."""
    start = pd.Timestamp(start)
    if periods is not None:
        # Overshoot, filter, then trim to exact length
        raw = pd.bdate_range(start=start, periods=int(periods * 1.6) + 30)
        keep = [d for d in raw if not ma_is_holiday(d)]
        return pd.DatetimeIndex(keep[:periods])
    raw = pd.bdate_range(start=start, end=pd.Timestamp(end))
    return pd.DatetimeIndex([d for d in raw if not ma_is_holiday(d)])

print(f"Calendrier marocain : {len(MA_FIXED_HOLIDAYS)} fériés civils fixes/an")'''


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

# Build a complete business-day index (Moroccan calendar) and detect gaps
full_idx = ma_business_days(series.index.min(), end=series.index.max())
n_gaps = len(full_idx.difference(series.index))
n_ma_holidays_in_range = sum(1 for d in pd.bdate_range(series.index.min(), series.index.max())
                             if ma_is_holiday(d))

# Intermittent-series guard: too many zeros → disable log transform
pct_zero_check = float((series == 0).mean() * 100)
if pct_zero_check > ZERO_PCT_NO_LOG and USE_LOG:
    print(f"⚠ {pct_zero_check:.1f}% de zéros (> {ZERO_PCT_NO_LOG}%) — log-transform désactivé")
    USE_LOG = False

print(f"Period       : {series.index.min().date()} → {series.index.max().date()}")
print(f"Observations : {len(series)}")
print(f"Business-day gaps (MA calendar) : {n_gaps}")
print(f"Fériés MA dans la période       : {n_ma_holidays_in_range}")
print(f"% zéros                         : {pct_zero_check:.2f}%")
print(f"USE_LOG effectif                : {USE_LOG}")
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


MODEL_BATTERY = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model battery: SARIMAX (seasonal), ETS + statistical tests + metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fit_sarimax(y, exog=None, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0),
                maxiter=FIT_MAXITER):
    """SARIMAX fit with safe fallback to None on failure."""
    try:
        model = SARIMAX(y, exog=exog, order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        return model.fit(disp=False, maxiter=maxiter)
    except Exception as e:
        print(f"  SARIMAX error order={order} seas={seasonal_order}: {type(e).__name__}")
        return None

# Backward-compat alias (older blocks call fit_arimax)
fit_arimax = fit_sarimax

def fit_ets(y, seasonal_periods=SEASONAL_PERIOD):
    """Holt-Winters ETS with safe fallback to None."""
    try:
        use_seasonal = len(y) > 2 * seasonal_periods + 5
        m = ExponentialSmoothing(
            y, trend="add",
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
            initialization_method="estimated",
        )
        return m.fit(optimized=True)
    except Exception as e:
        print(f"  ETS error: {type(e).__name__}: {e}")
        return None

def seasonal_naive_forecast(y_orig, horizon, season=SEASONAL_PERIOD):
    """Repeat the last `season` values cyclically."""
    out = np.empty(horizon, dtype=float)
    n = len(y_orig)
    for h in range(horizon):
        idx = n - season + (h % season)
        out[h] = y_orig[max(idx, 0)]
    return out

def chow_test(y, break_idx):
    """Chow test for a level break. Returns (F, p-value)."""
    n = len(y)
    if break_idx <= 5 or break_idx >= n - 5:
        return np.nan, 1.0
    y_arr = np.asarray(y, dtype=float)
    rss_full = float(((y_arr - y_arr.mean()) ** 2).sum())
    y1, y2 = y_arr[:break_idx], y_arr[break_idx:]
    rss_split = float(((y1 - y1.mean()) ** 2).sum()
                      + ((y2 - y2.mean()) ** 2).sum())
    k = 1
    dof_num, dof_den = k, n - 2 * k
    if rss_split <= 0 or dof_den <= 0:
        return np.nan, 1.0
    F = ((rss_full - rss_split) / dof_num) / (rss_split / dof_den)
    p = 1.0 - stats.f.cdf(F, dof_num, dof_den)
    return float(F), float(p)

def pinball_loss(y_true, y_pred_q, q):
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred_q, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))

def mase(y_true, y_pred, y_train, season=SEASONAL_PERIOD):
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) > season:
        scale = float(np.mean(np.abs(y_train[season:] - y_train[:-season])))
    elif len(y_train) > 1:
        scale = float(np.mean(np.abs(np.diff(y_train))))
    else:
        scale = 1.0
    scale = max(scale, 1e-9)
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)

def empirical_coverage(y_true, lo, hi):
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def back_to_original(yhat_log, target_resid_var):
    """Log-normal bias correction when returning to the original scale."""
    if USE_LOG and (series > 0).all():
        return np.exp(yhat_log + 0.5 * target_resid_var)
    return yhat_log

print("Batterie de modèles + tests stat + métriques chargée")'''


MODEL_BATTERY_EXTRA = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Trend-aware forecasters : Theta, Holt-damped, LightGBM, SARIMAX post-rupture
# Conçus pour CAPTURER LA PENTE RÉCENTE (vs SARIMAX qui projette une droite).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    HAS_THETA = True
except Exception:
    HAS_THETA = False
    print("⚠ ThetaModel indisponible (statsmodels trop ancien)")

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    print("⚠ LightGBM indisponible — `pip install lightgbm` pour activer")

def fit_theta_forecast(y, horizon):
    """Theta method — projette explicitement la pente. Sortie : array (horizon,)."""
    if not HAS_THETA:
        return None
    try:
        use_seas = len(y) > 2 * SEASONAL_PERIOD + 5
        m = ThetaModel(y, period=SEASONAL_PERIOD if use_seas else None,
                       deseasonalize=use_seas)
        res = m.fit()
        return np.asarray(res.forecast(horizon), dtype=float)
    except Exception as e:
        print(f"  Theta error: {type(e).__name__}: {e}")
        return None

def fit_holt_damped_forecast(y, horizon, seasonal=True):
    """Holt damped — extrapole la pente récente avec amortissement."""
    try:
        use_seasonal = seasonal and (len(y) > 2 * SEASONAL_PERIOD + 5)
        m = ExponentialSmoothing(
            y, trend="add", damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=SEASONAL_PERIOD if use_seasonal else None,
            initialization_method="estimated",
        )
        res = m.fit(optimized=True)
        return np.asarray(res.forecast(horizon), dtype=float)
    except Exception as e:
        print(f"  Holt-damped error: {type(e).__name__}: {e}")
        return None

def make_lag_features(y_arr, idx_dt, lags=(1, 2, 5, 10, 20), rolls=(7, 14, 30)):
    """Lag + rolling + calendar features alignés à y_arr."""
    n = len(y_arr)
    df = pd.DataFrame(index=idx_dt)
    s = pd.Series(y_arr)
    for L in lags:
        df[f"lag_{L}"] = s.shift(L).values
    for R in rolls:
        df[f"roll_mean_{R}"] = s.rolling(R, min_periods=max(R // 2, 1)).mean().values
        df[f"roll_std_{R}"]  = s.rolling(R, min_periods=max(R // 2, 1)).std().values
    df["dow"]   = idx_dt.dayofweek
    df["day"]   = idx_dt.day
    df["month"] = idx_dt.month
    df["t_idx"] = np.arange(n) / max(n, 1)
    return df

def fit_lightgbm_recursive(y_orig, idx_dt, horizon, future_idx):
    """LightGBM avec features lag/roll/calendar, prévision récursive."""
    if not HAS_LGB:
        return None
    try:
        y_arr = np.asarray(y_orig, dtype=float)
        feats = make_lag_features(y_arr, idx_dt)
        valid = feats.notna().all(axis=1)
        X_train = feats.loc[valid].values
        y_train = y_arr[valid.values]
        if len(y_train) < 100:
            return None
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15,
            min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9,
            bagging_freq=5, random_state=SEED, verbose=-1,
        )
        model.fit(X_train, y_train)

        history = list(y_arr)
        full_idx = idx_dt.append(future_idx)
        preds = []
        for h in range(horizon):
            arr = np.concatenate([np.asarray(history, dtype=float), [np.nan]])
            partial = make_lag_features(arr, full_idx[:len(arr)])
            x_pred = partial.iloc[-1].values.reshape(1, -1)
            if np.any(np.isnan(x_pred)):
                x_pred = np.nan_to_num(x_pred, nan=float(history[-1]))
            yhat = float(model.predict(x_pred)[0])
            preds.append(yhat)
            history.append(yhat)
        return np.asarray(preds, dtype=float)
    except Exception as e:
        print(f"  LightGBM error: {type(e).__name__}: {e}")
        return None

def fit_sarimax_post_rupture(y_log, exog, break_anchor, order, seasonal_order,
                              min_post_obs=120):
    """Refit SARIMAX en restreignant le training à partir de la rupture."""
    if break_anchor is None:
        return None, None
    try:
        cut = y_log.index.get_loc(break_anchor)
        if isinstance(cut, slice):
            cut = cut.start
        cut = int(cut)
        if (len(y_log) - cut) < min_post_obs:
            return None, None
        y_sub = y_log.iloc[cut:]
        x_sub = exog.iloc[cut:]
        res = fit_sarimax(y_sub, x_sub, order=order, seasonal_order=seasonal_order)
        return res, cut
    except Exception as e:
        print(f"  SARIMAX post-rupture error: {type(e).__name__}")
        return None, None

def inverse_mape_weights(mape_dict, eps=1e-6):
    """Poids = 1/MAPE normalisés à somme 1."""
    inv = {k: 1.0 / max(v, eps) for k, v in mape_dict.items()
           if (v is not None and np.isfinite(v))}
    total = sum(inv.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in inv.items()}

print(f"Trend-aware forecasters chargés : Theta={HAS_THETA}, LightGBM={HAS_LGB}, "
      f"Holt-damped=True, SARIMAX-post-rupture=True")'''


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

# Compare pre- vs post-break level + Chow test for statistical significance
chow_F, chow_p, jump_abs, jump_rel = np.nan, np.nan, np.nan, np.nan
pre  = series.loc[:break_candidate].iloc[:-1]
post = series.loc[break_candidate:]
if len(pre) >= 30 and len(post) >= 30:
    mean_pre  = pre.tail(60).mean()
    mean_post = post.head(60).mean()
    jump_abs  = float(mean_post - mean_pre)
    jump_rel  = jump_abs / max(abs(mean_pre), 1e-9)
    target_work = np.log(series) if (USE_LOG and (series > 0).all()) else series.copy()
    try:
        break_pos = target_work.index.get_loc(break_candidate)
        if isinstance(break_pos, slice):
            break_pos = break_pos.start
        chow_F, chow_p = chow_test(target_work, int(break_pos))
    except Exception:
        chow_F, chow_p = np.nan, 1.0
    print(f"Rupture candidate : {break_candidate.date()}")
    print(f"  Niveau moyen avant (60j) : {mean_pre:,.1f}")
    print(f"  Niveau moyen après (60j) : {mean_post:,.1f}")
    print(f"  Saut absolu              : {jump_abs:+,.1f}")
    print(f"  Saut relatif             : {jump_rel:+.2%}")
    print(f"  Test de Chow             : F={chow_F:.3f}  p-value={chow_p:.4g}")
    if chow_p >= CHOW_PVALUE_MAX:
        print(f"  → Rupture non significative (p ≥ {CHOW_PVALUE_MAX}) — abandonnée")
        break_candidate = None
    else:
        print(f"  → Rupture retenue (p < {CHOW_PVALUE_MAX})")
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
# Iterative outlier loop on SARIMAX(1,1,1) residuals (uses fit_sarimax above)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
# Walk-forward order selection — multi-fold rolling origin, MÉDIAN MAPE,
# 2-phase grid: (1) non-seasonal (p,1,q), (2) seasonal variants of best order.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def walk_forward_mape(order, seasonal_order, y, exog, folds=WF_FOLDS, val_len=30):
    n_total = len(y)
    train_min = n_total - folds * val_len
    if train_min < 60:
        return np.nan, np.nan, np.nan
    mapes = []
    for k in range(folds):
        end_train = train_min + k * val_len
        y_tr = y.iloc[:end_train]
        y_va = y.iloc[end_train:end_train + val_len]
        x_tr = exog.iloc[:end_train]
        x_va = exog.iloc[end_train:end_train + val_len]
        res = fit_sarimax(y_tr, x_tr, order=order, seasonal_order=seasonal_order)
        if res is None:
            return np.nan, np.nan, np.nan
        fc = res.get_forecast(steps=len(y_va), exog=x_va)
        yhat_log = fc.predicted_mean.values
        sigma2 = float(np.var(res.resid)) if hasattr(res, "resid") else 1e-6
        yhat = back_to_original(yhat_log, sigma2)
        y_va_orig = (np.exp(y_va.values) if (USE_LOG and (series > 0).all()) else y_va.values)
        denom = np.where(np.abs(y_va_orig) < 1e-6, 1e-6, np.abs(y_va_orig))
        mapes.append(float(np.mean(np.abs(yhat - y_va_orig) / denom) * 100))
    return float(np.median(mapes)), float(np.mean(mapes)), float(np.std(mapes))

# ── Phase 1 : grille non-saisonnière ────────────────────────────────────────
phase1 = []
print("Phase 1 — grid non-saisonnier :")
for p in range(ARIMA_PMAX + 1):
    for q in range(ARIMA_QMAX + 1):
        order = (p, 1, q)
        m_med, m_mean, m_std = walk_forward_mape(order, (0, 0, 0, 0),
                                                 y_target, current_exog)
        res_full = fit_sarimax(y_target, current_exog, order=order)
        aic = res_full.aic if res_full is not None else np.nan
        phase1.append({"order": order, "seasonal": (0, 0, 0, 0),
                       "mape_med": m_med, "mape_mean": m_mean,
                       "mape_std": m_std, "aic": aic})
        print(f"  order={order}  mape_med={m_med:6.3f}%  mape_mean={m_mean:6.3f}%  "
              f"±{m_std:5.3f}  aic={aic:10.2f}")

phase1_df = (pd.DataFrame(phase1).dropna(subset=["mape_med"])
                                  .sort_values("mape_med")
                                  .reset_index(drop=True))
if phase1_df.empty:
    raise RuntimeError("Aucun fit non-saisonnier n'a convergé.")
best_nonseas_order = phase1_df.iloc[0]["order"]
print(f"\\nMeilleur non-saisonnier : ARIMA{best_nonseas_order} "
      f"(mape_med={phase1_df.iloc[0]['mape_med']:.3f}%)")

# ── Phase 2 : variantes saisonnières du meilleur ordre ──────────────────────
print(f"\\nPhase 2 — variantes saisonnières (m={SEASONAL_PERIOD}) :")
seasonal_variants = [(1, 0, 0, SEASONAL_PERIOD),
                     (0, 0, 1, SEASONAL_PERIOD),
                     (1, 0, 1, SEASONAL_PERIOD)]
phase2 = []
for sord in seasonal_variants:
    m_med, m_mean, m_std = walk_forward_mape(best_nonseas_order, sord,
                                             y_target, current_exog)
    res_full = fit_sarimax(y_target, current_exog,
                           order=best_nonseas_order, seasonal_order=sord)
    aic = res_full.aic if res_full is not None else np.nan
    phase2.append({"order": best_nonseas_order, "seasonal": sord,
                   "mape_med": m_med, "mape_mean": m_mean,
                   "mape_std": m_std, "aic": aic})
    print(f"  order={best_nonseas_order} seas={sord}  "
          f"mape_med={m_med:6.3f}%  mape_mean={m_mean:6.3f}%  aic={aic:10.2f}")

grid_df = (pd.concat([phase1_df, pd.DataFrame(phase2)], ignore_index=True)
             .dropna(subset=["mape_med"])
             .sort_values("mape_med")
             .reset_index(drop=True))

print("━" * 80)
print("Top 8 toutes phases (par MAPE médian walk-forward) :")
print(grid_df.head(8).to_string(index=False))

best_row = grid_df.iloc[0]
best_order = best_row["order"]
best_seasonal = best_row["seasonal"] if isinstance(best_row["seasonal"], tuple) else (0, 0, 0, 0)
print(f"\\nOrdre retenu : ARIMA{best_order} × SARIMA{best_seasonal}  "
      f"(mape_med={best_row['mape_med']:.3f}%)")'''


FIT_DIAG = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final fit + residual diagnostics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

final_res = fit_sarimax(y_target, current_exog, order=best_order,
                        seasonal_order=best_seasonal, maxiter=FIT_MAXITER + 100)
if final_res is None:
    raise RuntimeError(f"Échec du fit final ARIMA{best_order} × SARIMA{best_seasonal}")
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
# Hold-out multi-modèles + ENSEMBLE pondéré par 1/MAPE
# Candidats : SARIMAX (full + post-rupture), ETS, Theta, Holt-damped, LightGBM
# Baselines : naïf, RWD, naïf saisonnier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_end = len(series) - HOLDOUT_LEN
y_train_log    = y_target.iloc[:train_end]
y_test_log     = y_target.iloc[train_end:]
exog_train_sub = current_exog.iloc[:train_end]
exog_test_sub  = current_exog.iloc[train_end:]
y_train_orig   = series.iloc[:train_end].values
y_test_orig    = (np.exp(y_test_log.values)
                  if (USE_LOG and (series > 0).all()) else y_test_log.values)
train_idx_dt   = series.index[:train_end]
test_idx_dt    = series.index[train_end:]

candidates = {}  # name → yhat_array (HOLDOUT_LEN,) en échelle originale

# ── (1) SARIMAX (full history, log) ─────────────────────────────────────────
yhat_log_test = None
se_log_test = None
sigma2_log_train = np.nan
res_train = fit_sarimax(y_train_log, exog_train_sub, order=best_order,
                        seasonal_order=best_seasonal, maxiter=FIT_MAXITER + 100)
if res_train is not None:
    fc = res_train.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)
    sigma2_log_train = float(np.var(res_train.resid))
    yhat_log_test = fc.predicted_mean.values
    se_log_test   = fc.se_mean.values
    candidates["SARIMAX_full"] = back_to_original(yhat_log_test, sigma2_log_train)

# ── (2) SARIMAX post-rupture (training tronqué) ─────────────────────────────
res_postr, cut_pos = fit_sarimax_post_rupture(y_train_log, exog_train_sub,
                                              break_candidate, best_order, best_seasonal)
if res_postr is not None:
    fc_pr = res_postr.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)
    sigma2_pr = float(np.var(res_postr.resid))
    candidates["SARIMAX_post_rupture"] = back_to_original(fc_pr.predicted_mean.values, sigma2_pr)
    print(f"SARIMAX post-rupture : training restreint à {len(y_train_log) - cut_pos} obs (post-{break_candidate.date()})")

# ── (3) ETS ─────────────────────────────────────────────────────────────────
ets_res = fit_ets(pd.Series(y_train_orig, index=train_idx_dt),
                  seasonal_periods=SEASONAL_PERIOD)
if ets_res is not None:
    candidates["ETS"] = np.asarray(ets_res.forecast(HOLDOUT_LEN), dtype=float)

# ── (4) Theta — extrapolation explicite de la pente ─────────────────────────
yhat_theta = fit_theta_forecast(pd.Series(y_train_orig, index=train_idx_dt), HOLDOUT_LEN)
if yhat_theta is not None:
    candidates["Theta"] = yhat_theta

# ── (5) Holt damped trend ───────────────────────────────────────────────────
yhat_holt = fit_holt_damped_forecast(pd.Series(y_train_orig, index=train_idx_dt), HOLDOUT_LEN)
if yhat_holt is not None:
    candidates["Holt_damped"] = yhat_holt

# ── (6) LightGBM avec features lag/roll/calendar ────────────────────────────
yhat_lgb = fit_lightgbm_recursive(y_train_orig, train_idx_dt, HOLDOUT_LEN, test_idx_dt)
if yhat_lgb is not None:
    candidates["LightGBM"] = yhat_lgb

# ── Baselines (référence uniquement, hors ensemble) ─────────────────────────
baselines = {
    "Naive_last":    np.full(HOLDOUT_LEN, y_train_orig[-1], dtype=float),
    "RWD":           y_train_orig[-1] + float(np.mean(np.diff(y_train_orig))) * np.arange(1, HOLDOUT_LEN + 1),
    f"Naive_m{SEASONAL_PERIOD}": seasonal_naive_forecast(y_train_orig, HOLDOUT_LEN, SEASONAL_PERIOD),
}

# ── Métriques ────────────────────────────────────────────────────────────────
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.where(np.abs(b) < 1e-6, 1e-6, np.abs(b))
    return float(np.mean(np.abs(a - b) / denom) * 100)

cand_mapes = {n: mape(yhat_, y_test_orig) for n, yhat_ in candidates.items()}
base_mapes = {n: mape(yhat_, y_test_orig) for n, yhat_ in baselines.items()}

# ── ENSEMBLE pondéré par 1/MAPE (uniquement sur les candidats, pas les baselines) ──
weights = inverse_mape_weights(cand_mapes)
ensemble_pred = None
if weights:
    ensemble_pred = np.zeros(HOLDOUT_LEN, dtype=float)
    for name, w in weights.items():
        ensemble_pred += w * candidates[name]
    candidates["ENSEMBLE_inv_MAPE"] = ensemble_pred
    cand_mapes["ENSEMBLE_inv_MAPE"] = mape(ensemble_pred, y_test_orig)

# ── Rapport global ──────────────────────────────────────────────────────────
rows = []
for name, yhat_ in candidates.items():
    rows.append({"model": name, "MAE": mae(yhat_, y_test_orig),
                 "RMSE": rmse(yhat_, y_test_orig), "MAPE": cand_mapes[name],
                 "MASE": mase(y_test_orig, yhat_, y_train_orig, season=SEASONAL_PERIOD)})
for name, yhat_ in baselines.items():
    rows.append({"model": "[base] " + name, "MAE": mae(yhat_, y_test_orig),
                 "RMSE": rmse(yhat_, y_test_orig), "MAPE": base_mapes[name],
                 "MASE": mase(y_test_orig, yhat_, y_train_orig, season=SEASONAL_PERIOD)})

metrics_df = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
print("━" * 90)
print(metrics_df.to_string(index=False))
print("━" * 90)
print("Poids ensemble (inverse-MAPE) :")
for n, w in weights.items():
    print(f"  {n:30s}  weight={w:.3f}  mape={cand_mapes[n]:6.3f}%")

# ── Choix final : ENSEMBLE par défaut, fallback meilleur individuel ─────────
if "ENSEMBLE_inv_MAPE" in candidates and cand_mapes["ENSEMBLE_inv_MAPE"] < np.inf:
    model_final_label = "ENSEMBLE_inv_MAPE"
    yhat = candidates["ENSEMBLE_inv_MAPE"]
else:
    model_final_label = min(cand_mapes, key=cand_mapes.get) if cand_mapes else "Naive_last"
    yhat = candidates.get(model_final_label, baselines["Naive_last"])

mape_model = mape(yhat, y_test_orig)
mae_model  = mae(yhat, y_test_orig)
rmse_model = rmse(yhat, y_test_orig)
mase_model = mase(y_test_orig, yhat, y_train_orig, season=SEASONAL_PERIOD)
mape_naive  = base_mapes["Naive_last"]
mape_rwd    = base_mapes["RWD"]
mape_seasnv = base_mapes[f"Naive_m{SEASONAL_PERIOD}"]
gain_naive  = (mape_naive  - mape_model) / max(mape_naive, 1e-6) * 100
gain_rwd    = (mape_rwd    - mape_model) / max(mape_rwd, 1e-6) * 100
gain_seasnv = (mape_seasnv - mape_model) / max(mape_seasnv, 1e-6) * 100

# Pour la projection : aligner sur le modèle retenu
use_sarimax_for_projection = model_final_label in ("SARIMAX_full", "SARIMAX_post_rupture")

print(f"\\nModèle retenu : {model_final_label}")
print(f"  MAPE = {mape_model:.2f}%   MASE = {mase_model:.3f}")
print(f"  vs naïf : {gain_naive:+.2f}%  vs RWD : {gain_rwd:+.2f}%  "
      f"vs saisonnier : {gain_seasnv:+.2f}%")

# ── Calibration empirique des IC sur le hold-out ─────────────────────────────
coverage    = np.nan
calib_factor = 1.0
pinball_05 = pinball_50 = pinball_95 = np.nan
if yhat_log_test is not None and se_log_test is not None:
    resid_full = final_res.resid.dropna()
    resid_std  = (resid_full - resid_full.mean()) / max(resid_full.std(), 1e-9)
    q_lo = float(np.quantile(resid_std.values,       ALPHA / 2))
    q_hi = float(np.quantile(resid_std.values, 1.0 - ALPHA / 2))
    lo_test = back_to_original(yhat_log_test + q_lo * se_log_test, sigma2_log_train)
    hi_test = back_to_original(yhat_log_test + q_hi * se_log_test, sigma2_log_train)
    coverage = empirical_coverage(y_test_orig, lo_test, hi_test)
    pinball_05 = pinball_loss(y_test_orig, lo_test, 0.025)
    pinball_50 = pinball_loss(y_test_orig, yhat,    0.500)
    pinball_95 = pinball_loss(y_test_orig, hi_test, 0.975)
    cov_dev = coverage - COVERAGE_TARGET
    if abs(cov_dev) > COVERAGE_TOL:
        # Recalibre via les quantiles des résidus standardisés sur hold-out
        z_test = (y_test_log.values - yhat_log_test) / np.maximum(se_log_test, 1e-9)
        z_test = z_test[np.isfinite(z_test)]
        if len(z_test) >= 5:
            q_lo_r = float(np.quantile(z_test,       ALPHA / 2))
            q_hi_r = float(np.quantile(z_test, 1.0 - ALPHA / 2))
            calib_factor = (q_hi_r - q_lo_r) / max(q_hi - q_lo, 1e-9)
            calib_factor = float(np.clip(calib_factor, 0.5, 3.0))
        print(f"⚠ Couverture IC95 = {coverage:.1%} (cible {COVERAGE_TARGET:.0%}) — "
              f"facteur de recalibration = {calib_factor:.3f}")
    else:
        print(f"✓ Couverture IC95 = {coverage:.1%} (cible {COVERAGE_TARGET:.0%})")
    print(f"  Pinball q=0.025 : {pinball_05:>14,.2f}")
    print(f"  Pinball q=0.500 : {pinball_50:>14,.2f}")
    print(f"  Pinball q=0.975 : {pinball_95:>14,.2f}")'''


PROJECTION_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Projection future — refit TOUS les candidats sur la série complète,
# applique les MÊMES poids ensemble qu'au hold-out.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

future_idx = ma_business_days(series.index[-1] + pd.Timedelta(days=1),
                              periods=FORECAST_HORIZON)

reliability = np.array(["haute"] * FORECAST_HORIZON, dtype=object)
reliability[30:60] = "moyenne"
reliability[60:]   = "faible"

# Exog future aligné sur current_exog
exog_future = build_exog(future_idx, break_candidate, [],
                         include_dow=True, include_eom=True)
for col in current_exog.columns:
    if col not in exog_future.columns:
        exog_future[col] = 0.0
exog_future = exog_future[current_exog.columns]

future_candidates = {}  # name → array (FORECAST_HORIZON,)
sigma2_log_full = float(np.var(resid))
yhat_log_future_full = None
se_log_future_full = None

# (1) SARIMAX full (final_res est déjà fitté sur tout l'échantillon)
if "SARIMAX_full" in candidates:
    fc_f = final_res.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
    yhat_log_future_full = fc_f.predicted_mean.values
    se_log_future_full   = fc_f.se_mean.values * calib_factor
    future_candidates["SARIMAX_full"] = back_to_original(yhat_log_future_full, sigma2_log_full)

# (2) SARIMAX post-rupture refit sur full
if "SARIMAX_post_rupture" in candidates and break_candidate is not None:
    res_pr_full, _ = fit_sarimax_post_rupture(y_target, current_exog, break_candidate,
                                               best_order, best_seasonal)
    if res_pr_full is not None:
        fc_pr = res_pr_full.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
        sigma2_pr = float(np.var(res_pr_full.resid))
        future_candidates["SARIMAX_post_rupture"] = back_to_original(fc_pr.predicted_mean.values, sigma2_pr)

# (3) ETS refit sur full
if "ETS" in candidates:
    ets_full = fit_ets(series, seasonal_periods=SEASONAL_PERIOD)
    if ets_full is not None:
        future_candidates["ETS"] = np.asarray(ets_full.forecast(FORECAST_HORIZON), dtype=float)

# (4) Theta refit sur full
if "Theta" in candidates:
    yt = fit_theta_forecast(series, FORECAST_HORIZON)
    if yt is not None:
        future_candidates["Theta"] = yt

# (5) Holt damped refit sur full
if "Holt_damped" in candidates:
    yh = fit_holt_damped_forecast(series, FORECAST_HORIZON)
    if yh is not None:
        future_candidates["Holt_damped"] = yh

# (6) LightGBM refit sur full
if "LightGBM" in candidates:
    yl = fit_lightgbm_recursive(series.values, series.index, FORECAST_HORIZON, future_idx)
    if yl is not None:
        future_candidates["LightGBM"] = yl

# Construction de l'ensemble future avec les MÊMES poids qu'au hold-out
ensemble_future = np.zeros(FORECAST_HORIZON, dtype=float)
total_w = 0.0
for name, w in weights.items():
    if name in future_candidates:
        ensemble_future += w * future_candidates[name]
        total_w += w
if total_w > 0:
    ensemble_future /= total_w
    future_candidates["ENSEMBLE_inv_MAPE"] = ensemble_future

# Sélection finale = même label que hold-out (en général ENSEMBLE_inv_MAPE)
if model_final_label in future_candidates:
    yhat_future = future_candidates[model_final_label]
    proj_engine = model_final_label
elif "ENSEMBLE_inv_MAPE" in future_candidates:
    yhat_future = future_candidates["ENSEMBLE_inv_MAPE"]
    proj_engine = "ENSEMBLE_inv_MAPE (fallback)"
elif future_candidates:
    proj_engine, yhat_future = next(iter(future_candidates.items()))
else:
    yhat_future = seasonal_naive_forecast(series.values, FORECAST_HORIZON,
                                          season=SEASONAL_PERIOD)
    proj_engine = "Naïf saisonnier (fallback)"

# IC : préférence SARIMAX (états + résidus), sinon empirique sur résidus hold-out
if yhat_log_future_full is not None and se_log_future_full is not None:
    resid_std = (resid - resid.mean()) / max(resid.std(), 1e-9)
    q_lo = float(np.quantile(resid_std.values,       ALPHA / 2))
    q_hi = float(np.quantile(resid_std.values, 1.0 - ALPHA / 2))
    lo_log = yhat_log_future_full + q_lo * se_log_future_full
    hi_log = yhat_log_future_full + q_hi * se_log_future_full
    lo_future = back_to_original(lo_log, sigma2_log_full)
    hi_future = back_to_original(hi_log, sigma2_log_full)
    # Centrage des IC sur la prévision retenue (peut être ensemble)
    spread_lo = lo_future - future_candidates["SARIMAX_full"]
    spread_hi = hi_future - future_candidates["SARIMAX_full"]
    lo_future = yhat_future + spread_lo
    hi_future = yhat_future + spread_hi
else:
    # IC empirique via erreurs hold-out de l'ensemble
    ens_holdout_err = y_test_orig - candidates.get("ENSEMBLE_inv_MAPE", yhat)
    ens_holdout_err = ens_holdout_err[np.isfinite(ens_holdout_err)]
    q_lo_e = (float(np.quantile(ens_holdout_err, ALPHA / 2)) if len(ens_holdout_err)
              else -1.96 * series.std())
    q_hi_e = (float(np.quantile(ens_holdout_err, 1.0 - ALPHA / 2)) if len(ens_holdout_err)
              else 1.96 * series.std())
    h_arr = np.sqrt(np.arange(1, FORECAST_HORIZON + 1))
    lo_future = yhat_future + q_lo_e * h_arr
    hi_future = yhat_future + q_hi_e * h_arr

projection = pd.DataFrame({
    "date"        : future_idx,
    "previs"      : yhat_future,
    "ic_bas"      : lo_future,
    "ic_haut"     : hi_future,
    "fiabilite"   : reliability,
    "day_of_week" : future_idx.day_name(),
    "moteur"      : proj_engine,
})
print(f"Moteur de projection : {proj_engine}")
print("Poids ensemble appliqués sur la projection :")
for n, w in weights.items():
    if n in future_candidates:
        print(f"  {n:30s}  w={w:.3f}")
print(projection.head(15).to_string(index=False))
print(f"\\nProjection sur {FORECAST_HORIZON} jours ouvrés MA "
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
             f"{model_final_label}  |  MAPE test = {mape_model:.2f}%  |  "
             f"Gain vs naïf = {gain_naive:+.1f}%")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()

# Save PNG to out/
_safe = (TARGET_COL.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\\\", "_")
                   .replace("'", ""))
png_path = OUTPUT_DIR / f"projection_{_safe}.png"
plt.savefig(png_path, dpi=120, bbox_inches="tight")
print(f"✅ PNG sauvegardé : {png_path.resolve()}")
plt.show()'''


EXPORT_CC = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Excel export — 3 onglets : projection / diagnostics / breakpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from datetime import datetime
stamp = datetime.now().strftime("%Y%m%d")
safe_name = (TARGET_COL.replace(" ", "_")
                       .replace("/", "_")
                       .replace("\\\\", "_")
                       .replace("'", ""))
out_path = OUTPUT_DIR / f"projection_robuste_{safe_name}_{stamp}.xlsx"

diagnostics = pd.DataFrame({
    "metric": [
        "model_final", "MAPE_test_pct", "MAE_test", "RMSE_test", "MASE_test",
        "MAPE_naive_pct", "MAPE_rwd_pct", "MAPE_seasonal_naive_pct",
        "Gain_vs_naive_pct", "Gain_vs_rwd_pct", "Gain_vs_seasnv_pct",
        "Pinball_q025", "Pinball_q500", "Pinball_q975",
        "Coverage_IC95_pct", "Calibration_factor",
        "ARIMA_order", "Seasonal_order", "USE_LOG", "horizon_days",
        "n_obs_total", "n_obs_train", "n_pulses", "break_date",
        "chow_F", "chow_pvalue",
    ],
    "value":  [
        model_final_label, mape_model, mae_model, rmse_model, mase_model,
        mape_naive, mape_rwd, mape_seasnv,
        gain_naive, gain_rwd, gain_seasnv,
        pinball_05, pinball_50, pinball_95,
        coverage * 100 if (coverage is not None and not np.isnan(coverage)) else np.nan,
        calib_factor,
        str(best_order), str(best_seasonal), USE_LOG, FORECAST_HORIZON,
        len(series), train_end, len(pulse_dates) + len(extra_pulses),
        str(break_candidate.date()) if break_candidate is not None else "",
        chow_F, chow_p,
    ],
})

breakpoints_df = pd.DataFrame(columns=["date", "chow_F", "chow_pvalue",
                                       "jump_abs", "jump_rel"])
if break_candidate is not None:
    breakpoints_df = pd.DataFrame([{
        "date": break_candidate.date(),
        "chow_F": chow_F,
        "chow_pvalue": chow_p,
        "jump_abs": jump_abs,
        "jump_rel": jump_rel,
    }])

candidates_df = pd.DataFrame([
    {"model": name,
     "holdout_MAPE_pct": cand_mapes.get(name, np.nan),
     "weight": weights.get(name, 0.0)}
    for name in candidates
])
candidates_df = candidates_df.sort_values("holdout_MAPE_pct").reset_index(drop=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    projection.to_excel(writer, sheet_name="projection", index=False)
    diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)
    breakpoints_df.to_excel(writer, sheet_name="breakpoints", index=False)
    candidates_df.to_excel(writer, sheet_name="candidates", index=False)

print(f"OK Projection exportee : {out_path.resolve()}")'''


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

break_candidates = cusum_breakpoints(work, max_breaks=3)
print(f"Ruptures candidates (CUSUM segmenté) : {len(break_candidates)}")

# Filter break candidates by Chow test (keep only p < CHOW_PVALUE_MAX)
target_work = np.log(series) if (USE_LOG and (series > 0).all()) else series.copy()
break_dates = []
break_diagnostics = []
for d in break_candidates:
    try:
        pos = target_work.index.get_loc(d)
        if isinstance(pos, slice):
            pos = pos.start
        F, p = chow_test(target_work, int(pos))
    except Exception:
        F, p = np.nan, 1.0
    pre  = series.loc[:d].tail(40)
    post = series.loc[d:].head(40)
    jabs = float(post.mean() - pre.mean()) if (len(pre) >= 10 and len(post) >= 10) else np.nan
    keep = (p < CHOW_PVALUE_MAX)
    print(f"  {d.date()}  Chow F={F:.3f}  p={p:.4g}  jump={jabs:+,.1f}  → "
          f"{'retenue' if keep else 'rejetée'}")
    if keep:
        break_dates.append(d)
        break_diagnostics.append({"date": d, "chow_F": F, "chow_p": p, "jump_abs": jabs})

print(f"\\nRuptures retenues après Chow (p<{CHOW_PVALUE_MAX}) : {len(break_dates)}")

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
# Hold-out multi-modèles + ENSEMBLE pondéré par 1/MAPE
# Candidats : SARIMAX (full + post-rupture), ETS, Theta, Holt-damped, LightGBM
# Baselines : naïf, quarterly mean, naïf saisonnier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_end = len(series) - HOLDOUT_LEN
y_train_log    = y_target.iloc[:train_end]
y_test_log     = y_target.iloc[train_end:]
exog_train_sub = current_exog.iloc[:train_end]
exog_test_sub  = current_exog.iloc[train_end:]
y_train_orig   = series.iloc[:train_end].values
y_test_orig    = (np.exp(y_test_log.values)
                  if (USE_LOG and (series > 0).all()) else y_test_log.values)
train_idx_dt   = series.index[:train_end]
test_idx_dt    = series.index[train_end:]

candidates = {}  # name → yhat_array (HOLDOUT_LEN,) en échelle originale

# ── (1) SARIMAX full ────────────────────────────────────────────────────────
yhat_log_test = None
se_log_test = None
sigma2_log_train = np.nan
res_train = fit_sarimax(y_train_log, exog_train_sub, order=best_order,
                        seasonal_order=best_seasonal, maxiter=FIT_MAXITER + 100)
if res_train is not None:
    fc = res_train.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)
    sigma2_log_train = float(np.var(res_train.resid))
    yhat_log_test = fc.predicted_mean.values
    se_log_test   = fc.se_mean.values
    candidates["SARIMAX_full"] = back_to_original(yhat_log_test, sigma2_log_train)

# ── (2) SARIMAX post-rupture (utilise la dernière rupture retenue) ─────────
last_break = break_dates[-1] if break_dates else None
res_postr, cut_pos = fit_sarimax_post_rupture(y_train_log, exog_train_sub,
                                              last_break, best_order, best_seasonal)
if res_postr is not None:
    fc_pr = res_postr.get_forecast(steps=HOLDOUT_LEN, exog=exog_test_sub)
    sigma2_pr = float(np.var(res_postr.resid))
    candidates["SARIMAX_post_rupture"] = back_to_original(fc_pr.predicted_mean.values, sigma2_pr)
    print(f"SARIMAX post-rupture : training restreint à {len(y_train_log) - cut_pos} obs (post-{last_break.date()})")

# ── (3) ETS ─────────────────────────────────────────────────────────────────
ets_res = fit_ets(pd.Series(y_train_orig, index=train_idx_dt),
                  seasonal_periods=SEASONAL_PERIOD)
if ets_res is not None:
    candidates["ETS"] = np.asarray(ets_res.forecast(HOLDOUT_LEN), dtype=float)

# ── (4) Theta ───────────────────────────────────────────────────────────────
yhat_theta = fit_theta_forecast(pd.Series(y_train_orig, index=train_idx_dt), HOLDOUT_LEN)
if yhat_theta is not None:
    candidates["Theta"] = yhat_theta

# ── (5) Holt damped ─────────────────────────────────────────────────────────
yhat_holt = fit_holt_damped_forecast(pd.Series(y_train_orig, index=train_idx_dt), HOLDOUT_LEN)
if yhat_holt is not None:
    candidates["Holt_damped"] = yhat_holt

# ── (6) LightGBM ────────────────────────────────────────────────────────────
yhat_lgb = fit_lightgbm_recursive(y_train_orig, train_idx_dt, HOLDOUT_LEN, test_idx_dt)
if yhat_lgb is not None:
    candidates["LightGBM"] = yhat_lgb

# ── Baselines ───────────────────────────────────────────────────────────────
train_q_means = pd.Series(y_train_orig, index=train_idx_dt).groupby(train_idx_dt.quarter).mean()
yhat_qmean = np.array([train_q_means.get(d.quarter, y_train_orig[-1]) for d in test_idx_dt],
                      dtype=float)
baselines = {
    "Naive_last":    np.full(HOLDOUT_LEN, y_train_orig[-1], dtype=float),
    "Quarterly_mean": yhat_qmean,
    f"Naive_m{SEASONAL_PERIOD}": seasonal_naive_forecast(y_train_orig, HOLDOUT_LEN, SEASONAL_PERIOD),
}

# ── Métriques ────────────────────────────────────────────────────────────────
def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
def mape(a, b):
    a, b = np.asarray(a), np.asarray(b)
    denom = np.where(np.abs(b) < 1e-6, 1e-6, np.abs(b))
    return float(np.mean(np.abs(a - b) / denom) * 100)

cand_mapes = {n: mape(yhat_, y_test_orig) for n, yhat_ in candidates.items()}
base_mapes = {n: mape(yhat_, y_test_orig) for n, yhat_ in baselines.items()}

# ── ENSEMBLE inverse-MAPE ───────────────────────────────────────────────────
weights = inverse_mape_weights(cand_mapes)
ensemble_pred = None
if weights:
    ensemble_pred = np.zeros(HOLDOUT_LEN, dtype=float)
    for name, w in weights.items():
        ensemble_pred += w * candidates[name]
    candidates["ENSEMBLE_inv_MAPE"] = ensemble_pred
    cand_mapes["ENSEMBLE_inv_MAPE"] = mape(ensemble_pred, y_test_orig)

# ── Rapport ─────────────────────────────────────────────────────────────────
rows = []
for name, yhat_ in candidates.items():
    rows.append({"model": name, "MAE": mae(yhat_, y_test_orig),
                 "RMSE": rmse(yhat_, y_test_orig), "MAPE": cand_mapes[name],
                 "MASE": mase(y_test_orig, yhat_, y_train_orig, season=SEASONAL_PERIOD)})
for name, yhat_ in baselines.items():
    rows.append({"model": "[base] " + name, "MAE": mae(yhat_, y_test_orig),
                 "RMSE": rmse(yhat_, y_test_orig), "MAPE": base_mapes[name],
                 "MASE": mase(y_test_orig, yhat_, y_train_orig, season=SEASONAL_PERIOD)})

metrics_df = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
print("━" * 90)
print(metrics_df.to_string(index=False))
print("━" * 90)
print("Poids ensemble (inverse-MAPE) :")
for n, w in weights.items():
    print(f"  {n:30s}  weight={w:.3f}  mape={cand_mapes[n]:6.3f}%")

# ── Choix final = ENSEMBLE par défaut ───────────────────────────────────────
if "ENSEMBLE_inv_MAPE" in candidates and cand_mapes["ENSEMBLE_inv_MAPE"] < np.inf:
    model_final_label = "ENSEMBLE_inv_MAPE"
    yhat = candidates["ENSEMBLE_inv_MAPE"]
else:
    model_final_label = min(cand_mapes, key=cand_mapes.get) if cand_mapes else "Naive_last"
    yhat = candidates.get(model_final_label, baselines["Naive_last"])

mape_model = mape(yhat, y_test_orig)
mae_model  = mae(yhat, y_test_orig)
rmse_model = rmse(yhat, y_test_orig)
mase_model = mase(y_test_orig, yhat, y_train_orig, season=SEASONAL_PERIOD)
mape_naive  = base_mapes["Naive_last"]
mape_qm     = base_mapes["Quarterly_mean"]
mape_seasnv = base_mapes[f"Naive_m{SEASONAL_PERIOD}"]
gain_naive  = (mape_naive  - mape_model) / max(mape_naive, 1e-6) * 100
gain_qm     = (mape_qm     - mape_model) / max(mape_qm, 1e-6) * 100
gain_seasnv = (mape_seasnv - mape_model) / max(mape_seasnv, 1e-6) * 100

use_sarimax_for_projection = model_final_label in ("SARIMAX_full", "SARIMAX_post_rupture")

print(f"\\nModèle retenu : {model_final_label}")
print(f"  MAPE = {mape_model:.2f}%   MASE = {mase_model:.3f}")
print(f"  vs naïf : {gain_naive:+.2f}%  vs quarterly_mean : {gain_qm:+.2f}%  "
      f"vs saisonnier : {gain_seasnv:+.2f}%")

# ── Calibration empirique des IC sur le hold-out ─────────────────────────────
coverage    = np.nan
calib_factor = 1.0
pinball_05 = pinball_50 = pinball_95 = np.nan
if yhat_log_test is not None and se_log_test is not None:
    resid_full = final_res.resid.dropna()
    resid_std  = (resid_full - resid_full.mean()) / max(resid_full.std(), 1e-9)
    q_lo = float(np.quantile(resid_std.values,       ALPHA / 2))
    q_hi = float(np.quantile(resid_std.values, 1.0 - ALPHA / 2))
    lo_test = back_to_original(yhat_log_test + q_lo * se_log_test, sigma2_log_train)
    hi_test = back_to_original(yhat_log_test + q_hi * se_log_test, sigma2_log_train)
    coverage = empirical_coverage(y_test_orig, lo_test, hi_test)
    pinball_05 = pinball_loss(y_test_orig, lo_test, 0.025)
    pinball_50 = pinball_loss(y_test_orig, yhat,    0.500)
    pinball_95 = pinball_loss(y_test_orig, hi_test, 0.975)
    cov_dev = coverage - COVERAGE_TARGET
    if abs(cov_dev) > COVERAGE_TOL:
        z_test = (y_test_log.values - yhat_log_test) / np.maximum(se_log_test, 1e-9)
        z_test = z_test[np.isfinite(z_test)]
        if len(z_test) >= 5:
            q_lo_r = float(np.quantile(z_test,       ALPHA / 2))
            q_hi_r = float(np.quantile(z_test, 1.0 - ALPHA / 2))
            calib_factor = (q_hi_r - q_lo_r) / max(q_hi - q_lo, 1e-9)
            calib_factor = float(np.clip(calib_factor, 0.5, 3.0))
        print(f"⚠ Couverture IC95 = {coverage:.1%} (cible {COVERAGE_TARGET:.0%}) — "
              f"facteur de recalibration = {calib_factor:.3f}")
    else:
        print(f"✓ Couverture IC95 = {coverage:.1%} (cible {COVERAGE_TARGET:.0%})")
    print(f"  Pinball q=0.025 : {pinball_05:>14,.2f}")
    print(f"  Pinball q=0.500 : {pinball_50:>14,.2f}")
    print(f"  Pinball q=0.975 : {pinball_95:>14,.2f}")'''


PROJECTION_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Projection future — refit TOUS les candidats sur la série complète,
# applique les MÊMES poids ensemble qu'au hold-out.
# Fiabilité fortement dégressive : série lumpy, > 30 j indicatif.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

future_idx = ma_business_days(series.index[-1] + pd.Timedelta(days=1),
                              periods=FORECAST_HORIZON)

reliability = np.array(["haute"] * FORECAST_HORIZON, dtype=object)
reliability[15:30] = "moyenne"
reliability[30:]   = "faible"

include_dow_flag = any(c.startswith("dow_") for c in current_exog.columns)
exog_future = build_exog(future_idx, break_dates, [],
                         include_dow=include_dow_flag,
                         include_eoq=True, include_august=True)
for col in current_exog.columns:
    if col not in exog_future.columns:
        exog_future[col] = 0.0
exog_future = exog_future[current_exog.columns]

future_candidates = {}
sigma2_log_full = float(np.var(resid))
yhat_log_future_full = None
se_log_future_full = None

# (1) SARIMAX full
if "SARIMAX_full" in candidates:
    fc_f = final_res.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
    yhat_log_future_full = fc_f.predicted_mean.values
    se_log_future_full   = fc_f.se_mean.values * calib_factor
    future_candidates["SARIMAX_full"] = back_to_original(yhat_log_future_full, sigma2_log_full)

# (2) SARIMAX post-rupture (utilise la dernière rupture retenue)
if "SARIMAX_post_rupture" in candidates and break_dates:
    res_pr_full, _ = fit_sarimax_post_rupture(y_target, current_exog, break_dates[-1],
                                               best_order, best_seasonal)
    if res_pr_full is not None:
        fc_pr = res_pr_full.get_forecast(steps=FORECAST_HORIZON, exog=exog_future)
        sigma2_pr = float(np.var(res_pr_full.resid))
        future_candidates["SARIMAX_post_rupture"] = back_to_original(fc_pr.predicted_mean.values, sigma2_pr)

# (3) ETS
if "ETS" in candidates:
    ets_full = fit_ets(series, seasonal_periods=SEASONAL_PERIOD)
    if ets_full is not None:
        future_candidates["ETS"] = np.asarray(ets_full.forecast(FORECAST_HORIZON), dtype=float)

# (4) Theta
if "Theta" in candidates:
    yt = fit_theta_forecast(series, FORECAST_HORIZON)
    if yt is not None:
        future_candidates["Theta"] = yt

# (5) Holt damped
if "Holt_damped" in candidates:
    yh = fit_holt_damped_forecast(series, FORECAST_HORIZON)
    if yh is not None:
        future_candidates["Holt_damped"] = yh

# (6) LightGBM
if "LightGBM" in candidates:
    yl = fit_lightgbm_recursive(series.values, series.index, FORECAST_HORIZON, future_idx)
    if yl is not None:
        future_candidates["LightGBM"] = yl

# Ensemble future = poids hold-out
ensemble_future = np.zeros(FORECAST_HORIZON, dtype=float)
total_w = 0.0
for name, w in weights.items():
    if name in future_candidates:
        ensemble_future += w * future_candidates[name]
        total_w += w
if total_w > 0:
    ensemble_future /= total_w
    future_candidates["ENSEMBLE_inv_MAPE"] = ensemble_future

# Choix du moteur
if model_final_label in future_candidates:
    yhat_future = future_candidates[model_final_label]
    proj_engine = model_final_label
elif "ENSEMBLE_inv_MAPE" in future_candidates:
    yhat_future = future_candidates["ENSEMBLE_inv_MAPE"]
    proj_engine = "ENSEMBLE_inv_MAPE (fallback)"
elif future_candidates:
    proj_engine, yhat_future = next(iter(future_candidates.items()))
else:
    yhat_future = seasonal_naive_forecast(series.values, FORECAST_HORIZON,
                                          season=SEASONAL_PERIOD)
    proj_engine = "Naïf saisonnier (fallback)"

# IC : SARIMAX si dispo, sinon empirique
if yhat_log_future_full is not None and se_log_future_full is not None:
    resid_std = (resid - resid.mean()) / max(resid.std(), 1e-9)
    q_lo = float(np.quantile(resid_std.values,       ALPHA / 2))
    q_hi = float(np.quantile(resid_std.values, 1.0 - ALPHA / 2))
    lo_log = yhat_log_future_full + q_lo * se_log_future_full
    hi_log = yhat_log_future_full + q_hi * se_log_future_full
    lo_future = back_to_original(lo_log, sigma2_log_full)
    hi_future = back_to_original(hi_log, sigma2_log_full)
    spread_lo = lo_future - future_candidates["SARIMAX_full"]
    spread_hi = hi_future - future_candidates["SARIMAX_full"]
    lo_future = yhat_future + spread_lo
    hi_future = yhat_future + spread_hi
else:
    ens_holdout_err = y_test_orig - candidates.get("ENSEMBLE_inv_MAPE", yhat)
    ens_holdout_err = ens_holdout_err[np.isfinite(ens_holdout_err)]
    q_lo_e = (float(np.quantile(ens_holdout_err, ALPHA / 2)) if len(ens_holdout_err)
              else -1.96 * series.std())
    q_hi_e = (float(np.quantile(ens_holdout_err, 1.0 - ALPHA / 2)) if len(ens_holdout_err)
              else 1.96 * series.std())
    h_arr = np.sqrt(np.arange(1, FORECAST_HORIZON + 1))
    lo_future = yhat_future + q_lo_e * h_arr
    hi_future = yhat_future + q_hi_e * h_arr

projection = pd.DataFrame({
    "date"        : future_idx,
    "previs"      : yhat_future,
    "ic_bas"      : lo_future,
    "ic_haut"     : hi_future,
    "fiabilite"   : reliability,
    "day_of_week" : future_idx.day_name(),
    "moteur"      : proj_engine,
})
print(f"Moteur de projection : {proj_engine}")
print("Poids ensemble appliqués sur la projection :")
for n, w in weights.items():
    if n in future_candidates:
        print(f"  {n:30s}  w={w:.3f}")
print(projection.head(15).to_string(index=False))
print(f"\\nProjection sur {FORECAST_HORIZON} jours ouvrés MA "
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
             f"{model_final_label}  |  MAPE test = {mape_model:.2f}%  |  "
             f"Gain vs naïf = {gain_naive:+.1f}%")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()

# Save PNG to out/
_safe = (TARGET_COL.replace(" ", "_")
                   .replace("/", "_")
                   .replace("\\\\", "_")
                   .replace("'", ""))
png_path = OUTPUT_DIR / f"projection_{_safe}.png"
plt.savefig(png_path, dpi=120, bbox_inches="tight")
print(f"✅ PNG sauvegardé : {png_path.resolve()}")
plt.show()'''


EXPORT_CE = '''# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Excel export — 3 onglets : projection / diagnostics / breakpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from datetime import datetime
stamp = datetime.now().strftime("%Y%m%d")
safe_name = (TARGET_COL.replace(" ", "_")
                       .replace("/", "_")
                       .replace("\\\\", "_")
                       .replace("'", ""))
out_path = OUTPUT_DIR / f"projection_robuste_{safe_name}_{stamp}.xlsx"

diagnostics = pd.DataFrame({
    "metric": [
        "model_final", "MAPE_test_pct", "MAE_test", "RMSE_test", "MASE_test",
        "MAPE_naive_pct", "MAPE_quarterly_mean_pct", "MAPE_seasonal_naive_pct",
        "Gain_vs_naive_pct", "Gain_vs_qm_pct", "Gain_vs_seasnv_pct",
        "Pinball_q025", "Pinball_q500", "Pinball_q975",
        "Coverage_IC95_pct", "Calibration_factor",
        "ARIMA_order", "Seasonal_order", "USE_LOG", "horizon_days",
        "n_obs_total", "n_obs_train", "n_pulses", "n_breaks_retained",
    ],
    "value":  [
        model_final_label, mape_model, mae_model, rmse_model, mase_model,
        mape_naive, mape_qm, mape_seasnv,
        gain_naive, gain_qm, gain_seasnv,
        pinball_05, pinball_50, pinball_95,
        coverage * 100 if (coverage is not None and not np.isnan(coverage)) else np.nan,
        calib_factor,
        str(best_order), str(best_seasonal), USE_LOG, FORECAST_HORIZON,
        len(series), train_end, len(pulse_dates) + len(extra_pulses),
        len(break_dates),
    ],
})

if break_diagnostics:
    breakpoints_df = pd.DataFrame([{
        "date": d["date"].date(),
        "chow_F": d["chow_F"],
        "chow_pvalue": d["chow_p"],
        "jump_abs": d["jump_abs"],
    } for d in break_diagnostics])
else:
    breakpoints_df = pd.DataFrame(columns=["date", "chow_F", "chow_pvalue", "jump_abs"])

candidates_df = pd.DataFrame([
    {"model": name,
     "holdout_MAPE_pct": cand_mapes.get(name, np.nan),
     "weight": weights.get(name, 0.0)}
    for name in candidates
])
candidates_df = candidates_df.sort_values("holdout_MAPE_pct").reset_index(drop=True)

with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    projection.to_excel(writer, sheet_name="projection", index=False)
    diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)
    breakpoints_df.to_excel(writer, sheet_name="breakpoints", index=False)
    candidates_df.to_excel(writer, sheet_name="candidates", index=False)

print(f"OK Projection exportee : {out_path.resolve()}")'''


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

MD_HELPERS_MA = '''## Calendrier bancaire marocain

Les jours ouvrés au Maroc ne suivent pas le calendrier occidental seul : il faut retirer les fériés civils marocains (Fête du Trône, Marche Verte, Fête de l'Indépendance, etc.). Cette section MVP gère uniquement les fériés **à date fixe** — les fériés islamiques mobiles (Aïd, Mawlid, Achoura) ne sont pas pris en compte. Toute date de projection ou de détection de gap passe par `ma_business_days`.'''

MD_BATTERY = '''## Batterie de modèles et métriques

Définition centralisée de la batterie : `fit_sarimax` (avec ordre saisonnier), `fit_ets` (Holt-Winters comme benchmark et fallback), `seasonal_naive_forecast` (baseline y_{t-m}), `chow_test` (test formel des ruptures), `pinball_loss`, `mase`, `empirical_coverage`, `back_to_original` (correction log-normale). Tous les blocs suivants réutilisent ces fonctions.'''

MD_BATTERY_EXTRA = '''## Forecasters trend-aware (Theta, Holt-damped, LightGBM, SARIMAX post-rupture)

SARIMAX projette mécaniquement une droite à long horizon (drift constant). Pour capturer la **pente récente** et les **inflexions de tendance**, on ajoute quatre forecasters spécialisés :

- **SARIMAX post-rupture** : refit en tronquant le training à la dernière rupture détectée. Drift et coefficients AR estimés UNIQUEMENT sur le nouveau régime.
- **Theta** : décompose en niveau + pente projetée linéairement. Champion des compétitions M3/M4 sur séries à tendance.
- **Holt damped** : extrapole explicitement la pente avec amortissement (évite l'explosion à long horizon).
- **LightGBM** : modèle non-linéaire sur features lag/rolling/calendar + index temporel. Capture les courbures récentes que ARIMA ne peut pas voir.

Ces forecasters sont ensuite **assemblés** dans un **ensemble pondéré par 1/MAPE** sur le hold-out — le mécanisme de sélection demandé. Plus un modèle est bon en hold-out, plus son poids est élevé dans la projection finale.'''

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
        md(MD_HELPERS_MA),
        code(HELPERS_MA),
        md(MD_LOAD),
        code(CHARGEMENT),
        md(MD_EDA),
        code(EDA_GRID),
        md(MD_STAT),
        code(STATIONNARITE),
        md(MD_BATTERY),
        code(MODEL_BATTERY),
        md(MD_BATTERY_EXTRA),
        code(MODEL_BATTERY_EXTRA),
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
        md(MD_HELPERS_MA),
        code(HELPERS_MA),
        md(MD_LOAD),
        code(CHARGEMENT),
        md(MD_EDA),
        code(EDA_GRID),
        md(MD_STAT),
        code(STATIONNARITE),
        md(MD_BATTERY),
        code(MODEL_BATTERY),
        md(MD_BATTERY_EXTRA),
        code(MODEL_BATTERY_EXTRA),
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

    print("OK Notebooks generated:")
    print(f"  - {here / 'modelisation_comptes_cheques.ipynb'}")
    print(f"  - {here / 'modelisation_credits_equipement.ipynb'}")


if __name__ == "__main__":
    main()
