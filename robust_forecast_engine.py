from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents


np.random.seed(42)
warnings.filterwarnings("ignore")


MA_FIXED_HOLIDAYS = {
    (1, 1),
    (1, 11),
    (5, 1),
    (7, 30),
    (8, 14),
    (8, 20),
    (8, 21),
    (11, 6),
    (11, 18),
}


@dataclass(frozen=True)
class ForecastConfig:
    input_path: Path = Path("in/reserve_in.xlsx")
    output_dir: Path = Path("out")
    date_col: str = "date"
    holdout_len: int = 60
    forecast_horizon: int = 90
    wf_folds: int = 5
    wf_horizon: int = 30
    seasonal_period: int = 5
    min_gain_vs_baseline_pct: float = 5.0
    top_n_ensemble: int = 3
    alpha: float = 0.05
    peak_quantile: float = 0.80
    peak_metric_weight: float = 0.45
    peak_weight_multiplier: float = 4.0
    max_mape_degradation_vs_baseline_pct: float = 25.0


@dataclass
class ForecastResult:
    target: str
    final_model: str
    projection: pd.DataFrame
    holdout: pd.DataFrame
    diagnostics: pd.DataFrame
    candidates: pd.DataFrame
    profile: pd.DataFrame


def is_ma_holiday(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    return (ts.month, ts.day) in MA_FIXED_HOLIDAYS


def ma_business_days(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    raw = pd.bdate_range(start=pd.Timestamp(start), periods=int(periods * 1.8) + 30)
    keep = [d for d in raw if not is_ma_holiday(d)]
    return pd.DatetimeIndex(keep[:periods])


def load_workbook(config: ForecastConfig) -> pd.DataFrame:
    if not config.input_path.exists():
        raise FileNotFoundError(config.input_path)
    df = pd.read_excel(config.input_path)
    if config.date_col not in df.columns:
        date_candidates = [c for c in df.columns if str(c).lower() in {"date", "jour", "day"}]
        if not date_candidates:
            raise KeyError("No date column found")
        date_col = date_candidates[0]
    else:
        date_col = config.date_col
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df = (
        df.dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .set_index(date_col)
    )
    df.index.name = "date"
    return df


def calendar_exog(index: pd.DatetimeIndex) -> pd.DataFrame:
    exog = pd.DataFrame(index=index)
    for k in range(1, 5):
        exog[f"dow_{k}"] = (index.dayofweek == k).astype(float)
    exog["eom"] = (index.day >= 25).astype(float)
    exog["bom"] = (index.day <= 5).astype(float)
    exog["eoq"] = (index.month.isin([3, 6, 9, 12]) & (index.day >= 20)).astype(float)
    exog["august"] = (index.month == 8).astype(float)
    return exog


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-9, np.nan, denom)
    return float(np.nanmean(2.0 * np.abs(y_pred - y_true) / denom) * 100)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) > season:
        scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    else:
        scale = np.mean(np.abs(np.diff(y_train)))
    scale = max(float(scale), 1e-9)
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def peak_threshold(y_train: pd.Series | np.ndarray, quantile: float) -> float:
    vals = np.asarray(y_train, dtype=float)
    return float(np.quantile(vals[np.isfinite(vals)], quantile))


def local_spike_threshold(y_train: pd.Series | np.ndarray, quantile: float) -> float:
    vals = pd.Series(np.asarray(y_train, dtype=float))
    local_base = vals.rolling(20, min_periods=8).median().bfill()
    excess = (vals - local_base).replace([np.inf, -np.inf], np.nan).dropna()
    if excess.empty:
        return 0.0
    return max(float(np.quantile(excess, quantile)), 0.0)


def local_spike_reference(y_train: pd.Series | np.ndarray, horizon: int) -> np.ndarray:
    vals = np.asarray(y_train, dtype=float)
    drift = float(np.mean(np.diff(vals))) if len(vals) > 1 else 0.0
    return vals[-1] + drift * np.arange(1, horizon + 1)


def local_spike_mask(
    y_values: np.ndarray,
    y_train: pd.Series | np.ndarray,
    quantile: float,
) -> np.ndarray:
    y_values = np.asarray(y_values, dtype=float)
    ref = local_spike_reference(y_train, len(y_values))
    threshold = local_spike_threshold(y_train, quantile)
    return (y_values - ref) >= threshold


def peak_weighted_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: pd.Series | np.ndarray,
    quantile: float,
    peak_multiplier: float,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    peak_mask = local_spike_mask(y_true, y_train, quantile)
    weights = np.ones_like(y_true, dtype=float)
    weights[peak_mask] = peak_multiplier
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    ape = np.abs(y_true - y_pred) / denom * 100
    return float(np.nansum(weights * ape) / np.nansum(weights))


def peak_capture_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    actual_peak = local_spike_mask(y_true, y_train, quantile)
    if actual_peak.sum() == 0:
        return np.nan
    predicted_peak = local_spike_mask(y_pred, y_train, quantile)
    return float((actual_peak & predicted_peak).sum() / actual_peak.sum())


def balanced_model_score(
    mape_value: float,
    peak_mape_value: float,
    peak_capture_value: float,
    config: ForecastConfig,
) -> float:
    capture_penalty = 0.0
    if np.isfinite(peak_capture_value):
        capture_penalty = max(0.0, 0.50 - peak_capture_value) * 4.0
    return float(
        (1.0 - config.peak_metric_weight) * mape_value
        + config.peak_metric_weight * peak_mape_value
        + capture_penalty
    )


def seasonal_naive(y_train: pd.Series, horizon: int, season: int = 5) -> np.ndarray:
    vals = y_train.values.astype(float)
    out = np.empty(horizon, dtype=float)
    for h in range(horizon):
        out[h] = vals[max(0, len(vals) - season + (h % season))]
    return out


def drift_forecast(y_train: pd.Series, horizon: int) -> np.ndarray:
    vals = y_train.values.astype(float)
    drift = float(np.mean(np.diff(vals))) if len(vals) > 1 else 0.0
    return vals[-1] + drift * np.arange(1, horizon + 1)


def fit_sarimax_log(y_train: pd.Series, horizon: int, future_index: pd.DatetimeIndex) -> np.ndarray | None:
    if (y_train <= 0).any():
        return None
    try:
        y_log = np.log(y_train)
        exog = calendar_exog(y_train.index)
        exog_f = calendar_exog(future_index)
        best = None
        best_aic = np.inf
        # Keep this grid intentionally small. The engine evaluates the full
        # model panel across rolling folds; a wide SARIMAX grid makes the
        # notebook too slow for decision-support usage.
        for order in [(0, 1, 1), (1, 1, 0), (1, 1, 1)]:
            for seasonal in [(0, 0, 0, 0)]:
                try:
                    res = SARIMAX(
                        y_log,
                        exog=exog,
                        order=order,
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=250)
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best = res
                        best_aic = res.aic
                except Exception:
                    continue
        if best is None:
            return None
        fc = best.get_forecast(steps=horizon, exog=exog_f).predicted_mean.values
        sigma2 = float(np.var(best.resid))
        return np.exp(fc + 0.5 * sigma2)
    except Exception:
        return None


def fit_ets(y_train: pd.Series, horizon: int, season: int = 5) -> np.ndarray | None:
    try:
        model = ExponentialSmoothing(
            y_train,
            trend="add",
            damped_trend=True,
            seasonal="add" if len(y_train) > 2 * season + 20 else None,
            seasonal_periods=season if len(y_train) > 2 * season + 20 else None,
            initialization_method="estimated",
        )
        return np.asarray(model.fit(optimized=True).forecast(horizon), dtype=float)
    except Exception:
        return None


def fit_theta(y_train: pd.Series, horizon: int, season: int = 5) -> np.ndarray | None:
    try:
        use_seasonal = len(y_train) > 2 * season + 20
        model = ThetaModel(y_train, period=season if use_seasonal else None, deseasonalize=use_seasonal)
        return np.asarray(model.fit().forecast(horizon), dtype=float)
    except Exception:
        return None


def fit_ucm(y_train: pd.Series, horizon: int) -> np.ndarray | None:
    try:
        model = UnobservedComponents(y_train, level="local linear trend", seasonal=5)
        res = model.fit(disp=False, maxiter=250)
        return np.asarray(res.get_forecast(steps=horizon).predicted_mean, dtype=float)
    except Exception:
        try:
            model = UnobservedComponents(y_train, level="local linear trend")
            res = model.fit(disp=False, maxiter=250)
            return np.asarray(res.get_forecast(steps=horizon).predicted_mean, dtype=float)
        except Exception:
            return None


def fit_knn_analog_paths(
    y_train: pd.Series,
    horizon: int,
    future_index: pd.DatetimeIndex,
    window: int = 30,
    k: int = 8,
) -> np.ndarray | None:
    """Historical analog paths.

    Looks for past windows shaped like the most recent window and replays the
    following path, scaled to today's level. This is deliberately peak-aware:
    if similar historical contexts were followed by spikes, the forecast path
    can contain spikes instead of collapsing to a straight line.
    """
    vals = y_train.values.astype(float)
    n = len(vals)
    if n < window + horizon + 80:
        return None

    current = vals[-window:]
    cur_scale = max(float(np.mean(current)), 1e-9)
    current_shape = current / cur_scale - 1.0
    candidates = []
    for start in range(window, n - horizon):
        past = vals[start - window : start]
        future = vals[start : start + horizon]
        past_scale = max(float(np.mean(past)), 1e-9)
        past_shape = past / past_scale - 1.0
        shape_dist = float(np.sqrt(np.mean((current_shape - past_shape) ** 2)))
        dow_penalty = 0.02 * abs(int(y_train.index[start].dayofweek) - int(future_index[0].dayofweek))
        dist = shape_dist + dow_penalty
        scale = vals[-1] / max(past[-1], 1e-9)
        candidates.append((dist, future * scale))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    chosen = candidates[:k]
    weights = np.array([1.0 / max(d, 1e-6) for d, _ in chosen], dtype=float)
    weights = weights / weights.sum()
    out = np.zeros(horizon, dtype=float)
    for w, (_, path) in zip(weights, chosen):
        out += w * path
    return np.maximum(out, 0.0)


def _calendar_bucket(index: pd.DatetimeIndex) -> pd.Series:
    days = pd.Series(index.day, index=index)
    return pd.cut(days, bins=[0, 5, 10, 15, 20, 25, 31], labels=False, include_lowest=True)


def fit_peak_calendar_overlay(y_train: pd.Series, horizon: int, future_index: pd.DatetimeIndex) -> np.ndarray | None:
    """Drift forecast multiplied by robust calendar peak factors."""
    if len(y_train) < 160:
        return None
    base_hist = y_train.rolling(20, min_periods=10).median().bfill()
    ratio = (y_train / base_hist).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return None

    train_idx = ratio.index
    dow_effect = ratio.groupby(train_idx.dayofweek).median()
    dom_effect = ratio.groupby(_calendar_bucket(train_idx)).median()
    eom_effect = ratio.groupby(train_idx.day >= 25).median()
    eoq_mask = train_idx.month.isin([3, 6, 9, 12]) & (train_idx.day >= 20)
    eoq_effect = ratio.groupby(eoq_mask).median()

    baseline = drift_forecast(y_train, horizon)
    factors = []
    future_bucket = _calendar_bucket(future_index)
    for i, d in enumerate(future_index):
        pieces = [
            float(dow_effect.get(d.dayofweek, 1.0)),
            float(dom_effect.get(future_bucket.iloc[i], 1.0)),
            float(eom_effect.get(d.day >= 25, 1.0)),
            float(eoq_effect.get(d.month in [3, 6, 9, 12] and d.day >= 20, 1.0)),
        ]
        factor = 0.35 * pieces[0] + 0.25 * pieces[1] + 0.25 * pieces[2] + 0.15 * pieces[3]
        factors.append(float(np.clip(factor, 0.90, 1.12)))
    return np.maximum(baseline * np.asarray(factors), 0.0)


def fit_recent_shape_replay(y_train: pd.Series, horizon: int, pattern: int = 20) -> np.ndarray | None:
    """Replay the recent local shape on top of a drift baseline."""
    if len(y_train) < pattern + 80:
        return None
    baseline = drift_forecast(y_train, horizon)
    hist_base = drift_forecast(y_train.iloc[:-pattern], pattern)
    recent = y_train.iloc[-pattern:].values.astype(float)
    ratios = recent / np.maximum(hist_base, 1e-9)
    ratios = np.clip(ratios, 0.88, 1.14)
    repeated = np.resize(ratios, horizon)
    return np.maximum(baseline * repeated, 0.0)


def model_forecasts(y_train: pd.Series, horizon: int, future_index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    forecasts: dict[str, np.ndarray] = {
        "BASELINE_last": np.full(horizon, y_train.iloc[-1], dtype=float),
        "BASELINE_drift": drift_forecast(y_train, horizon),
        "BASELINE_seasonal_m5": seasonal_naive(y_train, horizon),
    }
    candidates: dict[str, Callable[[], np.ndarray | None]] = {
        "SARIMAX_log_calendar": lambda: fit_sarimax_log(y_train, horizon, future_index),
        "ETS_damped": lambda: fit_ets(y_train, horizon),
        "Theta": lambda: fit_theta(y_train, horizon),
        "UCM_local_linear": lambda: fit_ucm(y_train, horizon),
        "KNN_analog_paths": lambda: fit_knn_analog_paths(y_train, horizon, future_index),
        "Peak_calendar_overlay": lambda: fit_peak_calendar_overlay(y_train, horizon, future_index),
        "Recent_shape_replay": lambda: fit_recent_shape_replay(y_train, horizon),
    }
    for name, func in candidates.items():
        pred = func()
        if pred is not None and len(pred) == horizon and np.all(np.isfinite(pred)):
            forecasts[name] = np.maximum(pred.astype(float), 0.0)
    anchor = 0.50 * forecasts["BASELINE_last"] + 0.50 * forecasts["BASELINE_drift"]
    for name in ["UCM_local_linear", "KNN_analog_paths", "Peak_calendar_overlay", "Recent_shape_replay"]:
        if name in forecasts:
            forecasts[f"SoftPeak_{name}"] = np.maximum(0.65 * anchor + 0.35 * forecasts[name], 0.0)
    return forecasts


def evaluate_walk_forward(y: pd.Series, config: ForecastConfig) -> pd.DataFrame:
    rows = []
    n = len(y)
    train_min = n - config.wf_folds * config.wf_horizon
    if train_min < 180:
        raise ValueError("Not enough observations for robust walk-forward evaluation")
    for fold in range(config.wf_folds):
        end_train = train_min + fold * config.wf_horizon
        y_train = y.iloc[:end_train]
        y_true = y.iloc[end_train : end_train + config.wf_horizon]
        future_idx = y_true.index
        forecasts = model_forecasts(y_train, len(y_true), future_idx)
        for model_name, pred in forecasts.items():
            mape_value = mape(y_true.values, pred)
            peak_mape_value = peak_weighted_mape(
                y_true.values,
                pred,
                y_train.values,
                config.peak_quantile,
                config.peak_weight_multiplier,
            )
            peak_capture_value = peak_capture_rate(
                y_true.values,
                pred,
                y_train.values,
                config.peak_quantile,
            )
            rows.append(
                {
                    "fold": fold + 1,
                    "model": model_name,
                    "MAPE": mape_value,
                    "sMAPE": smape(y_true.values, pred),
                    "MASE": mase(y_true.values, pred, y_train.values, config.seasonal_period),
                    "peak_wMAPE": peak_mape_value,
                    "peak_capture": peak_capture_value,
                    "score": balanced_model_score(mape_value, peak_mape_value, peak_capture_value, config),
                }
            )
    wf = pd.DataFrame(rows)
    med = (
        wf.groupby("model", as_index=False)
        .agg(
            wf_MAPE_median=("MAPE", "median"),
            wf_peak_wMAPE_median=("peak_wMAPE", "median"),
            wf_peak_capture_mean=("peak_capture", "mean"),
            wf_score_median=("score", "median"),
            wf_MAPE_mean=("MAPE", "mean"),
            wf_MASE_median=("MASE", "median"),
        )
        .sort_values("wf_score_median")
        .reset_index(drop=True)
    )
    return med


def inverse_mape_weights(candidate_scores: pd.DataFrame, top_n: int) -> dict[str, float]:
    non_base = candidate_scores[~candidate_scores["model"].str.startswith("BASELINE_")].copy()
    score_col = "wf_score_median" if "wf_score_median" in non_base.columns else "wf_MAPE_median"
    non_base = non_base.replace([np.inf, -np.inf], np.nan).dropna(subset=[score_col])
    non_base = non_base.sort_values(score_col).head(top_n)
    if non_base.empty:
        return {}
    inv = 1.0 / np.maximum(non_base[score_col].values, 1e-6)
    inv = inv / inv.sum()
    return {m: float(w) for m, w in zip(non_base["model"], inv)}


def weighted_ensemble(forecasts: dict[str, np.ndarray], weights: dict[str, float], horizon: int) -> np.ndarray | None:
    active = {name: w for name, w in weights.items() if name in forecasts}
    if not active:
        return None
    total = sum(active.values())
    out = np.zeros(horizon, dtype=float)
    for name, weight in active.items():
        out += (weight / total) * forecasts[name]
    return out


def series_profile(y: pd.Series) -> pd.DataFrame:
    diff = y.diff().dropna()
    robust_z = (diff - diff.median()) / max(1.4826 * (diff - diff.median()).abs().median(), 1e-9)
    outlier_count = int((robust_z.abs() > 5).sum())
    last60 = y.iloc[-60:].mean()
    prev60 = y.iloc[-120:-60].mean() if len(y) >= 120 else np.nan
    return pd.DataFrame(
        [
            {"metric": "start_date", "value": y.index.min().date()},
            {"metric": "end_date", "value": y.index.max().date()},
            {"metric": "n_obs", "value": len(y)},
            {"metric": "min", "value": y.min()},
            {"metric": "max", "value": y.max()},
            {"metric": "mean", "value": y.mean()},
            {"metric": "std", "value": y.std()},
            {"metric": "cv", "value": y.std() / max(abs(y.mean()), 1e-9)},
            {"metric": "zeros", "value": int((y == 0).sum())},
            {"metric": "negatives", "value": int((y < 0).sum())},
            {"metric": "robust_diff_outliers_z5", "value": outlier_count},
            {"metric": "last60_mean", "value": last60},
            {"metric": "prev60_mean", "value": prev60},
            {"metric": "last60_vs_prev60_pct", "value": (last60 / prev60 - 1) * 100 if prev60 else np.nan},
        ]
    )


def run_one_target(df: pd.DataFrame, target: str, config: ForecastConfig) -> ForecastResult:
    if target not in df.columns:
        raise KeyError(target)
    y = df[target].astype(float).dropna()
    if len(y) < config.holdout_len + config.wf_folds * config.wf_horizon + 120:
        raise ValueError(f"{target}: history too short")

    selection_y = y.iloc[: -config.holdout_len]
    wf_scores = evaluate_walk_forward(selection_y, config)
    weights = inverse_mape_weights(wf_scores, config.top_n_ensemble)

    train = y.iloc[: -config.holdout_len]
    holdout = y.iloc[-config.holdout_len :]
    holdout_forecasts = model_forecasts(train, len(holdout), holdout.index)
    ensemble_holdout = weighted_ensemble(holdout_forecasts, weights, len(holdout))
    if ensemble_holdout is not None:
        holdout_forecasts["ENSEMBLE_wf_top"] = ensemble_holdout

    holdout_rows = []
    for name, pred in holdout_forecasts.items():
        mape_value = mape(holdout.values, pred)
        peak_mape_value = peak_weighted_mape(
            holdout.values,
            pred,
            train.values,
            config.peak_quantile,
            config.peak_weight_multiplier,
        )
        peak_capture_value = peak_capture_rate(
            holdout.values,
            pred,
            train.values,
            config.peak_quantile,
        )
        holdout_rows.append(
            {
                "model": name,
                "holdout_MAPE": mape_value,
                "holdout_sMAPE": smape(holdout.values, pred),
                "holdout_MAE": mae(holdout.values, pred),
                "holdout_RMSE": rmse(holdout.values, pred),
                "holdout_MASE": mase(holdout.values, pred, train.values, config.seasonal_period),
                "holdout_peak_wMAPE": peak_mape_value,
                "holdout_peak_capture": peak_capture_value,
                "holdout_score": balanced_model_score(
                    mape_value,
                    peak_mape_value,
                    peak_capture_value,
                    config,
                ),
            }
        )
    holdout_scores = pd.DataFrame(holdout_rows).sort_values("holdout_score").reset_index(drop=True)

    best_baseline = (
        holdout_scores[holdout_scores["model"].str.startswith("BASELINE_")]
        .sort_values("holdout_score")
        .iloc[0]
    )
    best_mape_baseline = (
        holdout_scores[holdout_scores["model"].str.startswith("BASELINE_")]
        .sort_values("holdout_MAPE")
        .iloc[0]
    )
    non_baseline_scores = (
        holdout_scores[~holdout_scores["model"].str.startswith("BASELINE_")]
        .sort_values("holdout_score")
        .reset_index(drop=True)
    )
    proposed_model = non_baseline_scores.iloc[0]["model"]
    proposed_score = float(holdout_scores.loc[holdout_scores["model"] == proposed_model, "holdout_score"].iloc[0])
    proposed_mape = float(holdout_scores.loc[holdout_scores["model"] == proposed_model, "holdout_MAPE"].iloc[0])
    baseline_score = float(best_baseline["holdout_score"])
    baseline_mape = float(best_mape_baseline["holdout_MAPE"])
    gain = (baseline_score - proposed_score) / max(baseline_score, 1e-9) * 100
    mape_degradation = (proposed_mape - baseline_mape) / max(baseline_mape, 1e-9) * 100
    final_model = str(best_mape_baseline["model"])
    for _, row in non_baseline_scores.iterrows():
        row_score = float(row["holdout_score"])
        row_mape = float(row["holdout_MAPE"])
        row_gain = (baseline_score - row_score) / max(baseline_score, 1e-9) * 100
        row_mape_degradation = (row_mape - baseline_mape) / max(baseline_mape, 1e-9) * 100
        if (
            row_gain >= config.min_gain_vs_baseline_pct
            and row_mape_degradation <= config.max_mape_degradation_vs_baseline_pct
        ):
            final_model = str(row["model"])
            break

    peak_candidates = non_baseline_scores.copy()
    peak_candidates["capture_rank"] = peak_candidates["holdout_peak_capture"].fillna(-1.0)
    peak_scenario_model = str(
        peak_candidates.sort_values(
            ["capture_rank", "holdout_peak_wMAPE", "holdout_score"],
            ascending=[False, True, True],
        ).iloc[0]["model"]
    )

    future_idx = ma_business_days(y.index[-1] + pd.Timedelta(days=1), config.forecast_horizon)
    future_forecasts = model_forecasts(y, config.forecast_horizon, future_idx)
    ensemble_future = weighted_ensemble(future_forecasts, weights, config.forecast_horizon)
    if ensemble_future is not None:
        future_forecasts["ENSEMBLE_wf_top"] = ensemble_future
    if final_model not in future_forecasts:
        final_model = str(best_mape_baseline["model"])
    yhat_future = future_forecasts[final_model]
    peak_scenario_future = future_forecasts.get(peak_scenario_model, yhat_future)

    holdout_pred = (
        holdout_forecasts[final_model]
        if final_model in holdout_forecasts
        else holdout_forecasts[str(best_mape_baseline["model"])]
    )
    residuals = holdout.values - holdout_pred
    q_lo = float(np.quantile(residuals, config.alpha / 2))
    q_hi = float(np.quantile(residuals, 1 - config.alpha / 2))
    horizon_scale = np.sqrt(np.arange(1, config.forecast_horizon + 1) / max(config.wf_horizon, 1))
    horizon_scale = np.clip(horizon_scale, 1.0, 2.0)
    lo = np.maximum(yhat_future + q_lo * horizon_scale, 0.0)
    hi = np.maximum(yhat_future + q_hi * horizon_scale, 0.0)
    threshold = peak_threshold(y, config.peak_quantile)
    local_threshold = local_spike_threshold(y, config.peak_quantile)
    future_reference = local_spike_reference(y, config.forecast_horizon)
    peak_risk = np.select(
        [
            (yhat_future - future_reference) >= local_threshold,
            (hi - future_reference) >= local_threshold,
        ],
        ["high", "watch"],
        default="normal",
    )

    projection = pd.DataFrame(
        {
            "date": future_idx,
            "target": target,
            "forecast": yhat_future,
            "peak_scenario_forecast": peak_scenario_future,
            "ic_low": lo,
            "ic_high": hi,
            "reliability": np.select(
                [
                    np.arange(config.forecast_horizon) < 30,
                    np.arange(config.forecast_horizon) < 60,
                ],
                ["high", "medium"],
                default="low",
            ),
            "model": final_model,
            "peak_scenario_model": peak_scenario_model,
            "level_peak_threshold": threshold,
            "local_spike_threshold": local_threshold,
            "local_reference": future_reference,
            "peak_risk": peak_risk,
            "day_of_week": future_idx.day_name(),
            "is_fixed_ma_holiday": [is_ma_holiday(d) for d in future_idx],
        }
    )

    holdout_df = pd.DataFrame(
        {
            "date": holdout.index,
            "target": target,
            "actual": holdout.values,
            "prediction": holdout_pred,
            "error": holdout.values - holdout_pred,
            "abs_pct_error": np.abs(holdout.values - holdout_pred) / np.maximum(np.abs(holdout.values), 1e-9) * 100,
            "actual_peak": local_spike_mask(holdout.values, train.values, config.peak_quantile),
            "predicted_peak": local_spike_mask(holdout_pred, train.values, config.peak_quantile),
            "model": final_model,
        }
    )

    candidates = holdout_scores.merge(wf_scores, on="model", how="outer")
    candidates["ensemble_weight"] = candidates["model"].map(weights).fillna(0.0)
    candidates = candidates.sort_values(["holdout_score", "wf_score_median"], na_position="last")

    final_metrics = holdout_scores.loc[holdout_scores["model"] == final_model].iloc[0]
    diagnostics = pd.DataFrame(
        [
            {"metric": "target", "value": target},
            {"metric": "final_model", "value": final_model},
            {"metric": "proposed_model", "value": proposed_model},
            {"metric": "peak_scenario_model", "value": peak_scenario_model},
            {"metric": "best_baseline", "value": best_baseline["model"]},
            {"metric": "best_mape_baseline", "value": best_mape_baseline["model"]},
            {"metric": "gain_vs_best_baseline_pct", "value": gain},
            {"metric": "mape_degradation_vs_best_baseline_pct", "value": mape_degradation},
            {"metric": "min_required_gain_pct", "value": config.min_gain_vs_baseline_pct},
            {"metric": "max_allowed_mape_degradation_pct", "value": config.max_mape_degradation_vs_baseline_pct},
            {"metric": "holdout_MAPE", "value": final_metrics["holdout_MAPE"]},
            {"metric": "holdout_peak_wMAPE", "value": final_metrics["holdout_peak_wMAPE"]},
            {"metric": "holdout_peak_capture", "value": final_metrics["holdout_peak_capture"]},
            {"metric": "holdout_score", "value": final_metrics["holdout_score"]},
            {"metric": "holdout_sMAPE", "value": final_metrics["holdout_sMAPE"]},
            {"metric": "holdout_MAE", "value": final_metrics["holdout_MAE"]},
            {"metric": "holdout_RMSE", "value": final_metrics["holdout_RMSE"]},
            {"metric": "holdout_MASE", "value": final_metrics["holdout_MASE"]},
            {"metric": "holdout_resid_skew", "value": stats.skew(residuals)},
            {"metric": "holdout_resid_kurtosis", "value": stats.kurtosis(residuals)},
            {"metric": "n_obs", "value": len(y)},
            {"metric": "holdout_len", "value": config.holdout_len},
            {"metric": "forecast_horizon", "value": config.forecast_horizon},
        ]
    )

    return ForecastResult(
        target=target,
        final_model=final_model,
        projection=projection,
        holdout=holdout_df,
        diagnostics=diagnostics,
        candidates=candidates,
        profile=series_profile(y),
    )


def save_result_bundle(name: str, results: list[ForecastResult], config: ForecastConfig) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.today().strftime("%Y%m%d")
    out_path = config.output_dir / f"robust_forecast_{name}_{stamp}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary = pd.concat([r.diagnostics.assign(series=r.target) for r in results], ignore_index=True)
        summary.to_excel(writer, sheet_name="summary", index=False)
        for r in results:
            short = safe_sheet_name(r.target)
            r.projection.to_excel(writer, sheet_name=f"proj_{short}", index=False)
            r.holdout.to_excel(writer, sheet_name=f"hold_{short}", index=False)
            r.candidates.to_excel(writer, sheet_name=f"models_{short}", index=False)
            r.profile.to_excel(writer, sheet_name=f"profile_{short}", index=False)
    return out_path


def safe_sheet_name(name: str) -> str:
    repl = (
        name.replace("Depots Clientele_", "")
        .replace("Credit Decaissement_", "")
        .replace("Credit Décaissement_", "")
        .replace(" ", "_")
        .replace("è", "e")
        .replace("é", "e")
        .replace("à", "a")
        .replace("'", "")
    )
    return repl[:20]


def plot_result(result: ForecastResult, history: pd.Series, output_dir: Path, prefix: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_tail = history.iloc[-260:]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(hist_tail.index, hist_tail.values, label="history", color="#1f4e79", lw=1.2)
    ax.plot(result.holdout["date"], result.holdout["actual"], label="holdout actual", color="black", lw=1.0)
    ax.plot(result.holdout["date"], result.holdout["prediction"], label="holdout forecast", color="#f28e2b", lw=1.1)
    ax.plot(result.projection["date"], result.projection["forecast"], label="90d forecast", color="#c00000", lw=1.3)
    ax.fill_between(
        result.projection["date"],
        result.projection["ic_low"],
        result.projection["ic_high"],
        color="#4e79a7",
        alpha=0.18,
        label="empirical interval",
    )
    mape_value = result.diagnostics.loc[result.diagnostics["metric"] == "holdout_MAPE", "value"].iloc[0]
    ax.set_title(f"{result.target} | {result.final_model} | holdout MAPE={float(mape_value):.2f}%")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    path = output_dir / f"{prefix}_{safe_sheet_name(result.target)}.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def run_bundle(name: str, targets: list[str], config: ForecastConfig, add_total: bool = False) -> list[ForecastResult]:
    df = load_workbook(config)
    if add_total:
        total_name = "Depots Clientele_total_cheques_courants"
        df[total_name] = df[targets].sum(axis=1)
        targets = [*targets, total_name]

    results = [run_one_target(df, target, config) for target in targets]
    out_xlsx = save_result_bundle(name, results, config)
    for result in results:
        plot_result(result, df[result.target].astype(float).dropna(), config.output_dir, name)
    print(f"Saved workbook: {out_xlsx}")
    for result in results:
        mape_value = result.diagnostics.loc[result.diagnostics["metric"] == "holdout_MAPE", "value"].iloc[0]
        gain_value = result.diagnostics.loc[result.diagnostics["metric"] == "gain_vs_best_baseline_pct", "value"].iloc[0]
        print(f"{result.target}: model={result.final_model}, MAPE={float(mape_value):.3f}%, gain={float(gain_value):+.2f}%")
    return results
