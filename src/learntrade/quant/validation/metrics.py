"""
Out-of-sample evaluation metrics: IC, Sharpe, block-bootstrap CI, deflated Sharpe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class ICResult:
    ic_mean: float
    ic_std: float
    ic_tstat: float
    rank_ic_mean: float
    hit_rate: float
    n_periods: int


def information_coefficient(signals: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank IC between signals and forward returns."""
    mask = ~(np.isnan(signals) | np.isnan(labels))
    if mask.sum() < 5:
        return float("nan")
    r, _ = spearmanr(signals[mask], labels[mask])
    return float(r)


def pearson_ic(signals: np.ndarray, labels: np.ndarray) -> float:
    """Pearson (linear) IC between signals and forward returns."""
    mask = ~(np.isnan(signals) | np.isnan(labels))
    if mask.sum() < 5:
        return float("nan")
    r, _ = pearsonr(signals[mask], labels[mask])
    return float(r)


def ic_by_date(
    dates: np.ndarray,
    signals: np.ndarray,
    labels: np.ndarray,
    min_names: int = 5,
) -> dict[str, np.ndarray]:
    """
    Compute IC (Pearson) and Rank IC (Spearman) separately for each date's
    cross-section, instead of pooling every (date, instrument) row from a
    fold into a single correlation.

    Pooling across dates inflates the apparent sample size (a 5-fold CV with
    260 trading days per fold looks like "n=5" when it is treated as one IC
    per fold, understating uncertainty) while also conflating the forecast's
    cross-sectional skill with any day-to-day autocorrelation in the signal
    or the label — two different things a single pooled correlation cannot
    separate. Returns one IC/Rank-IC value per date that had at least
    ``min_names`` instruments.
    """
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    ics: list[float] = []
    rank_ics: list[float] = []
    for d in unique_dates:
        m = dates == d
        if m.sum() < min_names:
            continue
        sig_d, lab_d = signals[m], labels[m]
        valid = ~(np.isnan(sig_d) | np.isnan(lab_d))
        if valid.sum() < min_names:
            continue
        ic = pearson_ic(sig_d[valid], lab_d[valid])
        ric = information_coefficient(sig_d[valid], lab_d[valid])
        if not np.isnan(ic):
            ics.append(ic)
        if not np.isnan(ric):
            rank_ics.append(ric)
    return {
        "ic": np.array(ics),
        "rank_ic": np.array(rank_ics),
    }


def ic_tstat(ic_series: np.ndarray) -> float:
    """t-statistic for the IC series: mean / (std / sqrt(n))."""
    n = len(ic_series)
    if n < 2:
        return float("nan")
    std = float(np.std(ic_series, ddof=1))
    if std == 0:
        return float("nan")
    return float(np.mean(ic_series) / (std / np.sqrt(n)))


def ic_tstat_overlap(ic_series: np.ndarray, horizon_days: int) -> float:
    """
    t-statistic that accounts for overlapping forecast windows.

    A 20-day-ahead forecast made on consecutive days shares 19/20 of its outcome
    window with the previous day's, so daily ICs are strongly autocorrelated and
    the naive t-stat (``ic_tstat``) counts ~20x too many independent observations.
    Dividing the sample size by the horizon is the standard first-order fix.
    """
    n = len(ic_series)
    n_eff = n / max(horizon_days, 1)
    if n < 2 or n_eff < 2:
        return float("nan")
    std = float(np.std(ic_series, ddof=1))
    if std == 0:
        return float("nan")
    return float(np.mean(ic_series) / (std / np.sqrt(n_eff)))


def block_bootstrap_ci(
    metric_values: np.ndarray,
    ci: float = 0.95,
    n_bootstrap: int = 1000,
    min_block: int = 1,
) -> tuple[float, float]:
    """
    Block-bootstrap confidence interval (overlapping blocks of size ~sqrt(n)).
    Returns (lower, upper) bounds.
    """
    n = len(metric_values)
    if n < 4:
        m = float(np.mean(metric_values))
        return m, m
    block_size = min(n, max(min_block, int(np.sqrt(n))))
    rng = np.random.default_rng(42)
    boot_means: list[float] = []
    for _ in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, size=n // block_size)
        sample = np.concatenate([metric_values[s: s + block_size] for s in starts])[:n]
        boot_means.append(float(np.mean(sample)))
    alpha = 1 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    n_obs: int = 252,
) -> float:
    """
    Bailey & López de Prado (2014) Deflated Sharpe Ratio.
    Adjusts for multiple testing and non-normal return distributions.
    Returns the probability that the strategy SR > 0 after correction.
    """
    from scipy.stats import norm

    if n_trials <= 0 or n_obs <= 1:
        return float("nan")

    # Expected maximum Sharpe under multiple testing (Euler–Mascheroni approximation)
    gamma = 0.5772156649
    z = norm.ppf(1 - 1.0 / max(n_trials, 2))
    e_max_sr = (1 - gamma) * z + gamma * norm.ppf(1 - 1.0 / (max(n_trials, 2) * np.e))

    # Variance adjustment for non-normality
    skew_adj = skewness * sharpe / 3
    kurt_adj = (kurtosis - 3) * sharpe ** 2 / 4
    sigma_sr = np.sqrt(max((1 - skew_adj + kurt_adj) / (n_obs - 1), 1e-12))

    return float(norm.cdf((sharpe - e_max_sr) / sigma_sr))


def compute_portfolio_metrics(
    daily_returns: np.ndarray,
    annualise_factor: float = 252.0,
) -> dict[str, float]:
    """Sharpe, Sortino, CAGR, max drawdown, Calmar from daily returns."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 10:
        return {"n_days": len(r)}

    n_years = len(r) / annualise_factor
    cum = np.cumprod(1 + r)
    cagr = float(cum[-1] ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(np.std(r, ddof=1) * np.sqrt(annualise_factor))
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(annualise_factor)) if vol > 0 else float("nan")

    running_max = np.maximum.accumulate(cum)
    drawdown = cum / running_max - 1
    max_dd = float(np.min(drawdown))

    neg_r = r[r < 0]
    down_vol = float(np.std(neg_r, ddof=1) * np.sqrt(annualise_factor)) if len(neg_r) > 1 else float("nan")
    sortino = (cagr / down_vol) if down_vol and down_vol > 0 else float("nan")

    return {
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": abs(cagr / max_dd) if max_dd != 0 else float("nan"),
        "n_days": len(r),
    }
