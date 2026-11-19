"""
Portfolio allocators — EqualWeight, InverseVol, MeanVariance, HRP, RiskParity, CVaR.

All allocators implement the same interface:
    allocate(forecasts, covariance, constraints) -> {InstrumentId: weight}

Weights sum to ≤ 1.0. No allocator enforces portfolio constraints directly —
that is handled by apply_constraints() in constraints.py.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from leantrade.core.types import Forecast, InstrumentId

logger = logging.getLogger(__name__)


class Allocator(ABC):
    @abstractmethod
    def allocate(
        self,
        forecasts: list[Forecast],
        covariance: np.ndarray | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> dict[InstrumentId, float]:
        ...


# ── Baselines ─────────────────────────────────────────────────────────────────

class EqualWeightAllocator(Allocator):
    def allocate(self, forecasts, covariance=None, constraints=None):
        n = len(forecasts)
        if n == 0:
            return {}
        w = 1.0 / n
        return {f.instrument_id: w for f in forecasts}


class InverseVolAllocator(Allocator):
    def allocate(self, forecasts, covariance=None, constraints=None):
        if not forecasts:
            return {}
        inv_vols = np.array([1.0 / max(f.volatility, 1e-9) for f in forecasts])
        weights = inv_vols / inv_vols.sum()
        return {f.instrument_id: float(w) for f, w in zip(forecasts, weights)}


# ── Mean-Variance ─────────────────────────────────────────────────────────────

class MeanVarianceAllocator(Allocator):
    """
    Maximise Sharpe ratio subject to long-only and sum-to-1 constraints.
    Uses Ledoit-Wolf shrinkage on the covariance matrix.
    """

    def __init__(self, risk_free_rate: float = 0.0) -> None:
        self._rf = risk_free_rate

    def allocate(self, forecasts, covariance=None, constraints=None):
        if not forecasts:
            return {}
        n = len(forecasts)
        mu = np.array([f.expected_return for f in forecasts])

        if covariance is None or covariance.shape != (n, n):
            # Fall back to diagonal covariance from forecast vols
            vols = np.array([f.volatility for f in forecasts])
            covariance = np.diag(vols ** 2)

        # Ledoit-Wolf shrinkage
        cov = _shrink_covariance(covariance)

        try:
            from scipy.optimize import minimize
            w0 = np.ones(n) / n

            def neg_sharpe(w):
                port_ret = w @ mu
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_ret - self._rf) / (port_vol + 1e-9)

            result = minimize(
                neg_sharpe, w0,
                bounds=[(0, 1)] * n,
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                method="SLSQP",
            )
            weights = result.x if result.success else w0
        except Exception as exc:
            logger.warning("MeanVariance failed: %s — falling back to equal weight", exc)
            weights = np.ones(n) / n

        weights = np.clip(weights, 0, 1)
        total = weights.sum()
        if total > 0:
            weights /= total
        return {f.instrument_id: float(w) for f, w in zip(forecasts, weights)}


# ── Hierarchical Risk Parity ──────────────────────────────────────────────────

class HRPAllocator(Allocator):
    """
    Hierarchical Risk Parity — López de Prado (2016).
    Clusters instruments by correlation and allocates via recursive bisection.
    """

    def allocate(self, forecasts, covariance=None, constraints=None):
        if not forecasts:
            return {}
        n = len(forecasts)
        if n == 1:
            return {forecasts[0].instrument_id: 1.0}

        vols = np.array([max(f.volatility, 1e-9) for f in forecasts])
        if covariance is None or covariance.shape != (n, n):
            covariance = np.diag(vols ** 2)
        cov = _shrink_covariance(covariance)

        weights = _hrp_weights(cov)
        weights = np.clip(weights, 0, 1)
        total = weights.sum()
        if total > 0:
            weights /= total
        return {f.instrument_id: float(w) for f, w in zip(forecasts, weights)}


# ── Risk Parity ───────────────────────────────────────────────────────────────

class RiskParityAllocator(Allocator):
    """Equal risk contribution from each asset."""

    def allocate(self, forecasts, covariance=None, constraints=None):
        if not forecasts:
            return {}
        n = len(forecasts)
        vols = np.array([max(f.volatility, 1e-9) for f in forecasts])
        if covariance is None or covariance.shape != (n, n):
            covariance = np.diag(vols ** 2)
        cov = _shrink_covariance(covariance)

        weights = _risk_parity_weights(cov)
        weights = np.clip(weights, 0, 1)
        total = weights.sum()
        if total > 0:
            weights /= total
        return {f.instrument_id: float(w) for f, w in zip(forecasts, weights)}


# ── CVaR ──────────────────────────────────────────────────────────────────────

class CVaRAllocator(Allocator):
    """
    Minimise Conditional Value at Risk (CVaR / Expected Shortfall).
    Requires cvxpy; falls back to InverseVol if not available.
    """

    def __init__(self, alpha: float = 0.05, n_scenarios: int = 1000) -> None:
        self._alpha = alpha
        self._n_scenarios = n_scenarios

    def allocate(self, forecasts, covariance=None, constraints=None):
        if not forecasts:
            return {}
        n = len(forecasts)
        vols = np.array([max(f.volatility, 1e-9) for f in forecasts])
        if covariance is None or covariance.shape != (n, n):
            covariance = np.diag(vols ** 2)
        cov = _shrink_covariance(covariance)

        try:
            weights = _cvar_weights(cov, self._alpha, self._n_scenarios)
            weights = np.clip(weights, 0, 1)
            total = weights.sum()
            if total > 0:
                weights /= total
            return {f.instrument_id: float(w) for f, w in zip(forecasts, weights)}
        except Exception as exc:
            logger.warning("CVaR failed (%s) — falling back to InverseVol", exc)
            return InverseVolAllocator().allocate(forecasts, covariance, constraints)


# ── Helper functions ──────────────────────────────────────────────────────────

def _shrink_covariance(cov: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Ledoit-Wolf–style constant-correlation shrinkage."""
    n = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std + 1e-12)
    target = np.eye(n)  # diagonal target
    shrunk_corr = (1 - shrinkage) * corr + shrinkage * target
    return np.outer(std, std) * shrunk_corr


def _hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Compute HRP weights from a covariance matrix."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    n = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    corr = cov / (np.outer(std, std) + 1e-12)
    corr = np.clip(corr, -1, 1)

    # Distance matrix for clustering
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Sorted leaf order from dendrogram
    from scipy.cluster.hierarchy import leaves_list
    order = list(leaves_list(link))

    # Recursive bisection
    weights = np.ones(n)
    _hrp_bisect(cov, order, weights)
    return weights / weights.sum()


def _cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    sub = cov[np.ix_(indices, indices)]
    inv_diag = 1.0 / (np.diag(sub) + 1e-12)
    w = inv_diag / inv_diag.sum()
    return float(w @ sub @ w)


def _hrp_bisect(cov: np.ndarray, order: list[int], weights: np.ndarray) -> None:
    if len(order) <= 1:
        return
    split = len(order) // 2
    left, right = order[:split], order[split:]
    var_l = _cluster_var(cov, left)
    var_r = _cluster_var(cov, right)
    alpha = 1 - var_l / (var_l + var_r + 1e-12)
    for i in left:
        weights[i] *= alpha
    for i in right:
        weights[i] *= (1 - alpha)
    _hrp_bisect(cov, left, weights)
    _hrp_bisect(cov, right, weights)


def _risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    n = cov.shape[0]
    target_rc = 1.0 / n

    def objective(w):
        sigma = np.sqrt(w @ cov @ w + 1e-12)
        rc = w * (cov @ w) / sigma
        return float(np.sum((rc / sigma - target_rc) ** 2))

    w0 = np.ones(n) / n
    result = minimize(
        objective, w0,
        bounds=[(1e-4, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        method="SLSQP", options={"ftol": 1e-10, "maxiter": 1000},
    )
    return result.x if result.success else w0


def _cvar_weights(cov: np.ndarray, alpha: float, n_scenarios: int) -> np.ndarray:
    import cvxpy as cp

    n = cov.shape[0]
    # Sample scenarios from covariance
    rng = np.random.default_rng(42)
    scenarios = rng.multivariate_normal(np.zeros(n), cov, size=n_scenarios)

    w = cp.Variable(n, nonneg=True)
    aux = cp.Variable(n_scenarios, nonneg=True)
    var_val = cp.Variable()

    port_ret = scenarios @ w
    cvar_expr = var_val + (1 / (alpha * n_scenarios)) * cp.sum(aux)
    constraints = [
        cp.sum(w) == 1,
        aux >= -port_ret - var_val,
    ]
    prob = cp.Problem(cp.Minimize(cvar_expr), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return np.array(w.value) if w.value is not None else np.ones(n) / n
