# src/ml_in_finance_ensae/sdf_linear.py
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_lambda_from_time_series_betas(
    ff25_excess: pd.DataFrame,
    ff3_factors: pd.DataFrame,
) -> dict:
    """
    2-pass style (but simple, non-FamaMacBeth):
    1) time-series: for each asset i, estimate beta_i in R^e_i = a_i + beta_i' f + eps
    2) cross-section: mean(R^e_i) = beta_i' lambda + error

    Returns:
      betas: (N×K) DataFrame
      lambda_: (K,) Series
      mu: (N,) Series of mean excess returns
      cs_resid: (N,) Series
    """
    # Align
    idx = ff25_excess.index.intersection(ff3_factors.index)
    R = ff25_excess.loc[idx]
    F = ff3_factors.loc[idx]

    T = len(idx)
    if T < 10:
        raise ValueError("Too few observations after alignment.")

    # Time-series betas (with intercept)
    X = np.column_stack([np.ones(T), F.to_numpy()])  # T×(1+K)
    XtX_inv = np.linalg.inv(X.T @ X)
    Xp = XtX_inv @ X.T  # (1+K)×T, common across assets

    # For each asset: coeffs = Xp @ y
    Y = R.to_numpy()  # T×N
    B = (Xp @ Y).T  # N×(1+K)

    # Keep only betas (drop intercept)
    K = F.shape[1]
    betas = pd.DataFrame(B[:, 1:1+K], index=R.columns, columns=F.columns)

    # Cross-section: mu = betas @ lambda + u
    mu = R.mean(axis=0)  # N
    Z = betas.to_numpy()  # N×K
    # OLS lambda = (Z'Z)^-1 Z' mu
    lambda_hat = np.linalg.solve(Z.T @ Z, Z.T @ mu.to_numpy())
    lambda_ = pd.Series(lambda_hat, index=F.columns, name="lambda")

    cs_resid = mu - betas @ lambda_

    return {
        "betas": betas,
        "lambda": lambda_,
        "mu": mu,
        "cs_resid": cs_resid,
        "index": idx,
    }


def b_from_lambda(ff3_factors: pd.DataFrame, lambda_: pd.Series) -> pd.Series:
    """
    Given lambda (prices of risk) and factor covariance, build b for SDF:
      M_t = 1 - b' f_t
    Standard linear pricing link: b = Sigma_f^{-1} * lambda
    """
    # Align columns
    F = ff3_factors[lambda_.index]
    Sigma_f = F.cov().to_numpy()
    b = np.linalg.solve(Sigma_f, lambda_.to_numpy())
    return pd.Series(b, index=lambda_.index, name="b")


def sdf_series(ff3_factors: pd.DataFrame, b: pd.Series) -> pd.Series:
    """
    Compute M_t = 1 - b' f_t (a=1 normalization).
    """
    F = ff3_factors[b.index]
    m = 1.0 - (F.to_numpy() @ b.to_numpy())
    return pd.Series(m, index=F.index, name="M")


def moment_vector(ff25_excess: pd.DataFrame, m: pd.Series) -> pd.Series:
    """
    Compute g = (1/T) sum_t M_t * R^e_t  (vector length N).
    """
    idx = ff25_excess.index.intersection(m.index)
    R = ff25_excess.loc[idx]
    mm = m.loc[idx].to_numpy().reshape(-1, 1)  # T×1
    g = (R.to_numpy() * mm).mean(axis=0)  # N,
    return pd.Series(g, index=R.columns, name="g")


def rms(x: pd.Series) -> float:
    v = x.to_numpy()
    return float(np.sqrt(np.mean(v * v)))


def top_violations(g: pd.Series, k: int = 5) -> pd.DataFrame:
    out = pd.DataFrame({"g": g, "abs_g": g.abs()})
    return out.sort_values("abs_g", ascending=False).head(k)
