# src/ml_in_finance_ensae/bench_ff3.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _ols_with_se(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    OLS (no robust SE). Returns (beta, se_beta).
    X must include intercept column if desired.
    """
    n, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    resid = y - X @ beta
    dof = n - k
    if dof <= 0:
        raise ValueError(f"Not enough observations for OLS (n={n}, k={k}).")

    sigma2 = (resid @ resid) / dof
    cov_beta = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov_beta))
    return beta, se


def ff3_time_series_benchmark(
    ff25_excess: pd.DataFrame,
    ff3_factors: pd.DataFrame,
) -> dict:
    """
    Runs: R^e_i,t = alpha_i + beta_i' f_t + eps_i,t for each FF25 portfolio.

    Inputs:
      ff25_excess: T×25 DataFrame (decimals)
      ff3_factors: T×3 DataFrame with columns ["Mkt-RF","SMB","HML"] (decimals)

    Returns dict with:
      - table: DataFrame indexed by portfolio with alpha, t_alpha, betas
      - summary: dict with rms_alpha_m, rms_alpha_ann, share_|t|>2, n_obs
    """
    # Align dates
    idx = ff25_excess.index.intersection(ff3_factors.index)
    R = ff25_excess.loc[idx].copy()
    F = ff3_factors.loc[idx].copy()

    # Build X = [1, factors]
    X = np.column_stack([np.ones(len(idx)), F.to_numpy()])
    factor_cols = list(F.columns)

    rows = []
    for p in R.columns:
        y = R[p].to_numpy()
        beta, se = _ols_with_se(y, X)

        alpha = float(beta[0])
        se_alpha = float(se[0])
        t_alpha = alpha / se_alpha if se_alpha > 0 else np.nan

        row = {
            "alpha_m": alpha,
            "t_alpha": t_alpha,
        }
        # betas in same order as factor_cols
        for j, c in enumerate(factor_cols, start=1):
            row[f"beta_{c}"] = float(beta[j])
        rows.append((p, row))

    table = pd.DataFrame({k: v for k, v in rows}).T
    table.index.name = "portfolio"

    # Summary stats
    alphas = table["alpha_m"].to_numpy()
    rms_alpha_m = float(np.sqrt(np.mean(alphas**2)))
    share_sig = float(np.mean(np.abs(table["t_alpha"].to_numpy()) > 2.0))

    summary = {
        "n_obs": int(len(idx)),
        "start": idx.min(),
        "end": idx.max(),
        "rms_alpha_m": rms_alpha_m,
        "rms_alpha_ann": float(12.0 * rms_alpha_m),
        "share_|t_alpha|>2": share_sig,
    }

    return {"table": table, "summary": summary}
