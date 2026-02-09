from __future__ import annotations

import numpy as np
import pandas as pd


def _align(ff25_excess: pd.DataFrame, ff3_factors: pd.DataFrame):
    idx = ff25_excess.index.intersection(ff3_factors.index)
    R = ff25_excess.loc[idx]
    F = ff3_factors.loc[idx]
    return idx, R, F


def moments_g(R: pd.DataFrame, F: pd.DataFrame, b: pd.Series) -> pd.Series:
    """
    g(b) = (1/T) sum_t M_t(b) * R_t,  où M_t(b) = 1 - f_t' b
    Returns Series length N (assets).
    """
    Fm = F[b.index]
    m = 1.0 - (Fm.to_numpy() @ b.to_numpy())  # (T,)
    g = (R.to_numpy() * m.reshape(-1, 1)).mean(axis=0)  # (N,)
    return pd.Series(g, index=R.columns, name="g")


def adversary_h(g: pd.Series, eps: float = 1e-12) -> pd.Series:
    """
    h*(b) = g / ||g|| (L2), the worst-case portfolio direction.
    """
    ng = float(np.linalg.norm(g.to_numpy()))
    if ng < eps:
        return pd.Series(np.zeros(len(g)), index=g.index, name="h")
    return pd.Series(g.to_numpy() / ng, index=g.index, name="h")


def fit_minimax_b2(
    ff25_excess: pd.DataFrame,
    ff3_factors: pd.DataFrame,
    b0: pd.Series,
    n_steps: int = 500,
    lr: float = 0.5,
    tol: float = 1e-10,
    eps: float = 1e-12,
) -> dict:
    """
    B2: explicit alternating game (no DL):
      - given b, compute g(b)
      - adversary picks h = g/||g||
      - update b to reduce ||g(b)||

    Key simplification (linear SDF in factors):
      g(b) = mu_R - C b, where C_{j,k} = E[R_j * f_k]
      so Jacobian wrt b is constant: d g / d b = -C

    We do gradient descent on L(b)=||g(b)|| with:
      grad = - C' g / ||g||  (with eps for stability)

    Returns dict with:
      b, g, h, history(DataFrame)
    """
    idx, R, F = _align(ff25_excess, ff3_factors)
    F = F[b0.index]  # ensure column order matches b

    # Precompute C = E[R * f']  => N×K
    # C[j,k] = mean_t R_{t,j} * F_{t,k}
    Rm = R.to_numpy()
    Fm = F.to_numpy()
    C = (Rm[:, :, None] * Fm[:, None, :]).mean(axis=0)  # (N,K)

    b = b0.copy().astype(float)

    hist = []
    for step in range(n_steps):
        g = moments_g(R, F, b)
        ng = float(np.linalg.norm(g.to_numpy()))
        h = adversary_h(g, eps=eps)

        hist.append(
            {
                "step": step,
                "norm_g": ng,
                "b_Mkt-RF": float(b.get("Mkt-RF", np.nan)),
                "b_SMB": float(b.get("SMB", np.nan)),
                "b_HML": float(b.get("HML", np.nan)),
            }
        )

        if ng < tol:
            break

        # grad_b = - C' g / ||g||
        grad = -(C.T @ g.to_numpy()) / max(ng, eps)  # (K,)
        b = pd.Series(b.to_numpy() - lr * grad, index=b.index, name="b")

    # final objects
    g_final = moments_g(R, F, b)
    h_final = adversary_h(g_final, eps=eps)

    return {
        "index": idx,
        "b": b,
        "g": g_final,
        "h": h_final,
        "history": pd.DataFrame(hist),
    }


def top_violations(g: pd.Series, k: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"g": g, "abs_g": g.abs()})
    return df.sort_values("abs_g", ascending=False).head(k)