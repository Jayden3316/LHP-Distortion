"""KL-constrained social welfare solver, empirical distortion, and Theorem 6 plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq


def kl_divergence(pi: np.ndarray, pi_ref: np.ndarray) -> float:
    """Compute exact D_KL(pi || pi_ref) in nats for discrete distributions."""
    p = np.asarray(pi, dtype=np.float64)
    q = np.asarray(pi_ref, dtype=np.float64)
    if p.ndim != 1 or q.ndim != 1 or p.shape != q.shape:
        raise ValueError("pi and pi_ref must be 1D arrays of equal length.")
    pos = p > 0
    if np.any(q[pos] <= 0):
        return float("inf")
    val = float(np.sum(p[pos] * (np.log(p[pos]) - np.log(q[pos]))))
    return max(0.0, val)


def _tilted_policy(u: np.ndarray, pi_ref: np.ndarray, alpha: float) -> np.ndarray:
    """Compute pi_alpha(y) propto pi_ref(y) * exp(alpha * u(y)) stably for alpha >= 0.

    Subtracting max(u) before multiplying by alpha cancels out upon normalization and
    prevents float64 overflow in exp(alpha * u) when alpha is large.
    """
    support = pi_ref > 0
    u_sup = u[support]
    p0_sup = pi_ref[support]
    u_shifted = u_sup - np.max(u_sup)

    if alpha == 0.0:
        pi_sup = p0_sup / np.sum(p0_sup)
    else:
        log_w = np.log(p0_sup) + alpha * u_shifted
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        pi_sup = w / np.sum(w)

    pi = np.zeros_like(pi_ref, dtype=np.float64)
    pi[support] = pi_sup
    return pi


def solve_kl_ball_max_util(
    avg_util: np.ndarray,
    pi_ref: np.ndarray,
    tau: float,
    tol: float = 1e-11,
) -> tuple[np.ndarray, float]:
    """Exactly solve max_{pi in B_tau(pi_ref)} E_{y ~ pi}[avg_util(y)] via 1D root-finding.

    By convex duality (Gölz et al., 2025, Appendix E.3-E.4), the unique maximizer has the
    exponential-tilting form pi*(y) propto pi_ref(y) * exp(alpha * avg_util(y)) for a unique
    alpha >= 0 such that D_KL(pi_alpha || pi_ref) = tau (unless tau >= tau_max = -log p_opt_mass,
    which is the exact minimum KL divergence needed to put all mass on argmax avg_util).
    """
    u = np.asarray(avg_util, dtype=np.float64)
    p0 = np.asarray(pi_ref, dtype=np.float64)
    if u.ndim != 1 or p0.ndim != 1 or u.shape != p0.shape:
        raise ValueError("avg_util and pi_ref must be 1D arrays of equal length.")
    if tau < -1e-12:
        raise ValueError(f"tau must be non-negative, got {tau}.")
    tau = max(0.0, float(tau))

    p0 = p0 / np.sum(p0)
    support = p0 > 0
    u_sup = u[support]
    p0_sup = p0[support]

    # Boundary 1: zero KL budget or constant utility on support
    if tau <= 1e-15 or float(np.max(u_sup) - np.min(u_sup)) <= 1e-15:
        return p0.copy(), float(np.dot(p0, u))

    # Boundary 2: KL budget >= minimum KL needed to put all mass on M* = argmax(avg_util).
    # For pi_opt(y) = pi_ref(y) / p_opt_mass on M*, D_KL(pi_opt || pi_ref) = -log(p_opt_mass).
    u_max = float(np.max(u_sup))
    is_opt = np.isclose(u_sup, u_max, atol=1e-12, rtol=0.0)
    p_opt_mass = float(np.sum(p0_sup[is_opt]))
    tau_max = float(-np.log(p_opt_mass))

    if tau >= tau_max - 1e-12:
        pi_opt = np.zeros_like(p0, dtype=np.float64)
        pi_opt_sup = np.where(is_opt, p0_sup, 0.0)
        pi_opt_sup /= np.sum(pi_opt_sup)
        pi_opt[support] = pi_opt_sup
        return pi_opt, float(np.dot(pi_opt, u))

    # Bracket alpha_hi such that D_KL(pi_{alpha_hi} || pi_ref) >= tau
    def kl_gap(alpha: float) -> float:
        return kl_divergence(_tilted_policy(u, p0, alpha), p0) - tau

    u_span = float(np.max(u_sup) - np.min(u_sup))
    alpha_hi = 1.0 / u_span
    for _ in range(80):
        if kl_gap(alpha_hi) >= 0.0:
            break
        alpha_hi *= 2.0

    alpha_star = float(brentq(kl_gap, 0.0, alpha_hi, xtol=tol, rtol=tol, maxiter=200))
    pi_star = _tilted_policy(u, p0, alpha_star)
    return pi_star, float(np.dot(pi_star, u))


def compute_distortion(
    avg_util: np.ndarray,
    pi: np.ndarray,
    pi_ref: np.ndarray,
) -> dict[str, float]:
    """Compute per-query KL-constrained distortion for policy pi against pi_ref.

    Here `pi` is the aligned model's completion distribution over A(x) (either the
    closed-form KL-regularized RLHF policy pi_lam or the DPO policy pi_DPO), and
    `AvgUtil(pi) = E_{y ~ pi}[avg_util(y)]`.

    Returns dict with keys: tau, policy_util, opt_util, distortion.
    """
    u = np.asarray(avg_util, dtype=np.float64)
    p = np.asarray(pi, dtype=np.float64)
    p0 = np.asarray(pi_ref, dtype=np.float64)
    p = p / np.sum(p)
    p0 = p0 / np.sum(p0)

    tau = kl_divergence(p, p0)
    _, opt_util = solve_kl_ball_max_util(u, p0, tau)
    policy_util = float(np.dot(p, u))

    if policy_util <= 0.0:
        dist = 1.0 if opt_util <= 0.0 else float("inf")
    else:
        dist = float(opt_util / policy_util)

    return {
        "tau": float(tau),
        "policy_util": policy_util,
        "opt_util": float(opt_util),
        "distortion": dist,
    }


def theorem6_peak_distortion(beta_bt: float) -> float:
    """Tight worst-case peak distortion of Gölz et al. (2025) Theorem 6's 3-alternative instance.

    Taking the tight threshold delta/(1-delta) -> (sigma(beta) - sigma(beta(1-u_a))) / (sigma(beta) - 1/2)
    as u_a -> 0+ yields the closed-form constant-free limit:
        R(beta) = (sigma(beta) - 1/2) / (beta * sigma'(beta)) = sinh(beta) / beta = Theta(e^beta / beta).
    """
    if beta_bt < 0:
        raise ValueError("beta_bt must be non-negative.")
    if beta_bt < 1e-6:
        return 1.0 + (beta_bt**2) / 6.0
    if beta_bt > 700.0:
        return float("inf")
    return float(np.sinh(beta_bt) / beta_bt)


def _binary_kl_inverse(tau: float) -> float:
    """Solve q*log(2q) + (1-q)*log(2(1-q)) = tau for q in [0.5, 1.0], for tau in [0, log 2]."""
    if tau <= 1e-15:
        return 0.5
    ln2 = float(np.log(2.0))
    if tau >= ln2 - 1e-12:
        return 1.0

    def gap(q: float) -> float:
        return q * np.log(2.0 * q) + (1.0 - q) * np.log(2.0 * (1.0 - q)) - tau

    return float(brentq(gap, 0.5, 1.0 - 1e-15, xtol=1e-12, rtol=1e-12))


def theorem6_distortion_curve(tau_values: np.ndarray, beta_bt: float) -> np.ndarray:
    """Exact theoretical distortion vs. tau for the Theorem 6 construction (pi_ref = ((1-eps)/2, (1-eps)/2, eps)).

    - For tau in [0, log 2]:
        dist_Thm6(tau; beta) = (q(tau) * R(beta) + (1 - q(tau))) / ((1 - q(tau)) * R(beta) + q(tau))
      where q(tau) in [0.5, 1] solves D_KL((q, 1-q) || (1/2, 1/2)) = tau and R(beta) = sinh(beta) / beta.
    - For tau >= log 2:
        dist_Thm6(tau; beta) = R(beta) = sinh(beta) / beta.
    """
    taus = np.asarray(tau_values, dtype=np.float64)
    R = theorem6_peak_distortion(beta_bt)
    out = np.empty_like(taus, dtype=np.float64)
    ln2 = float(np.log(2.0))

    for i, t in enumerate(taus):
        if t <= 1e-15:
            out[i] = 1.0
        elif t >= ln2 - 1e-12:
            out[i] = R
        else:
            q = _binary_kl_inverse(float(t))
            if np.isinf(R):
                out[i] = q / max(1.0 - q, 1e-15)
            else:
                out[i] = (q * R + (1.0 - q)) / ((1.0 - q) * R + q)
    return out


def plot_distortion_panels(
    results_df: pd.DataFrame,
    save_path: str | Path,
    title: str = "Empirical KL-Constrained Distortion vs. Theoretical Theorem 6 Curve",
) -> Path:
    """Generate multi-panel scatter plot: tau (x-axis) vs. empirical distortion + Theorem 6 curve."""
    required = {"beta_bt", "tau", "distortion"}
    missing = required - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    betas = sorted(results_df["beta_bt"].unique())
    n_panels = len(betas)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, n_panels, figsize=(4.8 * n_panels, 4.4), dpi=160, squeeze=False)
    axes_flat = axes[0]

    for ax, beta in zip(axes_flat, betas):
        sub = results_df[results_df["beta_bt"] == beta]
        finite = sub[np.isfinite(sub["distortion"])]
        ax.scatter(
            finite["tau"],
            finite["distortion"],
            s=16,
            alpha=0.50,
            color="#2563eb",
            edgecolors="none",
            label="Empirical RLHF (per query)",
        )
        max_tau = max(1.0, float(finite["tau"].max()) if len(finite) > 0 else 1.0)
        tau_grid = np.concatenate([np.linspace(0.0, min(np.log(2.0), max_tau), 120), np.linspace(min(np.log(2.0), max_tau), max_tau, 60)])
        tau_grid = np.unique(tau_grid)
        th_vals = theorem6_distortion_curve(tau_grid, float(beta))
        peak = theorem6_peak_distortion(float(beta))
        ax.plot(
            tau_grid,
            th_vals,
            color="#dc2626",
            linewidth=2.0,
            label=f"Theorem 6 (peak={peak:.2f})",
        )
        ax.axhline(1.0, color="#94a3b8", linestyle=":", linewidth=1.1)

        if np.isfinite(peak) and peak > 8.0:
            ax.set_yscale("log")
        ax.set_title(rf"$\beta_{{\mathrm{{BT}}}} = {beta:.2g}$", fontsize=11.5, fontweight="bold")
        ax.set_xlabel(r"Realized KL radius $\tau(x) = D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{ref}})$")
        ax.set_ylabel("Distortion")
        ax.legend(fontsize=8, frameon=True)

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
