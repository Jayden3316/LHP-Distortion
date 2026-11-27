"""Population ("mythical user") Bradley-Terry reward model, beta_BT resampling, and Gibbs policy."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.user_bt import BTTrainHistory, bt_pairwise_loss


class MythicalUserBT(nn.Module):
    """Single shared linear Bradley-Terry reward model r_MLE(x, y) = <theta, psi(x, y)>."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.reward_head = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.reward_head.weight)

    def score(self, feat_xy: torch.Tensor) -> torch.Tensor:
        return self.reward_head(feat_xy).squeeze(-1)

    def forward_pairs(self, feat_pos: torch.Tensor, feat_neg: torch.Tensor) -> torch.Tensor:
        return self.score(feat_pos) - self.score(feat_neg)


class MythicalUserMLPBT(nn.Module):
    """Single shared 2-layer MLP Bradley-Terry reward model: hidden_dim -> bottleneck_dim -> 1."""

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 8) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("hidden_dim and bottleneck_dim must be positive.")
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.fc1 = nn.Linear(hidden_dim, bottleneck_dim, bias=True)
        self.fc2 = nn.Linear(bottleneck_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

    def score(self, feat_xy: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(feat_xy))).squeeze(-1)

    def forward_pairs(self, feat_pos: torch.Tensor, feat_neg: torch.Tensor) -> torch.Tensor:
        return self.score(feat_pos) - self.score(feat_neg)


def fit_mythical_bt(
    features_xy: torch.Tensor,
    pairs: list[tuple[int, int, int, int]],
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[MythicalUserBT, BTTrainHistory]:
    """Fit single shared-theta Bradley-Terry MLE reward model over frozen features."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features_xy.to(dev, dtype=torch.float32)
    model = MythicalUserBT(hidden_dim=int(feats.shape[1])).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_arr = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=dev)
    neg_arr = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=dev)
    n_pairs = len(pairs)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    accs: list[float] = []

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_pairs, generator=gen).to(dev)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            margins = model.forward_pairs(feats[pos_arr[idx]], feats[neg_arr[idx]])
            loss = bt_pairwise_loss(margins)
            loss.backward()
            optimizer.step()

            bs = int(idx.numel())
            total_loss += float(loss.item()) * bs
            total_correct += int((margins.detach() > 0).sum().item())

        losses.append(total_loss / n_pairs)
        accs.append(total_correct / n_pairs)

    model.eval()
    return model, BTTrainHistory(losses=losses, accuracies=accs)


def fit_mythical_mlp_bt(
    features_xy: torch.Tensor,
    pairs: list[tuple[int, int, int, int]],
    bottleneck_dim: int = 8,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[MythicalUserMLPBT, BTTrainHistory]:
    """Fit single shared 2-layer MLP Bradley-Terry reward model over frozen features."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features_xy.to(dev, dtype=torch.float32)
    model = MythicalUserMLPBT(hidden_dim=int(feats.shape[1]), bottleneck_dim=bottleneck_dim).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pos_arr = torch.tensor([p[2] for p in pairs], dtype=torch.long, device=dev)
    neg_arr = torch.tensor([p[3] for p in pairs], dtype=torch.long, device=dev)
    n_pairs = len(pairs)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    accs: list[float] = []

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_pairs, generator=gen).to(dev)
        total_loss = 0.0
        total_correct = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            margins = model.forward_pairs(feats[pos_arr[idx]], feats[neg_arr[idx]])
            loss = bt_pairwise_loss(margins)
            loss.backward()
            optimizer.step()

            bs = int(idx.numel())
            total_loss += float(loss.item()) * bs
            total_correct += int((margins.detach() > 0).sum().item())

        losses.append(total_loss / n_pairs)
        accs.append(total_correct / n_pairs)

    model.eval()
    return model, BTTrainHistory(losses=losses, accuracies=accs)


def fit_mythical_bt_soft(
    features_xy: torch.Tensor,
    pairs: list[tuple[int, int, int, int]],
    u_norm: np.ndarray,
    beta_bt: float,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[MythicalUserBT, BTTrainHistory]:
    """Fit shared MythicalUserBT against exact soft BT probabilities p_{iab} = sigma(beta_bt * du).

    Mathematically equals the exact expectation of `fit_mythical_bt` over infinitely many
    Bernoulli resamples on `pairs`, eliminating finite-sample coin-flip variance while
    preserving parametric linear-head function-approximation error.
    """
    if beta_bt < 0:
        raise ValueError("beta_bt must be non-negative.")
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features_xy.to(dev, dtype=torch.float32)
    model = MythicalUserBT(hidden_dim=int(feats.shape[1])).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    u_arr = np.asarray(u_norm, dtype=np.float64)
    u_idxs = np.array([p[0] for p in pairs], dtype=np.int64)
    a_idxs = np.array([p[2] for p in pairs], dtype=np.int64)
    b_idxs = np.array([p[3] for p in pairs], dtype=np.int64)
    du = beta_bt * (u_arr[u_idxs, a_idxs] - u_arr[u_idxs, b_idxs])
    p_soft_np = np.where(
        du >= 0,
        1.0 / (1.0 + np.exp(-np.clip(du, -500.0, 500.0))),
        np.exp(np.clip(du, -500.0, 500.0)) / (1.0 + np.exp(np.clip(du, -500.0, 500.0))),
    )

    pos_t = torch.tensor(a_idxs, dtype=torch.long, device=dev)
    neg_t = torch.tensor(b_idxs, dtype=torch.long, device=dev)
    p_soft_t = torch.tensor(p_soft_np, dtype=torch.float32, device=dev)
    n_pairs = len(pairs)

    losses: list[float] = []
    accs: list[float] = []
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n_pairs, device=dev)
        total_loss = 0.0
        total_agree = 0.0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            margins = model.forward_pairs(feats[pos_t[idx]], feats[neg_t[idx]])
            p_batch = p_soft_t[idx]
            loss = F.binary_cross_entropy_with_logits(margins, p_batch)
            loss.backward()
            optimizer.step()

            bs = int(idx.numel())
            total_loss += float(loss.item()) * bs
            # Expected pairwise accuracy against soft targets
            pred_pos = (margins.detach() > 0).float()
            total_agree += float((pred_pos * p_batch + (1.0 - pred_pos) * (1.0 - p_batch)).sum().item())

        losses.append(total_loss / n_pairs)
        accs.append(total_agree / n_pairs)

    model.eval()
    return model, BTTrainHistory(losses=losses, accuracies=accs)


def exact_population_query_bt_mle(
    u_query: np.ndarray,
    beta_bt: float,
) -> np.ndarray:
    """Solve exact tabular population BT-MLE for a single query x over its candidates A(x).

    Given user utility matrix `u_query` of shape `[num_users, num_candidates]`, computes
    exact population pairwise win probabilities:
        P[a, b] = mean_i sigmoid(beta_bt * (u_query[i, a] - u_query[i, b]))
    and minimizes -sum_{a, b} P[a, b] log sigmoid(r_a - r_b) over unrestricted r in R^m.

    Zero finite-sample noise and zero function-approximation error — isolates pure
    social-choice aggregation distortion (exact Theorem 6 setting).
    """
    U = np.asarray(u_query, dtype=np.float64)
    if U.ndim != 2 or U.shape[0] == 0 or U.shape[1] == 0:
        raise ValueError("u_query must be a non-empty 2D array [num_users, num_candidates].")
    if beta_bt < 0:
        raise ValueError("beta_bt must be non-negative.")
    m = U.shape[1]
    if m == 1 or beta_bt == 0.0:
        return np.zeros(m, dtype=np.float64)

    diff = beta_bt * (U[:, :, None] - U[:, None, :])  # [N, m, m]
    P = np.where(
        diff >= 0,
        1.0 / (1.0 + np.exp(-np.clip(diff, -500.0, 500.0))),
        np.exp(np.clip(diff, -500.0, 500.0)) / (1.0 + np.exp(np.clip(diff, -500.0, 500.0))),
    ).mean(axis=0)  # [m, m]

    def loss_and_grad(r: np.ndarray) -> tuple[float, np.ndarray]:
        dr = r[:, None] - r[None, :]
        sp = np.logaddexp(0.0, -dr)
        # Tiny mean-centering term fixes the additive shift Gauge freedom without biasing differences
        r_sum = float(np.sum(r))
        loss = float(np.sum(P * sp)) + 1e-8 * (r_sum ** 2)
        sig_neg = np.where(
            dr >= 0,
            np.exp(-np.clip(dr, -500.0, 500.0)) / (1.0 + np.exp(-np.clip(dr, -500.0, 500.0))),
            1.0 / (1.0 + np.exp(np.clip(dr, -500.0, 500.0))),
        )
        M = P * sig_neg
        grad = -M.sum(axis=1) + M.sum(axis=0) + 2e-8 * r_sum
        return loss, grad

    res = minimize(loss_and_grad, np.zeros(m, dtype=np.float64), jac=True, method="BFGS", options={"gtol": 1e-7})
    r_opt = res.x - float(np.mean(res.x))
    return r_opt


def resample_preferences(
    u_norm: np.ndarray,
    pairs: list[tuple[int, int, int, int]],
    beta_bt: float,
    seed: int = 42,
) -> list[tuple[int, int, int, int]]:
    """Resample pairwise preference orientations from normalized utilities u_norm in [0, 1].

    For each pair (u_idx, p_idx, xy_a, xy_b), flips or keeps orientation according to
    P(xy_a >_u xy_b) = sigmoid(beta_bt * (u_norm[u_idx, xy_a] - u_norm[u_idx, xy_b])).
    """
    if beta_bt < 0:
        raise ValueError("beta_bt must be non-negative.")
    rng = np.random.default_rng(seed)
    u_arr = np.asarray(u_norm, dtype=np.float64)

    u_idxs = np.array([p[0] for p in pairs], dtype=np.int64)
    a_idxs = np.array([p[2] for p in pairs], dtype=np.int64)
    b_idxs = np.array([p[3] for p in pairs], dtype=np.int64)

    logits = beta_bt * (u_arr[u_idxs, a_idxs] - u_arr[u_idxs, b_idxs])
    probs = np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -500.0, 500.0))),
        np.exp(np.clip(logits, -500.0, 500.0)) / (1.0 + np.exp(np.clip(logits, -500.0, 500.0))),
    )
    draws = rng.random(len(pairs)) < probs

    resampled: list[tuple[int, int, int, int]] = []
    for (u_idx, p_idx, xy_a, xy_b), a_wins in zip(pairs, draws):
        if a_wins:
            resampled.append((u_idx, p_idx, xy_a, xy_b))
        else:
            resampled.append((u_idx, p_idx, xy_b, xy_a))
    return resampled


def gibbs_policy(rewards: np.ndarray, pi_ref: np.ndarray, lam: float) -> np.ndarray:
    """Compute KL-regularized RLHF policy pi_lam(y) propto pi_ref(y) * exp(r(y) / lam).

    Handles extreme regimes cleanly:
    - lam == inf -> returns pi_ref
    - lam -> 0+ (or numerical overflow) -> puts mass on argmax(r) proportionally to pi_ref
    """
    r = np.asarray(rewards, dtype=np.float64)
    p0 = np.asarray(pi_ref, dtype=np.float64)
    if r.ndim != 1 or p0.ndim != 1 or r.shape != p0.shape:
        raise ValueError("rewards and pi_ref must be 1D arrays of equal length.")
    if lam < 0:
        raise ValueError("lam must be non-negative.")

    support = p0 > 0
    if not np.any(support):
        raise ValueError("pi_ref must have positive support.")
    if np.isinf(lam):
        return p0 / np.sum(p0)

    r_sup = r[support]
    p0_sup = p0[support]
    r_shifted = r_sup - np.max(r_sup)

    if lam == 0.0 or (np.min(r_shifted) < 0 and np.max(np.abs(r_shifted)) / lam > 700.0):
        is_max = np.isclose(r_shifted, 0.0, atol=1e-12)
        pi_sup = np.where(is_max, p0_sup, 0.0)
        pi_sup /= np.sum(pi_sup)
    else:
        log_w = np.log(p0_sup) + r_shifted / lam
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        pi_sup = w / np.sum(w)

    pi = np.zeros_like(p0, dtype=np.float64)
    pi[support] = pi_sup
    return pi
