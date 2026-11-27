"""Individual user-wise low-rank Bradley-Terry model and global affine normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder import FrozenModernBERTEncoder


class LowRankUserBT(nn.Module):
    """Bilinear low-rank Bradley-Terry utility model: s_i(x, y) = <w_i, W psi(x, y)>.

    - w_i in R^rank: purely user-ID embedding
    - W in R^{rank x hidden_dim}: trainable linear projection over frozen encoder features psi(x, y)
    """

    def __init__(self, num_users: int, hidden_dim: int, rank: int) -> None:
        super().__init__()
        if num_users <= 0 or hidden_dim <= 0 or rank <= 0:
            raise ValueError("num_users, hidden_dim, and rank must all be positive.")
        self.num_users = num_users
        self.hidden_dim = hidden_dim
        self.rank = rank

        self.user_emb = nn.Embedding(num_users, rank)
        self.proj = nn.Linear(hidden_dim, rank, bias=False)

        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.1)
        nn.init.xavier_uniform_(self.proj.weight)

    def project(self, feat_xy: torch.Tensor) -> torch.Tensor:
        """Map frozen features [..., hidden_dim] -> [..., rank]."""
        return self.proj(feat_xy)

    def score_from_features(self, user_idx: torch.Tensor, feat_xy: torch.Tensor) -> torch.Tensor:
        """Compute s_i(x, y) = <w_i, W psi(x, y)> for aligned batch tensors."""
        w_u = self.user_emb(user_idx)
        phi_xy = self.proj(feat_xy)
        return (w_u * phi_xy).sum(dim=-1)

    def score_all_users(self, feat_xy: torch.Tensor) -> torch.Tensor:
        """Compute full score matrix [num_users, num_xy] for a bank of frozen features."""
        phi_xy = self.proj(feat_xy)  # [M, rank]
        return self.user_emb.weight @ phi_xy.T  # [N, M]

    def forward_pairs(
        self,
        user_idx: torch.Tensor,
        feat_pos: torch.Tensor,
        feat_neg: torch.Tensor,
    ) -> torch.Tensor:
        """Return pairwise score margins s_i(x, y^+) - s_i(x, y^-)."""
        w_u = self.user_emb(user_idx)
        phi_diff = self.proj(feat_pos) - self.proj(feat_neg)
        return (w_u * phi_diff).sum(dim=-1)

    def forward_pairs_online(
        self,
        encoder: FrozenModernBERTEncoder,
        user_idx: torch.Tensor,
        prompts: list[str],
        pos_responses: list[str],
        neg_responses: list[str],
    ) -> torch.Tensor:
        """Compute pairwise margins via an online frozen encoder forward pass."""
        feat_pos = encoder(prompts, pos_responses)
        feat_neg = encoder(prompts, neg_responses)
        return self.forward_pairs(user_idx, feat_pos, feat_neg)


def bt_pairwise_loss(margins: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood under Bradley-Terry: mean(-log sigmoid(margin))."""
    return F.softplus(-margins).mean()


@dataclass
class BTTrainHistory:
    losses: list[float]
    accuracies: list[float]


def fit_user_bt(
    features_xy: torch.Tensor,
    pairs: list[tuple[int, int, int, int]],
    num_users: int,
    rank: int,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 1024,
    seed: int = 42,
    device: str | torch.device = "cpu",
    mean_centered_prior: bool = True,
    shared_weight_decay: float | None = None,
) -> tuple[LowRankUserBT, BTTrainHistory]:
    """Train LowRankUserBT on pairwise comparisons using precomputed frozen features.

    When `mean_centered_prior=True` (default), parameterizes each user's weight vector as
        w_i = w_bar + delta_i
    with hierarchical regularization: `shared_weight_decay` (defaults to `min(weight_decay, 0.1)`)
    on the shared projection `W` and population mean `w_bar`, and `weight_decay` on individual
    deviations `delta_i`. After training, `w_bar + delta_i` is folded back into
    `model.user_emb.weight` so the returned `LowRankUserBT` keeps the exact bilinear form
    `s_i(x, y) = <w_i, W psi(x, y)>` and unchanged state_dict.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features_xy.to(dev, dtype=torch.float32)
    hidden_dim = int(feats.shape[1])

    model = LowRankUserBT(num_users=num_users, hidden_dim=hidden_dim, rank=rank).to(dev)

    if mean_centered_prior:
        w_mean = nn.Parameter(torch.empty(rank, device=dev, dtype=torch.float32))
        nn.init.normal_(w_mean, mean=0.0, std=0.1)
        nn.init.zeros_(model.user_emb.weight)
        eff_shared_wd = min(weight_decay, 0.1) if shared_weight_decay is None else shared_weight_decay
        optimizer = torch.optim.AdamW(
            [
                {"params": [model.proj.weight, w_mean], "weight_decay": eff_shared_wd},
                {"params": [model.user_emb.weight], "weight_decay": weight_decay},
            ],
            lr=lr,
        )
    else:
        w_mean = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    u_arr = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
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
            u_b = u_arr[idx]
            f_pos = feats[pos_arr[idx]]
            f_neg = feats[neg_arr[idx]]

            optimizer.zero_grad()
            if w_mean is not None:
                w_u = w_mean + model.user_emb(u_b)
                phi_diff = model.proj(f_pos) - model.proj(f_neg)
                margins = (w_u * phi_diff).sum(dim=-1)
            else:
                margins = model.forward_pairs(u_b, f_pos, f_neg)
            loss = bt_pairwise_loss(margins)
            loss.backward()
            optimizer.step()

            bs = int(idx.numel())
            total_loss += float(loss.item()) * bs
            total_correct += int((margins.detach() > 0).sum().item())

        losses.append(total_loss / n_pairs)
        accs.append(total_correct / n_pairs)

    if w_mean is not None:
        with torch.no_grad():
            model.user_emb.weight.add_(w_mean.unsqueeze(0))

    model.eval()
    return model, BTTrainHistory(losses=losses, accuracies=accs)


def global_affine_normalize(scores: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Map raw unbounded utilities s into [0, 1] via global affine scaling.

    u_tilde = (s - s_min) / (s_max - s_min)

    This preserves every pairwise Bradley-Terry probability identically with implied
    natural inverse temperature beta_BT_nat = s_max - s_min:
        sigma(s_1 - s_2) == sigma(beta_BT_nat * (u_tilde_1 - u_tilde_2)).

    Constant-score edge case (s_max == s_min): returns 0.5 everywhere and beta_BT_nat = 0.0.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("scores must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite.")
    s_min = float(np.min(arr))
    s_max = float(np.max(arr))
    span = s_max - s_min
    if span < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64), 0.0, s_min, s_max
    u_norm = (arr - s_min) / span
    u_norm = np.clip(u_norm, 0.0, 1.0)
    return u_norm, span, s_min, s_max


def unit_range_normalize_per_query(
    scores: np.ndarray,
    prompt_global_xy_indices: list[list[int]],
) -> np.ndarray:
    """Per-user, per-query unit-range normalization into [0, 1] (Gölz et al., 2025, Footnote 3).

    For each user i and query x with candidate set A(x):
        u_tilde_i(y | x) = (s_i(x, y) - min_{y' in A(x)} s_i(x, y'))
                           / (max_{y' in A(x)} s_i(x, y') - min_{y' in A(x)} s_i(x, y'))
    Constant-score rows (max == min, e.g., |A(x)| == 1) map cleanly to 0.5 without NaN.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("scores must be a non-empty 2D [num_users, num_unique_xy] array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite.")

    out = np.full_like(arr, 0.5, dtype=np.float64)
    for g_idxs in prompt_global_xy_indices:
        if not g_idxs:
            continue
        sub = arr[:, g_idxs]  # [N, |A(x)|]
        s_min = np.min(sub, axis=1, keepdims=True)
        s_max = np.max(sub, axis=1, keepdims=True)
        span = s_max - s_min
        safe_span = np.where(span < 1e-12, 1.0, span)
        normed = np.where(span < 1e-12, 0.5, (sub - s_min) / safe_span)
        out[:, g_idxs] = np.clip(normed, 0.0, 1.0)
    return out


def per_query_normalize(
    scores: np.ndarray,
    prompt_global_xy_indices: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """BT-preserving per-query normalization into [0, 1].

    For each query x with candidate pool A(x):
      1. Remove each user's unidentifiable additive shift: d_i(y | x) = s_i(x, y) - min_{y' in A(x)} s_i(x, y').
      2. Divide ALL users on query x by a single shared query span:
             M(x) = max_{j in [N]} (max_{y' in A(x)} s_j(x, y') - min_{y' in A(x)} s_j(x, y')).
      3. Set u_tilde_i(y | x) = d_i(y | x) / M(x) in [0, 1] (0.5 if M(x) == 0).

    Properties:
      - Preserves EVERY user's within-query BT probabilities identically under one shared
        query-level temperature beta_BT(x) = M(x):
            sigma(s_i(x, y1) - s_i(x, y2)) == sigma(M(x) * (u_tilde_i(y1|x) - u_tilde_i(y2|x))).
      - Preserves relative preference intensities across users (indifferent users keep small
        utility span; passionate users span up to 1.0), unlike per-user unit-range scaling.
      - Isolates each query from cross-query outlier scales.

    Returns:
      (u_norm [num_users, num_unique_xy], query_spans [num_prompts]).
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("scores must be a non-empty 2D [num_users, num_unique_xy] array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("scores must be finite.")

    out = np.full_like(arr, 0.5, dtype=np.float64)
    query_spans = np.zeros(len(prompt_global_xy_indices), dtype=np.float64)

    for p_idx, g_idxs in enumerate(prompt_global_xy_indices):
        if not g_idxs:
            continue
        sub = arr[:, g_idxs]  # [N, |A(x)|]
        s_min = np.min(sub, axis=1, keepdims=True)
        shifted = sub - s_min  # >= 0, removes unidentifiable per-user-per-query constant c_i(x)
        max_span = float(np.max(shifted))
        query_spans[p_idx] = max_span
        if max_span < 1e-12:
            out[:, g_idxs] = 0.5
        else:
            out[:, g_idxs] = np.clip(shifted / max_span, 0.0, 1.0)
    return out, query_spans
