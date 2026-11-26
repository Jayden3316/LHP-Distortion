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
) -> tuple[LowRankUserBT, BTTrainHistory]:
    """Train LowRankUserBT on pairwise comparisons using precomputed frozen features."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    feats = features_xy.to(dev, dtype=torch.float32)
    hidden_dim = int(feats.shape[1])

    model = LowRankUserBT(num_users=num_users, hidden_dim=hidden_dim, rank=rank).to(dev)
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
            margins = model.forward_pairs(u_b, f_pos, f_neg)
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
