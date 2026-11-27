"""Check expressive single-user ("mythical user") baselines vs. UserBT on the exact Phase 3 split.

Tests whether the +13-15 pp held-out accuracy gap between Shared Linear BT (65%) and
LowRankUserBT (78-80.6%) is due to linear-probe underfitting or genuine user heterogeneity:
  1. Shared Nonlinear MLP (768 -> 512 -> 128 -> 1, ~460K params, >2x UserBT d=128 params)
  2. Tabular Single-User BT (one free scalar per unique (x, y), zero ModernBERT bottleneck)

Usage:
  python check_shared_mlp_baseline.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from run_phase3_user_bt import evaluate_margins, split_pairs_per_user
from src.data import load_turn1_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "data" / "community-alignment" / "community_alignment_en.csv"


class SharedMLPBT(nn.Module):
    """Expressive single-user nonlinear MLP reward model r(x, y) = MLP(psi(x, y))."""

    def __init__(self, hidden_dim: int = 768, width1: int = 512, width2: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, width1),
            nn.LayerNorm(width1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width1, width2),
            nn.LayerNorm(width2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width2, 1, bias=False),
        )

    def score(self, feat_xy: torch.Tensor) -> torch.Tensor:
        return self.net(feat_xy).squeeze(-1)

    def forward_pairs(self, feat_pos: torch.Tensor, feat_neg: torch.Tensor) -> torch.Tensor:
        return self.score(feat_pos) - self.score(feat_neg)


def fit_shared_mlp(
    feats: torch.Tensor,
    train_pairs: list[tuple[int, int, int, int]],
    val_pairs: list[tuple[int, int, int, int]],
    weight_decay: float,
    dropout: float,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 1024,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
) -> dict:
    torch.manual_seed(seed)
    model = SharedMLPBT(hidden_dim=int(feats.shape[1]), dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    feats_d = feats.to(device, dtype=torch.float32)
    pos_t = torch.tensor([p[2] for p in train_pairs], dtype=torch.long, device=device)
    neg_t = torch.tensor([p[3] for p in train_pairs], dtype=torch.long, device=device)
    n = len(train_pairs)

    best_val_loss = float("inf")
    best_metrics = {}
    best_epoch = 0

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            margins = model.forward_pairs(feats_d[pos_t[idx]], feats_d[neg_t[idx]])
            loss = F.softplus(-margins).mean()
            loss.backward()
            opt.step()

        if ep % 5 == 0 or ep == epochs:
            model.eval()
            _, vm = evaluate_margins(model, feats, val_pairs, device)
            if vm["loss"] < best_val_loss:
                best_val_loss = vm["loss"]
                _, tm = evaluate_margins(model, feats, train_pairs, device)
                best_metrics = {
                    "train_loss": tm["loss"],
                    "train_acc": tm["acc"],
                    "val_loss": vm["loss"],
                    "val_acc": vm["acc"],
                    "val_brier": vm["brier"],
                }
                best_epoch = ep

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": f"Shared MLP (drop={dropout})",
        "params": int(n_params),
        "weight_decay": float(weight_decay),
        "dropout": float(dropout),
        "best_epoch": int(best_epoch),
        **{k: round(float(v), 4) for k, v in best_metrics.items()},
    }


def fit_tabular_single_user_bt(
    num_unique_xy: int,
    train_pairs: list[tuple[int, int, int, int]],
    val_pairs: list[tuple[int, int, int, int]],
    weight_decay: float,
    lr: float = 5e-2,
    epochs: int = 80,
    batch_size: int = 4096,
    seed: int = 42,
) -> dict:
    """Unrestricted scalar r[xy] per unique (x, y) pair (zero ModernBERT bottleneck, no user IDs)."""
    torch.manual_seed(seed)
    emb = nn.Embedding(num_unique_xy, 1)
    nn.init.zeros_(emb.weight)
    opt = torch.optim.AdamW(emb.parameters(), lr=lr, weight_decay=weight_decay)

    tr_pos = torch.tensor([p[2] for p in train_pairs], dtype=torch.long)
    tr_neg = torch.tensor([p[3] for p in train_pairs], dtype=torch.long)
    va_pos = torch.tensor([p[2] for p in val_pairs], dtype=torch.long)
    va_neg = torch.tensor([p[3] for p in val_pairs], dtype=torch.long)
    n = len(train_pairs)

    best_val_loss = float("inf")
    best_row = {}
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        for s in range(0, n, batch_size):
            idx = perm[s : s + batch_size]
            opt.zero_grad()
            margins = (emb(tr_pos[idx]) - emb(tr_neg[idx])).squeeze(-1)
            loss = F.softplus(-margins).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            va_m = (emb(va_pos) - emb(va_neg)).squeeze(-1)
            va_loss = float(F.softplus(-va_m).mean().item())
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                tr_m = (emb(tr_pos) - emb(tr_neg)).squeeze(-1)
                va_probs = torch.sigmoid(va_m)
                best_row = {
                    "model": "Shared Tabular BT (no encoder)",
                    "params": int(num_unique_xy),
                    "weight_decay": float(weight_decay),
                    "dropout": 0.0,
                    "best_epoch": int(ep),
                    "train_loss": round(float(F.softplus(-tr_m).mean().item()), 4),
                    "train_acc": round(float((tr_m > 0).float().mean().item()), 4),
                    "val_loss": round(va_loss, 4),
                    "val_acc": round(float((va_m > 0).float().mean().item()), 4),
                    "val_brier": round(float(((1.0 - va_probs) ** 2).mean().item()), 4),
                }
    return best_row


def main() -> None:
    t0 = time.perf_counter()
    ds = load_turn1_dataset(DEFAULT_CSV, min_user_prefs_gt=20)
    feats = torch.load(ROOT / "results" / "phase3" / "modernbert_cls_features.pt", map_location="cpu", weights_only=True).float()
    train_pairs, val_pairs = split_pairs_per_user(ds.pairs, val_frac=0.2, seed=42)
    print(f"Exact Phase 3 split: train={len(train_pairs):,}, val={len(val_pairs):,}, unique_xy={ds.num_unique_xy:,}")

    rows = []
    for dropout, wd in [(0.1, 1e-3), (0.2, 1e-1), (0.3, 1.0)]:
        r = fit_shared_mlp(feats, train_pairs, val_pairs, weight_decay=wd, dropout=dropout, epochs=25, batch_size=2048)
        rows.append(r)
        print(
            f"  {r['model']:26s} wd={wd:5.3f} (ep={r['best_epoch']:2d}) | "
            f"train_acc={r['train_acc']:.4f} | val_acc={r['val_acc']:.4f}, val_loss={r['val_loss']:.4f}, val_brier={r['val_brier']:.4f}",
            flush=True,
        )

    for wd in (1e-2, 1e-1, 1.0):
        r = fit_tabular_single_user_bt(ds.num_unique_xy, train_pairs, val_pairs, weight_decay=wd)
        rows.append(r)
        print(
            f"  {r['model']:26s} wd={wd:5.3f} (ep={r['best_epoch']:2d}) | "
            f"train_acc={r['train_acc']:.4f} | val_acc={r['val_acc']:.4f}, val_loss={r['val_loss']:.4f}, val_brier={r['val_brier']:.4f}",
            flush=True,
        )

    out_dir = ROOT / "results" / "phase3"
    df = pd.DataFrame(rows)
    out_csv = out_dir / "shared_expressive_baselines.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved expressive shared baselines to {out_csv} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
