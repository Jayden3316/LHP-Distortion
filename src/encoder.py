"""Frozen ModernBERT encoder for prompt-response representations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.data import Turn1Dataset


def _ensure_torch_compile_compat() -> None:
    """Guard against torch<2.4 + Python 3.12 crashing on @torch.compile decorators in ModernBERT."""
    try:
        torch.compile(lambda x: x)
    except RuntimeError as exc:
        if "Dynamo is not supported on Python 3.12+" in str(exc):
            def _noop_compile(model=None, *args, **kwargs):
                if model is not None:
                    return model
                return lambda fn: fn

            torch.compile = _noop_compile  # type: ignore[assignment]
        else:
            raise


class FrozenModernBERTEncoder(nn.Module):
    """Frozen pretrained encoder (default: answerdotai/ModernBERT-base).

    All backbone parameters are frozen (`requires_grad = False`) and the backbone is
    permanently held in eval mode so online frozen forward passes and precomputed
    cached CLS activations are mathematically identical.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 512,
        pooling: str = "cls",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unsupported pooling={pooling!r}; expected 'cls' or 'mean'.")
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = torch.device(device)

        _ensure_torch_compile_compat()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Disable ModernBERT's torch.compile by default: variable-length batches trigger
        # Dynamo guard caches that creep VRAM and are unnecessary for frozen inference.
        kwargs = {"reference_compile": False} if "modernbert" in model_name.lower() else {}
        self.backbone = AutoModel.from_pretrained(model_name, **kwargs)
        self.backbone.to(self.device)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.hidden_size: int = int(self.backbone.config.hidden_size)

    def train(self, mode: bool = True) -> "FrozenModernBERTEncoder":
        """Keep the frozen backbone strictly in eval mode at all times."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            # .clone() detaches from the full [B, L, H] storage so unused token states can be freed immediately!
            return last_hidden_state[:, 0, :].contiguous().clone()
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (last_hidden_state * mask).sum(dim=1) / denom

    @torch.no_grad()
    def encode_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        batch_size: int = 32,
        output_device: str | torch.device | None = None,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """Encode `(prompt, response)` text pairs into frozen `[N, hidden_size]` features.

        Args:
            output_device: Where to accumulate batch outputs (defaults to `self.device`;
                set to `"cpu"` when encoding large datasets to keep GPU VRAM flat).
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have identical length.")
        target_dev = torch.device(output_device) if output_device is not None else self.device
        if len(prompts) == 0:
            return torch.empty((0, self.hidden_size), dtype=torch.float32, device=target_dev)

        self.backbone.eval()
        outputs: list[torch.Tensor] = []

        # Pick native CUDA autocast dtype: bf16 on Ampere+ (A100/H100), fp16 on Turing/Volta (T4/V100)
        amp_dtype: torch.dtype | None = None
        if self.device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        n_total = len(prompts)
        for batch_i, start in enumerate(range(0, n_total, batch_size), start=1):
            batch_p = [str(x) for x in prompts[start : start + batch_size]]
            batch_r = [str(y) for y in responses[start : start + batch_size]]
            enc = self.tokenizer(
                batch_p,
                batch_r,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = self.backbone(**enc)
            else:
                out = self.backbone(**enc)

            pooled = self._pool(out.last_hidden_state, enc["attention_mask"]).float()
            outputs.append(pooled.to(target_dev))
            del enc, out, pooled

            if show_progress and (batch_i % 50 == 0 or start + batch_size >= n_total):
                done = min(start + batch_size, n_total)
                print(f"  Encoded {done:,}/{n_total:,} unique (x, y) pairs...", flush=True)

        return torch.cat(outputs, dim=0)

    def forward(self, prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        """Online frozen forward pass for a single batch of `(prompt, response)` pairs."""
        return self.encode_pairs(prompts, responses, batch_size=max(1, len(prompts)))

    @torch.no_grad()
    def precompute_dataset_features(
        self,
        ds: Turn1Dataset,
        batch_size: int = 32,
        cache_path: str | Path | None = None,
    ) -> torch.Tensor:
        """Precompute frozen features `[num_unique_xy, hidden_size]` for all unique `(x, y)`."""
        if cache_path is not None and Path(cache_path).exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached.shape == (ds.num_unique_xy, self.hidden_size):
                return cached

        prompts = [ds.prompts[p_idx] for p_idx, _ in ds.unique_xy]
        responses = [ds.prompt_responses[p_idx][r_idx] for p_idx, r_idx in ds.unique_xy]
        # Accumulate outputs directly on CPU so GPU VRAM stays constant across all batches
        feats = self.encode_pairs(
            prompts,
            responses,
            batch_size=batch_size,
            output_device="cpu",
            show_progress=ds.num_unique_xy >= 1000,
        )

        if cache_path is not None:
            p = Path(cache_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feats, p)
        return feats
