"""Reference policy pi_ref computation from causal LLM completion log-probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def logprobs_to_probs(logprobs: np.ndarray) -> np.ndarray:
    """Convert raw completion log-probabilities to a valid simplex via log-sum-exp."""
    lp = np.asarray(logprobs, dtype=np.float64)
    if lp.ndim != 1 or lp.size == 0:
        raise ValueError("logprobs must be a non-empty 1D array.")
    if not np.all(np.isfinite(lp)):
        raise ValueError("logprobs must contain only finite values.")
    if lp.size == 1:
        return np.array([1.0], dtype=np.float64)
    shifted = lp - np.max(lp)
    log_z = np.log(np.sum(np.exp(shifted)))
    probs = np.exp(shifted - log_z)
    probs = probs / np.sum(probs)
    return probs


def ref_policy_diagnostics(probs: np.ndarray, collapse_threshold: float = 0.999) -> dict:
    """Compute entropy and collapse diagnostics for a discrete reference distribution."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("probs must be a non-empty 1D array.")
    pos = p[p > 0]
    entropy = float(-np.sum(pos * np.log(pos))) if pos.size > 0 else 0.0
    max_entropy = float(np.log(p.size)) if p.size > 1 else 0.0
    norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    max_prob = float(np.max(p))
    return {
        "num_candidates": int(p.size),
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "effective_support": float(np.exp(entropy)),
        "max_prob": max_prob,
        "min_prob": float(np.min(p)),
        "is_collapsed": bool(p.size > 1 and max_prob > collapse_threshold),
    }


def load_causal_lm(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
):
    """Load a causal LM (or saved PEFT/LoRA adapter directory) and tokenizer in eval mode."""
    dev = torch.device(device)
    if dtype is None:
        if dev.type == "cpu":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    adapter_cfg_path = Path(model_name) / "adapter_config.json"
    if adapter_cfg_path.is_file():
        import json
        from peft import PeftModel

        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_name = adapter_cfg["base_model_name_or_path"]
        tok_src = model_name if (Path(model_name) / "tokenizer_config.json").is_file() else base_name
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            base_model = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model.to(dev)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def compute_completion_logprobs(
    model,
    tokenizer,
    prompt: str,
    completions: Sequence[str],
    batch_size: int = 8,
    device: str | torch.device | None = None,
    max_length: int = 1024,
    length_normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sum_{t=1}^T log p(y_t | x, y_{<t}) for each completion y given prompt x.

    Tokenizes prompt and completion separately and concatenates token IDs to guarantee
    exact prompt-boundary masking without BPE boundary merge artifacts.

    Returns:
        (logprobs, token_lengths) as float64 and int64 1D numpy arrays.
    """
    if len(completions) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)

    dev = torch.device(device) if device is not None else next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    prompt_ids: list[int] = tokenizer.encode(prompt, add_special_tokens=True)
    if len(prompt_ids) == 0:
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if bos is not None:
            prompt_ids = [int(bos)]

    # Reserve room for prompt; keep at least 1 token of completion if prompt is very long
    if len(prompt_ids) >= max_length:
        prompt_ids = prompt_ids[-(max_length - 1) :]

    # Prefix completion with a separating newline/space only via explicit prompt formatting if caller wants;
    # here we tokenize completion directly without extra special tokens.
    full_seqs: list[list[int]] = []
    prompt_lens: list[int] = []
    comp_lens: list[int] = []

    max_comp_len = max(1, max_length - len(prompt_ids))
    for comp in completions:
        c_ids = tokenizer.encode(comp, add_special_tokens=False)[:max_comp_len]
        seq = prompt_ids + c_ids
        full_seqs.append(seq)
        prompt_lens.append(len(prompt_ids))
        comp_lens.append(len(c_ids))

    out_logprobs = np.zeros(len(completions), dtype=np.float64)
    out_lengths = np.array(comp_lens, dtype=np.int64)

    # Sort by sequence length before batching to minimize padding waste (huge VRAM win for 248K-vocab LLMs)
    order = sorted(range(len(completions)), key=lambda idx: len(full_seqs[idx]))

    # Low-VRAM path for standard CausalLM architectures (Qwen2/3/3.5, Llama, SmolLM2, Mistral),
    # including PEFT/LoRA-wrapped CausalLMs (where active LoRA layers live inside get_base_model()).
    # Runs backbone to get [B, L, H] hidden states, then projects ONLY completion tokens in 64-token
    # chunks via lm_head (~64 MB peak instead of multi-GB [B, L, vocab_size] logits + log_softmax).
    causal_lm = model.get_base_model() if hasattr(model, "peft_config") and hasattr(model, "get_base_model") else model
    base_model = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    cfg = getattr(causal_lm, "config", None)
    can_chunk_lm_head = (
        base_model is not None
        and lm_head is not None
        and getattr(cfg, "final_logit_softcapping", None) is None
        and getattr(cfg, "logit_scale", None) in (None, 1.0)
    )

    tok_chunk = 64
    for start in range(0, len(order), batch_size):
        batch_orig_idxs = order[start : start + batch_size]
        batch_seqs = [full_seqs[idx] for idx in batch_orig_idxs]
        batch_plens = [prompt_lens[idx] for idx in batch_orig_idxs]
        batch_clens = [comp_lens[idx] for idx in batch_orig_idxs]

        max_len = max(len(s) for s in batch_seqs)
        input_ids = torch.full((len(batch_seqs), max_len), pad_id, dtype=torch.long, device=dev)
        attention_mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long, device=dev)
        for i, seq in enumerate(batch_seqs):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=dev)
            attention_mask[i, :L] = 1

        if can_chunk_lm_head:
            base_out = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = base_out.last_hidden_state  # [B, L, H]
            logits_raw = None
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits_raw = out.logits  # [B, L, V]
            hidden = None

        for i, (orig_idx, p_len, c_len) in enumerate(zip(batch_orig_idxs, batch_plens, batch_clens)):
            if c_len == 0:
                val = 0.0
            else:
                if p_len == 0:
                    start_idx = 0
                    end_idx = max(0, c_len - 1)
                else:
                    start_idx = p_len - 1
                    end_idx = p_len + c_len - 1

                if end_idx <= start_idx:
                    val = 0.0
                else:
                    nll_sum = 0.0
                    for c_start in range(start_idx, end_idx, tok_chunk):
                        c_end = min(c_start + tok_chunk, end_idx)
                        if can_chunk_lm_head:
                            c_logits = lm_head(hidden[i, c_start:c_end, :]).float()
                        else:
                            c_logits = logits_raw[i, c_start:c_end, :].float()
                        c_targets = input_ids[i, c_start + 1 : c_end + 1]
                        nll_sum += float(F.cross_entropy(c_logits, c_targets, reduction="sum").item())
                        del c_logits
                    val = -nll_sum
                if length_normalize and c_len > 0:
                    val /= float(c_len)
            out_logprobs[orig_idx] = val

        if can_chunk_lm_head:
            del base_out, hidden, input_ids, attention_mask
        else:
            del out, logits_raw, input_ids, attention_mask

    return out_logprobs, out_lengths
