"""Phase 4B: Direct Preference Optimization (DPO) via TRL + PEFT LoRA and KL-distortion evaluation."""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from src.data import Turn1Dataset
from src.distortion import compute_distortion
from src.ref_policy import (
    compute_completion_logprobs,
    load_causal_lm,
    logprobs_to_probs,
    ref_policy_diagnostics,
)


def build_dpo_preference_dataset(
    ds: Turn1Dataset,
    pairs: Sequence[tuple[int, int, int, int]],
    max_pairs: int | None = None,
    seed: int = 42,
) -> Dataset:
    """Convert `(user_idx, prompt_idx, win_xy, lose_xy)` tuples into a TRL DPO Dataset.

    Returns a HuggingFace `Dataset` with standard TRL columns: `prompt`, `chosen`, `rejected`.
    """
    if len(pairs) == 0:
        raise ValueError("pairs must be non-empty.")

    pair_list = list(pairs)
    if max_pairs is not None and len(pair_list) > max_pairs:
        rng = np.random.default_rng(seed)
        chosen_idxs = rng.choice(len(pair_list), size=max_pairs, replace=False)
        pair_list = [pair_list[int(i)] for i in chosen_idxs]

    prompts: list[str] = []
    chosens: list[str] = []
    rejecteds: list[str] = []
    for _u_idx, p_idx, win_xy, lose_xy in pair_list:
        p_txt, chosen_txt = ds.get_xy_text(int(win_xy))
        _, rejected_txt = ds.get_xy_text(int(lose_xy))
        prompts.append(p_txt)
        chosens.append(chosen_txt)
        rejecteds.append(rejected_txt)

    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


def infer_lora_target_modules(model: torch.nn.Module) -> list[str]:
    """Infer clean LoRA target linear layer names across Llama/SmolLM2/Qwen2/Qwen3/Qwen3.5."""
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    present_leaf_names = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
    }
    matched = [name for name in preferred if name in present_leaf_names]
    if matched:
        return matched
    # Fallback: all linear layers except lm_head
    fallback = sorted(n for n in present_leaf_names if n != "lm_head")
    if not fallback:
        raise ValueError("No Linear modules found for LoRA injection.")
    return fallback


def train_dpo_model(
    model_name: str,
    train_dataset: Dataset,
    output_dir: Path,
    dpo_beta: float = 0.1,
    learning_rate: float = 5e-5,
    num_train_epochs: float = 1.0,
    max_steps: int = -1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 512,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    gradient_checkpointing: bool = False,
    seed: int = 42,
    device: str | torch.device = "cpu",
    save_adapter: bool = True,
    logging_steps: int = 20,
) -> tuple[torch.nn.Module, object, dict]:
    """Train a DPO policy using `trl.DPOTrainer` (with optional PEFT LoRA).

    When `use_lora=True`, passes `ref_model=None` + `peft_config=LoraConfig(...)` so TRL
    computes reference log-probs by temporarily disabling LoRA adapters (no second model
    copy in VRAM — essential for T4 15GB GPUs and fast on A100/H100).
    """
    t0 = time.perf_counter()
    dev = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model on the target device (fp32 on CPU, bf16 on Ampere+, fp16 on T4)
    model, tokenizer = load_causal_lm(model_name, device=dev)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False

    peft_config = None
    if use_lora:
        target_modules = infer_lora_target_modules(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
    else:
        target_modules = []

    use_bf16 = dev.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = dev.type == "cuda" and not use_bf16

    # Build DPOConfig safely across TRL 0.12+ / 1.x versions
    cfg_fields = set(DPOConfig.__dataclass_fields__.keys())
    cfg_kwargs: dict = {
        "output_dir": str(output_dir),
        "beta": float(dpo_beta),
        "loss_type": "sigmoid",
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "max_steps": int(max_steps),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "max_length": int(max_length),
        "logging_steps": int(logging_steps),
        "save_strategy": "no",
        "report_to": "none",
        "seed": int(seed),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "remove_unused_columns": False,
    }
    if "activation_checkpointing" in cfg_fields:
        cfg_kwargs["activation_checkpointing"] = bool(gradient_checkpointing)
    else:
        cfg_kwargs["gradient_checkpointing"] = bool(gradient_checkpointing)
    if "data_seed" in cfg_fields:
        cfg_kwargs["data_seed"] = int(seed)
    if "use_cpu" in cfg_fields and dev.type == "cpu":
        cfg_kwargs["use_cpu"] = True
    if "no_cuda" in cfg_fields and dev.type == "cpu":
        cfg_kwargs["no_cuda"] = True
    if "dataloader_pin_memory" in cfg_fields:
        cfg_kwargs["dataloader_pin_memory"] = dev.type == "cuda"
    if "warmup_steps" in cfg_fields:
        cfg_kwargs["warmup_steps"] = 0 if (max_steps > 0 and max_steps < 20) else 20

    # Keep only keys supported by installed TRL version
    filtered_kwargs = {k: v for k, v in cfg_kwargs.items() if k in cfg_fields}
    dpo_args = DPOConfig(**filtered_kwargs)

    trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs: dict = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": train_dataset,
    }
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if peft_config is not None and "peft_config" in trainer_sig:
        trainer_kwargs["peft_config"] = peft_config

    trainer = DPOTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trained_model = trainer.model
    trained_model.eval()

    saved_path: str | None = None
    if save_adapter:
        trained_model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        saved_path = str(output_dir)

    metrics = dict(train_result.metrics) if hasattr(train_result, "metrics") else {}
    train_stats = {
        "model_name": model_name,
        "dpo_beta": float(dpo_beta),
        "seed": int(seed),
        "num_train_pairs": int(len(train_dataset)),
        "use_lora": bool(use_lora),
        "lora_r": int(lora_r) if use_lora else 0,
        "lora_target_modules": target_modules,
        "train_loss": float(metrics.get("train_loss", getattr(train_result, "training_loss", float("nan")))),
        "global_step": int(getattr(trainer.state, "global_step", 0)),
        "saved_path": saved_path,
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
    }
    return trained_model, tokenizer, train_stats


def score_policy_on_eval_prompts(
    model,
    tokenizer,
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
    batch_size: int = 4,
    max_length: int = 1024,
    device: str | torch.device = "cpu",
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute raw completion log-probs and token lengths for every eval prompt's candidates."""
    raw_lps: dict[int, np.ndarray] = {}
    lengths: dict[int, np.ndarray] = {}
    t0 = time.perf_counter()
    for idx_i, p_idx in enumerate(eval_prompt_indices, start=1):
        lps, lens = compute_completion_logprobs(
            model=model,
            tokenizer=tokenizer,
            prompt=ds.prompts[p_idx],
            completions=ds.prompt_responses[p_idx],
            batch_size=batch_size,
            device=device,
            max_length=max_length,
            length_normalize=False,
        )
        raw_lps[p_idx] = lps
        lengths[p_idx] = lens
        if idx_i % 100 == 0 or idx_i == len(eval_prompt_indices):
            print(f"    Scored DPO log-probs on {idx_i}/{len(eval_prompt_indices)} prompts ({time.perf_counter() - t0:.1f}s)...", flush=True)
    return raw_lps, lengths


def evaluate_dpo_policy_distortion(
    ds: Turn1Dataset,
    eval_prompt_indices: list[int],
    avg_util: np.ndarray,
    ref_lps: dict[int, np.ndarray],
    ref_lens: dict[int, np.ndarray],
    dpo_lps: dict[int, np.ndarray],
    dpo_lens: dict[int, np.ndarray],
    ref_mode: str = "llm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exact per-query KL-constrained distortion of a trained DPO policy against pi_ref.

    Supports `ref_mode`:
      - `"llm"`: raw completion log-probs `sum_t log p(y_t | x, y_{<t})` for both pi_ref and pi_DPO.
      - `"llm_len_norm"`: length-normalized completion log-probs `(1/T) sum_t log p(...)` for both.
    """
    if ref_mode not in {"llm", "llm_len_norm"}:
        raise ValueError(f"Unsupported ref_mode={ref_mode!r}; expected 'llm' or 'llm_len_norm'.")

    dist_rows: list[dict] = []
    diag_rows: list[dict] = []

    for p_idx in eval_prompt_indices:
        g_idxs = ds.prompt_global_xy_indices[p_idx]
        au = avg_util[g_idxs]

        r_lp = np.asarray(ref_lps[p_idx], dtype=np.float64)
        d_lp = np.asarray(dpo_lps[p_idx], dtype=np.float64)
        if ref_mode == "llm_len_norm":
            r_len = np.maximum(np.asarray(ref_lens[p_idx], dtype=np.float64), 1.0)
            d_len = np.maximum(np.asarray(dpo_lens[p_idx], dtype=np.float64), 1.0)
            pi_ref = logprobs_to_probs(r_lp / r_len)
            pi_dpo = logprobs_to_probs(d_lp / d_len)
        else:
            pi_ref = logprobs_to_probs(r_lp)
            pi_dpo = logprobs_to_probs(d_lp)

        res = compute_distortion(avg_util=au, pi=pi_dpo, pi_ref=pi_ref)
        ref_util = float(np.dot(pi_ref, au))
        ref_top1 = int(np.argmax(pi_ref))
        dpo_top1 = int(np.argmax(pi_dpo))
        opt_top1 = int(np.argmax(au))

        dist_rows.append(
            {
                "prompt_idx": int(p_idx),
                "ref_mode": ref_mode,
                "num_candidates": len(g_idxs),
                "tau": res["tau"],
                "policy_util": res["policy_util"],
                "opt_util": res["opt_util"],
                "ref_util": ref_util,
                "distortion": res["distortion"],
                "top1_changed_vs_ref": bool(dpo_top1 != ref_top1),
                "dpo_matches_util_opt": bool(dpo_top1 == opt_top1),
            }
        )
        diag = ref_policy_diagnostics(pi_dpo)
        diag["prompt_idx"] = int(p_idx)
        diag["ref_mode"] = ref_mode
        diag_rows.append(diag)

    return pd.DataFrame(dist_rows), pd.DataFrame(diag_rows)
