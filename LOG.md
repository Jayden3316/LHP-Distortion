# Project Log — LHP-Distortion

Running log of completed work, locked-in design decisions (mapped to sections of `../LHP_Report/main.tex`), empirical sanity checks, and the current experimental roadmap (Phases 0–4B + Qualitative Analysis).

---

## Completed Milestones

### 2026-11-26 — Dataset Exploration & Turn-1 Selection Audit
- [x] Downloaded `HannahRoseKirk/prism-alignment` (`survey`, `conversations`, `utterances`, `metadata`) to `data/prism-alignment/`.
- [x] Analyzed PRISM per-user counts (`analyze_prism_users.py`, `prism_dataset_exploration.ipynb`) and audited prompt overlap → no shared prompt set across users (max 13/1,396 users on a substantive opening prompt). Switched datasets.
- [x] Streamed `facebook/community-alignment-dataset` (90,256 conversations) and filtered on-the-fly to English (`download_community_alignment_en.py`) → `data/community-alignment/community_alignment_en.csv` (30,856 rows, 517.7 MB).
- [x] Analyzed English split (`analyze_community_alignment_en.py`): 1,331 users, 30,856 conversations, 79,514 preference queries; 1,033 pregenerated Turn-1 prompts (560 with $\ge 10$ users; top prompts have ~70–136 unique responses).
- [x] Audited Turn-1-only user filtering thresholds (`audit_turn1_dataset.py` → `results/turn1_selection_audit.json`):

| Filter | Users | Turn-1 Rows | Pairwise Pairs | Unique `(x, y)` | Prompts ($\ge 10$u) | Prompts ($\ge 20$u) | Prompts ($\ge 30$u) | Prompts ($\ge 50$u) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $K > 10$ | 1,023 | 29,373 | 88,119 | 49,206 | 560 | 559 | 481 | 20 |
| $K > 15$ | 934 | 28,178 | 84,534 | 47,745 | 559 | 551 | 415 | 20 |
| **$K > 20$** | **839** | **26,467** | **79,323** (dedup) | **45,797** | **559** | **520** | **329** | **19** |
| $K > 25$ | 669 | 22,518 | 67,554 | 41,644 | 552 | 431 | 141 | 18 |
| $K > 30$ | 261 | 11,093 | 33,279 | 28,690 | 336 | 23 | 12 | 1 |

- [x] Selected **$K > 20$** active users (**839 users**, per-user pairs: min `63`, Q1 `78`, median `87`, mean `94.54`, Q3 `93`, p95 `146.4`, max `567`) and **$M \ge 20$** eval prompts (**520 prompts**, median $|\mathcal{A}(x)| = 18$, **10,851 unique `(x, y)`**).

### 2026-11-26 — Phases 0–4A Core Library, Normalization, Report & T4 Fixes
- [x] Upgraded local `.da5453` environment to `torch 2.15.0+cpu`, `transformers 5.24.1`, `numpy 2.5.5`, `trl 1.19.0`, `peft 0.22.0`, `accelerate 1.16.0`, `datasets 5.2.0`, `pytest 9.2.0`.
- [x] Implemented Phases 1–4A modules under `src/`:
  - `src/data.py`, `src/encoder.py`, `src/ref_policy.py`, `src/user_bt.py`, `src/mythical_bt.py`, `src/distortion.py`.
- [x] Fixed T4 GPU VRAM creep in `src/encoder.py` (`.contiguous().clone()` on CLS slice so `[B, L, H]` storage isn't pinned, CPU output accumulation, auto `bf16`/`fp16` detection, dynamic `reference_compile` guard for `transformers 5.x`).
- [x] Fixed T4 GPU OOM in `src/ref_policy.py` for 248K-vocab `Qwen/Qwen3.5-0.8B`:
  - Runs backbone `model.model(..., use_cache=False)` to get `[B, L, H]` hidden states (~8 MB) and projects **only completion tokens** through `model.lm_head` in 64-token chunks (~64 MB peak in fp32) with fused `-F.cross_entropy(..., reduction="sum")` (eliminating multi-GB `[B, L, 248320]` logits & `F.log_softmax` allocations).
  - Length-sorts completions before batching, auto-selects `float16` on T4 (`sm_75`), explicitly frees batch tensors, and transparently loads PEFT/LoRA adapter directories (`adapter_config.json`).
- [x] Updated `../LHP_Report/main.tex` (Section 3 Methodology: 3.1 Dataset & Subsampling + Table 1, 3.2 Latent Utilities & BT-preserving Per-Query Normalization, 3.3 Reference Policy & KL-Constrained Distortion) and verified clean `pdflatex` compilation (`main.pdf`, 5 pages, 0 warnings/errors).

### 2026-11-26 — Full Cloud Phase 3 Completed (`results/phase3/`)
- [x] Ran `run_phase3_user_bt.py --mode full --device cuda` on T4 GPU (46.7 min total; cached `modernbert_cls_features.pt` for all 45,797 `(x, y)` pairs):

| Model Family | Best WD | Val Loss | Val Acc | Val Brier | Full Acc | $\beta_{\mathrm{BT}}^{\mathrm{nat}}$ | Median Top-1 Majority |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared (Mythical) | `0.1` | `0.6153` | `65.02%` | `0.2137` | `66.60%` | — | `1.000` |
| UserBT (`d=2`) | `0.1` | `0.5773` | `67.64%` | `0.1983` | `70.66%` | `69.08` | `0.628` |
| UserBT (`d=8`) | `1.0` | `0.5783` | `68.38%` | `0.1985` | `73.85%` | `26.79` | `0.396` |
| UserBT (`d=32`) | `1.0` | `0.5178` | `73.90%` | `0.1729` | `86.27%` | `40.27` | `0.284` |
| UserBT (`d=64`) | `1.0` | `0.4830` | `76.81%` | `0.1583` | `90.29%` | `48.08` | `0.265` |
| **UserBT (`d=128`)** | **`1.0`** | **`0.4735`** | **`78.03%`** | **`0.1531`** | **`91.66%`** | **`58.90`** | **`0.242`** |

- **Key Phase 3 findings**:
  - Modeling heterogeneous users yields **+13.0 percentage points held-out accuracy** over the mythical single-user BT baseline (`65.0% → 78.0%`, `val_loss 0.615 → 0.473`), with near-perfect held-out probability calibration (`results/phase3/phase3_user_bt_diagnostics.png`).
  - Across the 520 eval prompts, the median top-1 majority share is only **24.2%** (`0.242`), and per-query shared spans $M(x)$ range `5.30–23.61` (median `11.83`).

### 2026-11-27 — Qualitative User-Space Geometry & Polarization Analysis (`results/qualitative_pca/`)
- [x] Implemented and upgraded `run_qualitative_pca.py` + `tests/test_qualitative_pca.py` to generate a **4-panel user-space analysis** (`qualitative_pca_user_space.png`):
  - **(A) Raw 2D PCA of $\phi(x, y) = W\psi(x, y)$** colored by query distortion with 2D Gaussian KDE contours for top-$N$ high- vs. low-distortion queries (plus DPO & Base/Ref LM generation overlays when `--dpo-model` is provided).
  - **(B) Within-Prompt Centered 2D PCA of $\Delta\phi(x, y) = \phi(x, y) - \bar{\phi}(x)$** (removes prompt-topic offset; captures `39.5%` variance in 2D on `rank=128`).
  - **(C) 1D Consensus Score Histogram + KDE** along $\langle \hat{\bar{w}}, \Delta\phi(x, y) \rangle$ (where $\bar{w} = \frac{1}{N}\sum_i w_i$ is the mean utilitarian user direction) comparing low- vs. high-distortion queries.
  - **(D) Prompt-Level User Polarization vs. Distortion** across all 520 eval prompts (User Fragmentation $1 - \max_y \mathrm{Top1Share}(y|x)$ and Cross-User Utility Std).
- [x] Ran on the full `rank=128` checkpoint (`user_bt_best.pt`, `top-n=100`, `N = 3,540` plotted responses across 200 extreme queries + all 520 prompts in Panel D):
  - **Consensus Bimodality Discovery (Panel C)**: Low-distortion prompts exhibit a **clearly bimodal** consensus-score distribution (mediocre responses around `-2.5`, trough around `+2.0`, and a distinct standout consensus-winner peak at `+6.0` to `+7.5`), whereas high-distortion prompts collapse into a **unimodal** cluster centered near `0.0` (no standout utilitarian winner).
  - **High-Dimensional Disagreement**: Top 5 PCs of centered user embeddings $W_{\mathrm{users}} - \bar{w}$ explain only `[5.0%, 3.7%, 3.3%, 3.0%, 2.9%]` of variance — disagreement is genuinely multi-dimensional rather than 1D, motivating the 1D consensus-direction histogram in Panel C over a single polarization axis.
  - **Polarization Envelope (Panel D)**: Both user fragmentation ($\rho = +0.244$, $p = 1.7 \times 10^{-8}$) and cross-user utility std ($\rho = +0.156$, $p = 3.5 \times 10^{-4}$) significantly predict distortion via a triangular "necessary-but-not-sufficient" envelope (consensus prompts with fragmentation $< 0.45$ almost never exceed distortion `1.04`).

### 2026-11-27 — Phase 4B DPO Implementation (`trl.DPOTrainer` + PEFT LoRA)
- [x] Implemented `src/dpo.py`, `run_phase4b_dpo.py`, and `tests/test_dpo.py`.
- [x] Verified `pytest -v tests/test_dpo.py` (**2/2 passed**) including a real 2-step `trl.DPOTrainer` + LoRA training, adapter save/reload via `load_causal_lm`, and exact numerical equivalence between chunked PEFT `compute_completion_logprobs` and brute-force autoregressive logits.
- [x] Verified end-to-end local CPU smoke run (`python run_phase4b_dpo.py --mode local-test --max-steps 3 --dpo-betas 0.1 0.5 --betas 2.0 --save-adapters`) → `results/phase4b_local_test/` (reuses cached `pi_ref` log-probs, evaluates both raw `llm` and `llm_len_norm`, overlays Phase 4A Gibbs medians & Theorem 6 curves).

### 2026-11-27 — $\beta_{\mathrm{BT}}$ Resampling & Recovery Sanity Check (`check_beta_recovery.py`)
- [x] Refit `LowRankUserBT` (`rank=32`, 80/20 split, `wd ∈ {0.001, 0.1, 1.0}`) and `MythicalUserBT` (`wd=0.1`) on ModernBERT features from resampled datasets at $\beta_{\mathrm{target}} \in \{1, 2, 4, 8, 16\}$ (`results/beta_recovery_check/beta_recovery_full_rank32.csv`):

| $\beta_{\mathrm{target}}$ | Bayes Acc | Bayes NLL | Val Acc (`wd=1.0`) | Val NLL (`wd=1.0`) | $\hat{\beta}_{\mathrm{slope}}$ (`wd=1.0`) | $\hat{\beta}_{\mathrm{slope}}$ (`wd=0.001`) | Med $M(x)$ (`wd=1.0`) | Med $M(x)$ (`wd=0.001`) | Mythical Span |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0` | `0.552` | `0.685` | `0.508` | `0.868` | `0.71` | `2.18` | `8.68` | `35.10` | `0.60` |
| `2.0` | `0.601` | `0.662` | `0.528` | `0.848` | `1.55` | `4.54` | `9.37` | `37.49` | `0.89` |
| `4.0` | `0.684` | `0.592` | `0.590` | `0.780` | `3.22` | `9.88` | `11.44` | `43.47` | `1.43` |
| `8.0` | `0.789` | `0.448` | `0.681` | `0.659` | `5.89` | `19.36` | `14.84` | `57.14` | `2.18` |
| `16.0` | `0.883` | `0.272` | `0.763` | `0.516` | `8.58` | `29.30` | `18.49` | `69.68` | `2.76` |

- **Key conclusions**:
  1. Single-shot pairs *do* identify finite $\beta_{\mathrm{BT}}$ in population because `79,323` feature-linked comparisons control label SNR (Bayes acc `55.2% → 88.3%`).
  2. However, for overparameterized per-user MLE (`rank=32`, `~51K` params on `63K` train pairs), **weak regularization (`wd=0.001`) causes severe finite-sample separation / logit blow-up** (`val_nll = 2.09` even at $\beta=1$, `med_M(x) = 35–70`). With the validation-selected Gaussian prior (`wd=1.0`), $\hat{\beta}_{\mathrm{slope}}$ tracks $\beta_{\mathrm{target}}$ closely (`0.71, 1.55, 3.22, 5.89` for `1, 2, 4, 8`) before bending downward from L2 shrinkage at $\beta=16$ (`8.58`).
  3. Taking the max span over 839 users ($M(x) = \max_j \mathrm{span}_j(x)$) adds an extreme-value noise floor (~`8.5`), whereas the non-separable 768-dim `MythicalUserBT` reward span scales cleanly (`0.60 → 2.76`). Crucially, Phase 4A/4B never refit `LowRankUserBT` on resampled pairs—$\tilde{u}_i$ stays fixed while only the alignment mechanism (`MythicalUserBT` / `DPOTrainer`) sees the resampled pairs.

---

## Locked-In Design Decisions (Mapped to Report)

| ID | Decision | Specification & Rationale | Report Section |
| :--- | :--- | :--- | :--- |
| **D0** | **Theoretical Focus** | Focus exclusively on **KL-constrained RLHF / DPO (Theorem 6 & Prop. 14)** of Gölz et al. (2025). Measure empirical distortion vs. realized per-query KL radius $\tau(x) = D_{\mathrm{KL}}(\pi(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x))$ across $\beta_{\mathrm{BT}}$ settings. | Sec. 2 (Background) |
| **D1a** | **Compute & Testing Protocol** | Every pipeline module has fast **local CPU unit & smoke tests** (`SmolLM2-135M`, `ModernBERT-base`) verified before cloud GPU runs. | Sec. 3 (Methodology) |
| **D1b** | **Cloud LLMs & Multi-Seed Plan** | Primary cloud LLM: `Qwen/Qwen3.5-0.8B` (for $\pi_{\mathrm{ref}}$ and Phase 4B DPO via `trl.DPOTrainer`, multi-seed to capture data-order sensitivity of $\tau(x)$). Scale up to 4B once 0.8B trends are confirmed. | Sec. 3 & 4 |
| **D1c** | **Encoder & User Embeddings** | Prompt–response features $\phi(x, y) = W \psi(x, y)$ using frozen `answerdotai/ModernBERT-base` + trainable linear head $W \in \mathbb{R}^{d \times 768}$. Pure user-ID embeddings $w_i \in \mathbb{R}^d$ (no demographics), sweeping $d \in \{2, 8, 32, 64, 128\}$ and $\mathrm{wd} \in \{10^{-3}, 10^{-2}, 10^{-1}, 1, 5\}$ on full data (best: `d=128, wd=1.0`). | Sec. 3 (Methodology) |
| **D2** | **BT-Preserving Per-Query Normalization** | Fit unconstrained bilinear logits $s_i(x, y) = \langle w_i, W\psi(x, y) \rangle$. Normalize per query by removing each user's unidentifiable offset $s_i(x, y) - \min_{y'} s_i(x, y')$ and dividing **all users on query $x$** by the single shared max-span $M(x) = \max_j(\max_{y'} s_j(x, y') - \min_{y'} s_j(x, y'))$. Preserves exact within-query BT probabilities (under shared $\beta_{\mathrm{BT}}(x) = M(x)$), relative cross-user intensity ratios, and isolation from cross-query outliers. Keep `unit-range` and `global` as ablations. | Sec. 3.2 |
| **D3** | **Phase 4A Sanity Check & $\pi_{\mathrm{ref}}$ Ablations** | Run Phase 4A closed-form Gibbs policy $\pi_\lambda(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x)\exp(r_{\mathrm{MLE}}(x, y)/\lambda)$ as an optimization-free sanity check before Phase 4B DPO. Track both raw log-probs (`llm`), length-normalized log-probs (`llm_len_norm`, prevents short-completion collapse), and `uniform` $\pi_{\mathrm{ref}}$. | Sec. 3.3 |
| **D4** | **Data Scope & Pairwise Loss** | Restrict strictly to **Turn-1 prompts & responses** from `community_alignment_en.csv` ($K > 20 \Rightarrow 839$ users, $79,323$ deduplicated pairs; $M \ge 20 \Rightarrow 520$ eval prompts). Each 4-way choice yields 3 ordered pairs `(chosen, rejected)`. | Sec. 3.1 |
| **D5** | **Theorem 6 $\tau$-Dependent Curves** | Plot both Distortion vs. $\beta_{\mathrm{BT}}$ (with $\tau$-dependent Theorem 6 curves) and faceted Distortion vs. $\tau$. Exact Theorem 6 curve rises from $1$ at $\tau = 0$ via $\frac{q(\tau)R(\beta) + (1-q(\tau))}{(1-q(\tau))R(\beta) + q(\tau)}$ ($D_{\mathrm{KL}}((q,1-q)\|(\frac{1}{2},\frac{1}{2})) = \tau$) to peak $R(\beta) = \sinh(\beta)/\beta$ at $\tau = \ln 2$. | Sec. 2 & 4 |
| **D6** | **Population Scope for $\mathrm{AvgUtil}$** | Compute $\mathrm{AvgUtil}(y \mid x)$ by averaging $\tilde{u}_i(y \mid x)$ over **all 839 retained users** on each of the 520 eval prompts. | Sec. 3.1–3.2 |
| **D7** | **Qualitative User-Space Evaluation** | Project extreme high- vs. low-distortion prompts in trained user space $\phi(x, y)$: (A) raw 2D PCA + KDE contours, (B) within-prompt centered $\Delta\phi(x, y)$ PCA, (C) 1D consensus-direction histogram + KDE (reveals bimodal standout-winner structure in low-distortion prompts vs. unimodal ambiguity in high-distortion prompts), and (D) prompt-level fragmentation vs. distortion across all 520 prompts. Overlay DPO vs. Base LM generations once Phase 4B checkpoints are trained. | Sec. 4 (Qualitative) |
| **D8** | **Phase 4B DPO Training (`trl` + LoRA)** | Train real LLMs with `trl.DPOTrainer` + PEFT LoRA (`ref_model=None` so adapters disable in-place for $\pi_{\mathrm{ref}}$ without duplicating model weights in VRAM), dynamically targeting attention/MLP/DeltaNet linear layers (`q/k/v/o_proj`, `in_proj_qkv`, `out_proj`, `gate/up/down_proj`), sweeping `--dpo-betas` ($\beta_{\mathrm{DPO}}$) and `--seeds`, and evaluating exact KL-constrained distortion under both `llm` and `llm_len_norm`. | Sec. 3.3 & 4 |
| **D9** | **$\beta_{\mathrm{BT}}$ Resampling & Prior Trade-off** | Resample pair orientations from fixed $\tilde{u}_i$ via $\sigma(\beta_{\mathrm{BT}}\Delta\tilde{u}_i)$ to probe Theorem 6's $\beta_{\mathrm{BT}}$ scaling. Verified via `check_beta_recovery.py` that finite-sample per-user MLE requires L2 regularization (`wd=1.0`) to prevent logit blow-up (at the cost of mild shrinkage at high $\beta_{\mathrm{BT}}$), while the 768-dim `MythicalUserBT` stays non-separable and scales monotonically. | Sec. 3.2 & Appendix |

---

## Current Status & Remaining Roadmap

- [x] **Phase 0–2**: Environment, Turn-1 dataset construction, frozen ModernBERT encoder, chunked PEFT-aware `ref_policy.py` (all unit tests passing: **29 total tests** across 9 test files).
- [x] **Phase 3 (Full Cloud Run Complete)**: `results/phase3/` (`rank=128, wd=1.0, val_acc=78.03%`).
- [x] **Qualitative Analysis (Dataset Contours Complete)**: `results/qualitative_pca/` (`rank=128`, 520 prompts, 3,540 plotted responses).
- [x] **$\beta_{\mathrm{BT}}$ Recovery Sanity Check Complete**: `results/beta_recovery_check/beta_recovery_full_rank32.csv`.
- [x] **Phase 4A & 4B Code + Local CPU Verification Complete**: `run_phase4a_distortion.py`, `run_phase4b_dpo.py`, `src/dpo.py`, `tests/test_dpo.py`.
- [ ] **Cloud Run — Phase 4A (Full)**: Run `run_phase4a_distortion.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B --also-len-norm-ref --also-uniform-ref` (now OOM-safe on T4).
- [ ] **Cloud Run — Phase 4B (Full DPO)**: Run `run_phase4b_dpo.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B` (observed multi-seed sweep + resampled $\beta_{\mathrm{BT}}$ sweep, saving LoRA adapters).
- [ ] **Qualitative DPO Shift**: Run `run_qualitative_pca.py --mode full --device cuda --dpo-model <adapter_dir> --ref-model Qwen/Qwen3.5-0.8B` to project before-vs-after DPO generations onto the PCA & 1D consensus distributions.
- [ ] **Scale-up & Report**: Optional 4B DPO scale-up check, then populate Section 4 (Results) & Section 5 (Discussion) of `../LHP_Report/main.tex`.
