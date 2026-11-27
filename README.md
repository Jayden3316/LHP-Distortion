# LHP-Distortion

**Empirical Distortion of Preference Alignment in Practical Settings**

This repository investigates how much *utilitarian distortion* arises in practical preference-alignment pipelines (single-reward Bradley–Terry RLHF and DPO) relative to the KL-constrained bounds established in Gölz, Haghtalab, & Yang (2025), *"Distortion of AI Alignment: Does Preference Optimization Optimize for Preferences?"* ([arXiv:2505.23749](https://arxiv.org/abs/2505.23749), Theorem 6 & Proposition 14).

---

## 1. Problem Formulation & Notation

To avoid overloading $\beta$ (which denotes the Bradley–Terry inverse temperature in Gölz et al. and the KL penalty in DPO), we use the following notation throughout the repository:

| Symbol | Meaning |
| :--- | :--- |
| $x$ | Input prompt / query |
| $\mathcal{A}(x)$ | Finite set of candidate responses $y$ for query $x$ |
| $\psi(x, y) \in \mathbb{R}^{768}$ | Frozen `answerdotai/ModernBERT-base` CLS embedding of `(x, y)` |
| $\phi(x, y) = W\psi(x, y) \in \mathbb{R}^d$ | Prompt–response embedding in learned user-preference space ($W \in \mathbb{R}^{d \times 768}$) |
| $w_i \in \mathbb{R}^d$ | Low-rank preference embedding for user $i \in \{1, \dots, N\}$ ($N = 839$) |
| $\tilde{u}_i(y \mid x) \in [0, 1]$ | Per-query BT-preserving normalized cardinal utility of user $i$ for response $y$ |
| $\mathrm{AvgUtil}(y \mid x)$ | Utilitarian social welfare: $\frac{1}{N} \sum_{i=1}^N \tilde{u}_i(y \mid x)$, extended to policies as $\mathbb{E}_{y \sim \pi(\cdot \mid x)}[\mathrm{AvgUtil}(y \mid x)]$ |
| $\beta_{\mathrm{BT}} > 0$ | User-level Bradley–Terry inverse temperature: $P(y_1 \succ_i y_2 \mid x) = \sigma\!\big(\beta_{\mathrm{BT}}(\tilde{u}_i(y_1 \mid x) - \tilde{u}_i(y_2 \mid x))\big)$ |
| $\lambda > 0$ (or $\beta_{\mathrm{DPO}}$) | KL-regularization strength in RLHF (Phase 4A) / DPO (`trl.DPOConfig.beta`, Phase 4B) |
| $\pi_{\mathrm{ref}}(\cdot \mid x)$ | Reference policy distribution over $\mathcal{A}(x)$ from base LLM completion log-probs |
| $\tau(x)$ | Realized KL radius for policy $\pi$ on query $x$: $\tau(x) := D_{\mathrm{KL}}\!\big(\pi(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)$ |
| $B_{\tau}(\pi_{\mathrm{ref}})$ | KL ball around reference policy: $\{\pi' \in \Delta(\mathcal{A}(x)) : D_{\mathrm{KL}}(\pi' \| \pi_{\mathrm{ref}}) \le \tau\}$ |

### Distortion Metric & Theorem 6 Benchmark

By Proposition 14 of Gölz et al. (2025), any $\lambda$-regularized RLHF/DPO policy $\pi$ is optimal for the $\tau$-constrained problem with $\tau(x) = D_{\mathrm{KL}}(\pi(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x))$. For each query $x$, the **empirical KL-constrained distortion** is:

$$\mathrm{dist}(x; \pi, \pi_{\mathrm{ref}}) = \frac{\max_{\pi^\star \in B_{\tau(x)}(\pi_{\mathrm{ref}}(\cdot \mid x))} \mathrm{AvgUtil}(\pi^\star \mid x)}{\mathrm{AvgUtil}(\pi(\cdot \mid x) \mid x)}$$

where the numerator's KL-constrained social-welfare maximizer $\pi^\star(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x)\exp(\mathrm{AvgUtil}(y \mid x)/\lambda'(x))$ is solved exactly via 1D root-finding (`scipy.optimize.brentq`). Under Theorem 6 (Appendix E.2), worst-case KL-constrained distortion rises from $1$ at $\tau = 0$ to its tight constant-free peak $R(\beta_{\mathrm{BT}}) = \frac{\sinh(\beta_{\mathrm{BT}})}{\beta_{\mathrm{BT}}} = \Theta(e^{\beta_{\mathrm{BT}}}/\beta_{\mathrm{BT}})$ at $\tau = \ln 2$.

---

## 2. Dataset & Setup

We use the **English Turn-1 split** of [`facebook/community-alignment-dataset`](https://huggingface.co/datasets/facebook/community-alignment-dataset), filtered to active users with $K > 20$ Turn-1 comparisons (**839 users**, **26,467 Turn-1 rows**, **79,323 deduplicated pairwise pairs**, **45,797 unique `(x, y)` pairs**), and evaluate distortion on the **520 prompts** rated by $M \ge 20$ retained users (**10,851 unique `(x, y)` pairs**, median $|\mathcal{A}(x)| = 18$).

### Environment Installation

```bash
pip install -r requirements.txt
```

### Phase 0–1: Download & Audit Dataset

```bash
# Stream and save the English split (~518 MB) to data/community-alignment/community_alignment_en.csv
python download_community_alignment_en.py

# Inspect user query counts & prompt overlap
python analyze_community_alignment_en.py

# Audit Turn-1 user/prompt filtering thresholds (outputs results/turn1_selection_audit.json)
python audit_turn1_dataset.py
```

---

## 3. Running Each Phase

Every pipeline stage supports a fast **local CPU dry-run** (`--mode local-test`, using `HuggingFaceTB/SmolLM2-135M` and a tiny fixture) before running **full-scale GPU** jobs (`--mode full --device cuda`, using `Qwen/Qwen3.5-0.8B`).

### Unit & Smoke Tests (Local CPU)

```bash
pytest -v
```

### Phase 3: Individual User-Wise Low-Rank BT Models & ModernBERT Features

Encodes all unique `(x, y)` pairs with frozen `answerdotai/ModernBERT-base` (cached to `modernbert_cls_features.pt`), sweeps rank $d$ and L2 weight decay, fits the shared mythical-user baseline, and generates 4-panel fit/calibration/disagreement diagnostics.

```bash
# Local CPU dry-run (ranks 2, 8, 32 on small fixture)
python run_phase3_user_bt.py --mode local-test

# Full cloud GPU run (839 users, 79,323 pairs, ranks 2, 8, 32, 64, 128)
python run_phase3_user_bt.py --mode full --device cuda
```

### Phase 4A: Mythical-User Closed-Form Gibbs Sanity Check

Computes cached base LLM reference log-probs $\pi_{\mathrm{ref}}$, fits shared mythical-user BT rewards on observed and $\beta_{\mathrm{BT}}$-resampled datasets, evaluates exact closed-form KL-regularized policies $\pi_\lambda$, plots Distortion vs. $\beta_{\mathrm{BT}}$ and faceted Distortion vs. $\tau$, and prints qualitative top-distortion query reports.

```bash
# Local CPU dry-run (SmolLM2-135M + per-query normalization)
python run_phase4a_distortion.py --mode local-test --also-len-norm-ref --also-uniform-ref

# Full cloud GPU run (520 eval prompts, Qwen/Qwen3.5-0.8B)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_phase4a_distortion.py \
  --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
  --also-len-norm-ref --also-uniform-ref
```

### Phase 4B: Real DPO Finetuning (`trl.DPOTrainer` + PEFT LoRA)

Trains actual language models with TRL `DPOTrainer` across `--dpo-betas` (KL penalty $\beta_{\mathrm{DPO}}$), `--seeds`, and observed or $\beta_{\mathrm{BT}}$-resampled preferences, then evaluates exact empirical DPO distortion against $\pi_{\mathrm{ref}}$, Phase 4A Gibbs baselines, and Theorem 6.

```bash
# Local CPU smoke run (SmolLM2-135M, LoRA r=8, 3 steps)
python run_phase4b_dpo.py --mode local-test --max-steps 3 \
  --dpo-betas 0.1 0.5 --betas 2.0 --save-adapters

# Cloud GPU: real observed-preferences DPO sweep (3 seeds, saves LoRA adapters)
python run_phase4b_dpo.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
  --only-observed --dpo-betas 0.05 0.1 0.25 0.5 --seeds 42 123 456 --save-adapters

# Cloud GPU: full Theorem 6 resampled beta_BT DPO sweep
python run_phase4b_dpo.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B \
  --betas 1.0 2.0 4.0 8.0 --dpo-betas 0.05 0.1 0.25 0.5 --seeds 42 123
```

### Qualitative User-Space Geometry, Contours & DPO Shift (`run_qualitative_pca.py`)

Visualizes trained user-space embeddings $\phi(x, y) = W\psi(x, y)$ across top-$N$ highest vs. lowest distortion prompts: (A) 2D PCA + KDE contours, (B) within-prompt centered spread $\Delta\phi(x, y)$, (C) 1D consensus-axis histogram + KDE, and (D) prompt-level user polarization vs. distortion across all 520 eval prompts. When `--dpo-model` is passed, generates $k$ completions per prompt and projects them onto the exact same axes.

```bash
# Dataset-only contour & polarization analysis from Phase 3 checkpoint (CPU, ~35s)
python run_qualitative_pca.py --mode full --top-n 100 --skip-gen --device cpu

# Full DPO vs. Base/Ref LM generation shift projection on GPU
python run_qualitative_pca.py --mode full --device cuda --top-n 10 --k 8 \
  --dpo-model results/phase4b/checkpoints/obs_dpob0p1_s42 --ref-model Qwen/Qwen3.5-0.8B
```

### Sanity Check: $\beta_{\mathrm{BT}}$ Recovery on Resampled Datasets

Refits `LowRankUserBT` and `MythicalUserBT` on ModernBERT features from $\beta_{\mathrm{BT}}$-resampled datasets across weight decays to verify recovered inverse temperature vs. regularization shrinkage / separability.

```bash
python check_beta_recovery.py --mode full --refit-rank 32 --epochs 40
```

---

## 4. Repository Structure

```text
LHP-Distortion/
├── README.md                            # Project overview & run commands
├── LOG.md                               # Design decisions, results & roadmap log
├── requirements.txt                     # Python dependencies (Phases 1–4B)
├── download_community_alignment_en.py   # Stream & filter EN split of Community Alignment
├── analyze_community_alignment_en.py    # EN split user query counts & prompt overlap
├── audit_turn1_dataset.py               # Turn-1 K / M threshold audit
├── run_phase3_user_bt.py                # Phase 3: ModernBERT caching + UserBT rank/WD sweep
├── run_phase4a_distortion.py            # Phase 4A: closed-form Gibbs KL-distortion & plots
├── run_phase4b_dpo.py                   # Phase 4B: TRL DPOTrainer + LoRA & empirical distortion
├── run_qualitative_pca.py               # Qualitative 2D PCA, 1D consensus KDE & DPO shift
├── check_beta_recovery.py               # Sanity check: recovering beta_BT from resampled pairs
├── src/
│   ├── data.py                          # Turn-1 loader, K>20 filter, 4-way -> 3-pair expansion
│   ├── encoder.py                       # Frozen ModernBERT-base encoder + linear head W
│   ├── ref_policy.py                    # Chunked completion log-probs & PEFT-aware loader
│   ├── user_bt.py                       # LowRankUserBT & per-query/unit-range/global normalization
│   ├── mythical_bt.py                   # MythicalUserBT, beta_BT resampling & Gibbs policy
│   ├── distortion.py                    # Exact KL-ball welfare solver & Theorem 6 curves
│   └── dpo.py                           # TRL DPOTrainer + LoRA training & distortion evaluation
├── tests/                               # Local CPU unit & smoke tests (pytest)
├── data/                                # Local datasets (gitignored)
└── results/                             # Checkpoints, CSVs, summaries & figures
```

See [LOG.md](LOG.md) for detailed experimental logs, locked-in design decisions, and results.
