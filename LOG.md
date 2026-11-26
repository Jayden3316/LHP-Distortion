# Project Log — LHP-Distortion

Running log of completed work, locked-in design decisions (mapped to sections of `../LHP_Report/main.tex`), and the step-by-step experimental roadmap (currently scoped through **Phase 4A**, before moving to DPO in Phase 4B).

---

## Completed Milestones

### 2026-11-26 — Dataset Exploration & Selection
- [x] Downloaded `HannahRoseKirk/prism-alignment` (`survey`, `conversations`, `utterances`, `metadata`) to `data/prism-alignment/`.
- [x] Analyzed PRISM per-user counts (`analyze_prism_users.py`, `prism_dataset_exploration.ipynb`) and audited prompt overlap → no shared prompt set across users (max 13/1,396 users on a substantive opening prompt). Switched datasets.
- [x] Streamed `facebook/community-alignment-dataset` (90,256 conversations) and filtered on-the-fly to English (`download_community_alignment_en.py`) → `data/community-alignment/community_alignment_en.csv` (30,856 rows, 517.7 MB).
- [x] Analyzed English split (`analyze_community_alignment_en.py`): 1,331 users, 30,856 conversations, 79,514 preference queries; 1,033 pregenerated Turn-1 prompts (560 with $\ge 10$ users; top prompts have ~70–136 unique responses).
- [x] Audited Turn-1-only user filtering thresholds ($K$ = min Turn-1 responses/user, each yielding 3 pairwise comparisons):

| Filter | Users | Turn-1 Rows | Pairwise Pairs | Unique `(x, y)` | Prompts ($\ge 10$u) | Prompts ($\ge 20$u) | Prompts ($\ge 30$u) | Prompts ($\ge 50$u) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| $K > 10$ | 1,023 | 29,373 | 88,119 | 49,206 | 560 | 559 | 481 | 20 |
| $K > 15$ | 934 | 28,178 | 84,534 | 47,745 | 559 | 551 | 415 | 20 |
| **$K > 20$** | **839** | **26,467** | **79,401** | **45,797** | **559** | **520** | **329** | **19** |
| $K > 25$ | 669 | 22,518 | 67,554 | 41,644 | 552 | 431 | 141 | 18 |
| $K > 30$ | 261 | 11,093 | 33,279 | 28,690 | 336 | 23 | 12 | 1 |

- [x] Audited theoretical framework in `../2505.23749v1.pdf` (Gölz, Haghtalab, & Yang, 2025).
- [x] Checked local environment: Python 3.12 (`.da5453`), `torch 2.3.1+cpu` (`cuda: False`), `scipy 1.17.1`, `scikit-learn 1.9.1`.

### 2026-11-26 — Phase 0–4A Core Implementation, Local Verification & Turn-1 Audit
- [x] Upgraded `transformers` to `4.49.0` (`tokenizers 0.21.4`), installed `pytest 9.2.0`, and cached `answerdotai/ModernBERT-base` + `HuggingFaceTB/SmolLM2-135M` locally.
- [x] Added compatibility guard in `src/encoder.py` for `torch 2.3.1` + Python 3.12 `@torch.compile` decorator inside `modeling_modernbert.py` (no-op fallback only when Dynamo raises Python 3.12 error; untouched on cloud `torch >= 2.4`).
- [x] Implemented Phases 1–4A modules under `src/`:
  - `src/data.py`, `src/encoder.py`, `src/ref_policy.py`, `src/user_bt.py`, `src/mythical_bt.py`, `src/distortion.py`.
- [x] Implemented and verified all 7 test modules (`tests/test_{data,encoder,ref_policy,user_bt,mythical_bt,distortion,e2e_smoke}.py`): **21/21 passed in 14.53s** on local CPU (including exact Theorem 6 synthetic reproduction: distortion `1.36 → 6.72 → 37.26` across $\beta_{\mathrm{BT}} \in \{4, 6, 8\}$ at $\tau = \log 2$, exceeding the $O(\beta^2)$ Borda upper bound of `16.0` at $\beta_{\mathrm{BT}} = 8$).
- [x] Ran Turn-1 dataset selection audit (`audit_turn1_dataset.py` → `results/turn1_selection_audit.json`):
  - $K > 20$ users: **839 users**, **8,719 unique prompts** (1,021 pregenerated), **45,797 unique `(x, y)` pairs**, **79,323 deduplicated pairwise comparisons** (78 raw duplicate/blank candidate pairs removed cleanly).
  - Prompt cutoff comparison:
    - $M \ge 20$ users/prompt: **520 prompts**, median $|\mathcal{A}(x)| = 18.0$ (min 4, max 124), **10,851 unique `(x, y)`**, **51,532 pairs**.
    - $M \ge 30$ users/prompt: **329 prompts**, median $|\mathcal{A}(x)| = 20.0$ (min 4, max 124), **7,564 unique `(x, y)`**, **36,828 pairs**.
    - $M \ge 40$ users/prompt: drops to **24 prompts** (median $|\mathcal{A}(x)| = 64.5$).
  - Word lengths on $M \ge 20$ prompts: prompt median **14** (p95 **35**, max **52**); response median **111** (p95 **203**, p99 **246**, max **931**) → `max_length = 512` covers >99% without truncation.

---

## Locked-In Design Decisions (Mapped to Report)

| ID | Decision | Specification & Rationale | Report Section |
| :--- | :--- | :--- | :--- |
| **D0** | **Theoretical Focus** | Focus exclusively on **KL-constrained RLHF / DPO (Theorem 6)** of Gölz et al. (2025). Measure empirical distortion as a function of realized per-query KL radius $\tau(x) = D_{\mathrm{KL}}(\pi(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x))$ across $\beta_{\mathrm{BT}}$ settings. | Sec. 2 (Background) |
| **D1a** | **Compute & Testing Protocol** | Every pipeline module has fast **local CPU unit & smoke tests** before cloud runs (A100-80GB up to 20h / H100-80GB up to 25h). Local test LLM: `HuggingFaceTB/SmolLM2-135M`. | Sec. 3 (Methodology) |
| **D1b** | **Cloud LLMs & Multi-Seed Plan** | Primary cloud LLM: `Qwen/Qwen3.5-0.8B` (for $\pi_{\mathrm{ref}}$ and Phase 4B DPO via `trl.DPOTrainer`, with multiple seeds to capture data-order sensitivity of $\tau(x)$). Once trends are clear, scale up to a 4B model. | Sec. 3 & 4 |
| **D1c** | **Encoder & User Embeddings** | Prompt–response features $\phi(x, y) = W \psi(x, y)$ using frozen `answerdotai/ModernBERT-base` (`transformers >= 4.48.0`) + trainable linear head $W \in \mathbb{R}^{d \times 768}$. User embeddings $w_i \in \mathbb{R}^d$ are purely user-ID embeddings (no demographics), sweeping rank $d \in \{2, 8, 32\}$ locally and $d \in \{2, 8, 32, 64, 128\}$ on full cloud runs. Online frozen forward pass for local runs; precomputed frozen CLS feature cache for full rank sweeps (mathematically identical). | Sec. 3 (Methodology) |
| **D2** | **Utility Fitting, Per-Query Normalization & $\beta_{\mathrm{BT}}$ Sweep** | Fit unconstrained linear logits $s_i(x, y) = \langle w_i, \phi(x, y) \rangle$ under pairwise BT loss. After fitting, apply **BT-preserving per-query normalization** into $[0, 1]$: remove each user's unidentifiable within-query offset $s_i(x, y) - \min_{y' \in \mathcal{A}(x)} s_i(x, y')$ and divide **all users on query $x$** by the single shared query max-span $M(x) = \max_j (\max_{y'} s_j(x, y') - \min_{y'} s_j(x, y'))$. This preserves (i) exact within-query BT comparison probabilities for every user under one shared temperature $\beta_{\mathrm{BT}}(x) = M(x)$, (ii) relative preference intensities across users (unlike per-user unit-range scaling, which inflates indifferent users to span 1), and (iii) isolation from cross-query outlier scales (unlike global affine scaling). Retain `unit-range` and `global` as ablations. For $\beta_{\mathrm{BT}}$ scaling sweeps, resample pairwise comparisons from $\tilde{u}_i$ under $P(y_1 \succ_i y_2 \mid x) = \sigma(\beta_{\mathrm{BT}}(\tilde{u}_i(y_1 \mid x) - \tilde{u}_i(y_2 \mid x)))$. | Sec. 3 (Methodology) |
| **D3** | **Denominator Policy & Reference Policy $\pi_{\mathrm{ref}}$** | Run **Option A (shared $\bar{w}$ BT-MLE + closed-form KL policy $\pi_\lambda(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x)\exp(r_{\mathrm{MLE}}(x, y)/\lambda)$)** first; inspect individual BT fit quality and Option A distortion before moving to **Option B (`trl.DPOTrainer`)**. Define $\pi_{\mathrm{ref}}(y \mid x) \propto \exp(\sum_{t=1}^T \log p_{\pi_0}(y_t \mid x, y_{<t}))$ normalized across $y \in \mathcal{A}(x)$, while monitoring entropy / length-bias collapse. | Sec. 3 (Methodology) |
| **D4** | **Data Scope & Loss** | Restrict strictly to **Turn-1 prompts & responses** from `community_alignment_en.csv`. No separate train/eval split required for distortion evaluation (theory operates on the training comparison distribution; held-out diagnostics can still be logged separately for user-BT sanity checks). Use **pairwise comparisons** only: each 4-way choice yields 3 ordered pairs `(chosen, rejected)`. | Sec. 3 (Methodology) |
| **D5** | **Target Plot & Theorem 6 Curve** | Multi-panel scatter plot with realized $\tau(x)$ on x-axis and empirical distortion on y-axis (one panel per $\beta_{\mathrm{BT}}$), overlaid with the exact closed-form Theorem 6 curve $\mathrm{dist}_{\mathrm{Thm6}}(\tau; \beta_{\mathrm{BT}})$ derived from Appendix E.2's 3-alternative instance: rises smoothly from $1$ at $\tau = 0$ via $\frac{q(\tau) R(\beta) + (1-q(\tau))}{(1-q(\tau)) R(\beta) + q(\tau)}$ (where $D_{\mathrm{KL}}((q, 1-q) \| (\frac{1}{2}, \frac{1}{2})) = \tau$) to its tight constant-free peak $R(\beta) = \frac{\sinh(\beta)}{\beta} = \Theta(e^\beta / \beta)$ at $\tau = \ln 2$ and plateaus at $R(\beta)$ for $\tau \ge \ln 2$. | Sec. 4 (Results) |
| **D6** | **User, Prompt & Population Scope** | Retain users with $K > 20$ Turn-1 preferences (**839 users**, >60 pairwise pairs/user, **26,467 Turn-1 rows**, **79,323 deduplicated pairs**, **45,797 unique `(x, y)`**). Evaluate distortion on Turn-1 prompts with **$M \ge 20$ retained annotators** (**520 prompts**, median $|\mathcal{A}(x)| = 18$, **10,851 unique `(x, y)`**). Compute $\mathrm{AvgUtil}(y \mid x)$ by averaging $\tilde{u}_i(y \mid x)$ over **all 839 retained users**. | Sec. 3 (Methodology) |

---

## Unit Test Plan (Phases 1–4A)

All test modules live under `tests/` and run locally on CPU:
1. `tests/test_data.py` — strict $K > 20$ boundary (`19/20/21`), 4-way → 3-pair expansion, blank/missing/duplicate response edge cases, Turn-2/3/4 isolation, deterministic ID round-trip.
2. `tests/test_encoder.py` — strict freezing of ModernBERT (`requires_grad == False`, unchanged weights, `grad is None`), exact equivalence between online frozen pass and cached CLS embeddings, empty/long-text truncation, rank shapes $d \in \{2, 8, 32\}$.
3. `tests/test_ref_policy.py` — prompt-token masking verified against brute-force token-by-token autoregressive log-probs on `SmolLM2-135M`, valid simplex & log-sum-exp numerical stability on extreme log-probs (`-5000`), single-candidate edge case, entropy/collapse diagnostics.
4. `tests/test_user_bt.py` — shift invariance & label-flip symmetry, synthetic 2-type heterogeneous recovery (rank-2 >98% vs. shared-$\bar{w}$ ~50%), global affine normalization invariants + constant-score `0/0` guard, L2 weight-decay shrinkage monotonicity.
5. `tests/test_mythical_bt.py` — Gibbs extremes ($\lambda \to \infty \Rightarrow \pi_{\mathrm{ref}}$, $\lambda \to 0^+ \Rightarrow \arg\max r_{\mathrm{MLE}}$ with proportional tie-splitting), KL monotonicity in $\lambda$, $\beta_{\mathrm{BT}}$ resampling extremes ($\beta_{\mathrm{BT}} = 0 \to 0.5$ win-rate, $\beta_{\mathrm{BT}} = 100 \to$ deterministic, seed reproducibility).
6. `tests/test_distortion.py` — 1D bisection vs. `scipy.optimize.minimize` (`< 1e-7`), $\tau = 0 \Rightarrow \mathrm{dist} = 1$, large $\tau \Rightarrow \max_y \mathrm{AvgUtil}(y)$, tied optimal responses, homogeneous population ($\mathrm{dist} = 1$ for all $\lambda$), universal $\mathrm{dist} \ge 1$ invariant, zero-denominator guard, and **exact reproduction of Theorem 6 (Appendix E.2)** showing exponential distortion in $\beta_{\mathrm{BT}}$.
7. `tests/test_e2e_smoke.py` — end-to-end local CPU smoke test (Phases 1 → 4A) on a tiny deterministic fixture with online frozen ModernBERT + `SmolLM2-135M`, verifying finite metrics and plot generation.

---

## Active Roadmap (Focused Through Phase 4A)

### Phase 0: Environment & Local Test Harness
- [x] Upgrade `transformers >= 4.48.0` (`4.49.0`) and install `pytest` (`9.2.0`) in `.da5453`.
- [x] Verify loading `answerdotai/ModernBERT-base` and `HuggingFaceTB/SmolLM2-135M` locally on CPU.

### Phase 1: Turn-1 Dataset Construction & Selection Audit
- [x] Implement `src/data.py` + `tests/test_data.py`.
- [x] Run Turn-1 dataset selection audit (`audit_turn1_dataset.py` → `results/turn1_selection_audit.json`).
- [x] Finalized prompt cutoff **$M \ge 20$** (520 prompts, 10,851 unique `(x, y)`) and $\mathrm{AvgUtil}$ population scope (**all 839 retained users**).
- [x] Pinned critical dependencies in `requirements.txt` (`transformers>=4.48.0,<5.0`, `tokenizers>=0.21.0,<0.22.0`, etc.).

### Phase 2: Frozen ModernBERT Feature Pipeline & Reference Policy $\pi_{\mathrm{ref}}$
- [x] Implement `src/encoder.py` + `tests/test_encoder.py` (frozen ModernBERT + linear head $W$, supporting online mode and frozen CLS cache mode).
- [x] Implement `src/ref_policy.py` + `tests/test_ref_policy.py` (completion log-probs $\sum_t \log p_{\pi_0}(y_t \mid x, y_{<t})$ normalized over $\mathcal{A}(x)$ + entropy/collapse diagnostics).

### Phase 3: Individual User-Wise Bradley–Terry Models & Quality Analysis
- [x] Implement `src/user_bt.py` + `tests/test_user_bt.py` (unconstrained bilinear logits $s_i(x, y) = \langle w_i, W\psi(x, y) \rangle$, global affine normalization to $[0, 1]$, implied $\beta_{\mathrm{BT}}^{\mathrm{nat}} = s_{\max} - s_{\min}$).
- [x] Implement Phase 3 training & diagnostic script `run_phase3_user_bt.py` with `--weight-decays` sweep (`[1e-3, 1e-2, 1e-1, 1.0]`), `--mode local-test` on CPU, and `--mode full` with `bf16` CUDA autocast on cloud GPU.
- [x] Verified `run_phase3_user_bt.py --mode local-test` on real Community Alignment slice (16 users, 6 prompts, 165 unique `(x, y)`, 212 pairs; 44.4s using cached ModernBERT CLS features) → `results/phase3_local_test/phase3_user_bt_diagnostics.png`:
  - `Shared (best wd=1e-3)`: `val_acc = 0.821`, `val_loss = 0.344`.
  - `UserBT (d=2, best wd=1.0)`: `val_acc = 0.718`, `val_loss = 0.639`, `beta_nat = 49.9`.
  - `UserBT (d=8, best wd=1.0)`: `val_acc = 0.821`, `val_loss = 0.345`, `beta_nat = 33.0`, `med_top1_maj = 0.25`.
  - `UserBT (d=32, best wd=1.0)`: `val_acc = 0.897`, `val_loss = 0.245` (down from `0.325` before WD sweep), `val_brier = 0.077`, `beta_nat = 42.5`, `med_top1_maj = 0.28`.
- [ ] Run `run_phase3_user_bt.py --mode full --device cuda` on cloud GPU (A100/H100) for all 839 users & 79,323 pairs and inspect full-scale Phase 3 diagnostics.

### Phase 4A: Mythical-User BT Policy, $\beta_{\mathrm{BT}}$ Resampling & KL-Constrained Distortion
- [x] Implement `src/mythical_bt.py` + `tests/test_mythical_bt.py`.
- [x] Implement `src/distortion.py` + `tests/test_distortion.py` (including Theorem 6 synthetic verification).
- [x] Implement `tests/test_e2e_smoke.py` and verify full local CPU pass (21/21 passing).
- [x] Implement standalone Phase 4A runner `run_phase4a_distortion.py` (`--mode local-test` with `SmolLM2-135M`; `--mode full` with `Qwen/Qwen3.5-0.8B`; supports raw `llm`, `llm_len_norm`, and `uniform` $\pi_{\mathrm{ref}}$ ablations + $\pi_{\mathrm{ref}}$ collapse diagnostics).
- [x] Verified `run_phase4a_distortion.py --mode local-test --also-len-norm-ref --also-uniform-ref` on real slice → `results/phase4a_local_test/`:
  - **Key finding on $\pi_{\mathrm{ref}}$**: raw unnormalized completion log-probs $\sum_t \log p(y_t \mid x, y_{<t})$ with `SmolLM2-135M` suffered **severe length-bias collapse** (`4/6` prompts collapsed to `max_prob = 1.000`, `median_norm_entropy = 0.000`, pushing `max_tau` to `75–159 nats`), whereas length-normalized log-probs $\frac{1}{T}\sum_t \log p(y_t \mid x, y_{<t})$ gave well-calibrated support (`0/6` collapsed, `median_norm_entropy = 0.982`, `max_tau = 3.4–4.9 nats`, empirical distortion up to `1.37`). Track both raw and length-normalized $\pi_{\mathrm{ref}}$ in the cloud `Qwen/Qwen3.5-0.8B` run and report this practical artifact.
- [ ] Run `run_phase4a_distortion.py --mode full --device cuda --llm Qwen/Qwen3.5-0.8B --also-len-norm-ref --also-uniform-ref` on cloud GPU (520 prompts, 839 users).

---

### Deferred (After Phase 4A Analysis)
- **Phase 4B — DPO Finetuning**: Train $\pi_{\mathrm{DPO}}$ with `trl.DPOTrainer` (local smoke test with `SmolLM2-135M`; cloud multi-seed runs on `Qwen/Qwen3.5-0.8B`, then 4B scale-up) and compare empirical DPO distortion against Phase 4A and Theorem 6.
- **Qualitative User-Space PCA Evaluation (`run_qualitative_pca.py`)**: Ready — selects top-10 highest & top-10 lowest distortion queries, projects all dataset candidate responses through frozen `ModernBERT-base` + trained linear head $W$ into 2D PCA space (colored by query distortion), generates $k$ completions from the DPO-trained LM (and optional base/ref LM), and projects them onto the exact same principal components to visualize before-vs-after DPO distributional shift. Verified locally with `tests/test_qualitative_pca.py` and `--mode local-test`.
