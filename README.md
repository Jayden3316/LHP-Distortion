# LHP-Distortion

**Empirical Distortion of Preference Alignment in Practical Settings**

This repository investigates how much *utilitarian distortion* arises in practical preference-alignment pipelines (single-reward Bradley–Terry RLHF and DPO) relative to the theoretical bounds established in Gölz, Haghtalab, & Yang (2025), *"Distortion of AI Alignment: Does Preference Optimization Optimize for Preferences?"* ([arXiv:2505.23749](https://arxiv.org/abs/2505.23749)).

---

## 1. Problem Formulation & Notation

To avoid overloading $\beta$ (which denotes the Bradley–Terry inverse temperature in Gölz et al. and the KL-penalty in DPO), we use the following notation throughout the project:

| Symbol | Meaning |
| :--- | :--- |
| $x$ | Input prompt / query |
| $\mathcal{A}(x)$ | Finite set of candidate responses $y$ for query $x$ |
| $\phi(x, y) \in \mathbb{R}^d$ | Prompt–response feature representation (e.g., LLM hidden state + linear projection) |
| $w_i \in \mathbb{R}^d$ | Low-rank preference embedding for user $i \in \{1, \dots, N\}$ |
| $u_i(y \mid x) \in [0, 1]$ | Estimated cardinal utility of user $i$ for response $y$ to query $x$ |
| $\mathrm{AvgUtil}(y \mid x)$ | Utilitarian social welfare: $\frac{1}{N} \sum_{i=1}^N u_i(y \mid x)$, extended to policies as $\mathrm{AvgUtil}(\pi \mid x) = \mathbb{E}_{y \sim \pi(\cdot \mid x)}[\mathrm{AvgUtil}(y \mid x)]$ |
| $\beta_{\mathrm{BT}} > 0$ | User-level Bradley–Terry inverse temperature: $P(y_1 \succ_i y_2 \mid x) = \sigma\!\big(\beta_{\mathrm{BT}}(u_i(y_1 \mid x) - u_i(y_2 \mid x))\big)$ |
| $\lambda > 0$ (or $\beta_{\mathrm{DPO}}$) | KL-regularization strength in RLHF / DPO |
| $\pi_{\mathrm{ref}}(\cdot \mid x)$ | Reference policy distribution over $\mathcal{A}(x)$ |
| $\tau(x)$ | Realized KL radius for policy $\pi$ on query $x$: $\tau(x) := D_{\mathrm{KL}}\!\big(\pi(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)$ |
| $B_{\tau}(\pi_{\mathrm{ref}})$ | KL ball around reference policy: $\{\pi' \in \Delta(\mathcal{A}(x)) : D_{\mathrm{KL}}(\pi' \| \pi_{\mathrm{ref}}) \le \tau\}$ |

### Distortion Metric (Gölz et al., 2025, Sec. 2 & Prop. 14)

By Proposition 14 (Appendix E.4), any $\lambda$-regularized RLHF/DPO policy $\pi$ is optimal for the $\tau$-constrained problem with $\tau(x) = D_{\mathrm{KL}}(\pi(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x))$. For each query $x$, the **empirical KL-constrained distortion** is:

$$\mathrm{dist}(x; \pi, \pi_{\mathrm{ref}}) = \frac{\max_{\pi^\star \in B_{\tau(x)}(\pi_{\mathrm{ref}}(\cdot \mid x))} \mathrm{AvgUtil}(\pi^\star \mid x)}{\mathrm{AvgUtil}(\pi(\cdot \mid x) \mid x)}$$

where the numerator's KL-constrained social-welfare maximizer has the exact Gibbs form
$$\pi^\star(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x) \exp\!\big(\mathrm{AvgUtil}(y \mid x) / \lambda'(x)\big),$$
with $\lambda'(x) \ge 0$ found via 1D bisection so that $D_{\mathrm{KL}}(\pi^\star \| \pi_{\mathrm{ref}}) = \tau(x)$.

In the unconstrained limit ($\tau \to \infty$), this reduces to **social-choice distortion**: $\frac{\max_{y \in \mathcal{A}(x)} \mathrm{AvgUtil}(y \mid x)}{\mathrm{AvgUtil}(\pi(\cdot \mid x) \mid x)}$.

### Theoretical Benchmarks (Functions of $\beta_{\mathrm{BT}}$)

- **Universal lower bound & NLHF upper bound** (Thms. 3 & 7, Cor. 8): $\frac{\beta_{\mathrm{BT}}}{2} \cdot \frac{1 + e^{-\beta_{\mathrm{BT}}}}{1 - e^{-\beta_{\mathrm{BT}}}}$
- **Unconstrained RLHF / Borda lower bound** (Thms. 5 & 12): $\max_{0 < \gamma < 1} \frac{\beta_{\mathrm{BT}}}{2}\frac{1+e^{-\beta_{\mathrm{BT}}}}{1-e^{-\beta_{\mathrm{BT}}}}\!\left(1 - \gamma + \frac{\sigma(\beta_{\mathrm{BT}}\gamma) - 1/2}{\sigma(\beta_{\mathrm{BT}}) - 1/2}\right) \ge (1 - o(1))\beta_{\mathrm{BT}}$
- **Unconstrained RLHF / Borda upper bound** (Thm. 2): $\left(\frac{\beta_{\mathrm{BT}}}{2} \cdot \frac{1 + e^{-\beta_{\mathrm{BT}}}}{1 - e^{-\beta_{\mathrm{BT}}}}\right)^2 = O(\beta_{\mathrm{BT}}^2)$
- **KL-constrained RLHF / DPO worst-case lower bound** (Thm. 6): $e^{\Omega(\beta_{\mathrm{BT}})}$ (and unbounded in $\beta_{\mathrm{BT}}$ under correlated pair sampling, Thm. 9)

---

## 2. Dataset

We initially explored `HannahRoseKirk/prism-alignment`, which has 1,396 active users and 68,371 scored utterances, but found **no meaningful shared-prompt overlap** across users (each user writes their own prompts; the most repeated substantive opening prompt appears for only 13 users).

We therefore switched to the **English split** (`assigned_lang == "en"`) of [`facebook/community-alignment-dataset`](https://huggingface.co/datasets/facebook/community-alignment-dataset), streamed and filtered locally to `data/community-alignment/community_alignment_en.csv` (517.7 MB):

- **1,331** unique English annotators
- **30,856** conversations (up to 4 turns each; 4 candidate responses `response_a`–`response_d` per turn)
- **79,514** preference queries across turns 1–4 (median **67** queries and **26** conversations per user; mean **59.7** queries)
- **Strong prompt overlap**: 1,033 pregenerated first-turn prompts, **560** rated by $\ge 10$ users (median **31** users/prompt, max **221**), with ~70–136 unique candidate responses per heavily-rated prompt.

---

## 3. Experimental Pipeline

1. **Phase 0–1 — Local Test Harness & Turn-1 Dataset Construction**
   - Filter `community_alignment_en.csv` to Turn-1 preferences from users with $K > 20$ Turn-1 choices (**839 users**, **26,467 rows**, **79,401 pairwise comparisons**, **45,797 unique `(x, y)` pairs**, **520 prompts** with $\ge 20$ users).
   - Maintain fast CPU unit/smoke tests (using `HuggingFaceTB/SmolLM2-135M` and a tiny dataset fixture) before running full-scale jobs on cloud A100/H100 GPUs.
2. **Phase 2 — Frozen ModernBERT Representations & Reference Policy $\pi_{\mathrm{ref}}$**
   - Encode prompt–response pairs with frozen `answerdotai/ModernBERT-base` + trainable linear projection head $W \in \mathbb{R}^{d \times 768}$.
   - Compute reference policy $\pi_{\mathrm{ref}}(y \mid x) \propto \exp\!\big(\sum_{t=1}^T \log p_{\pi_0}(y_t \mid x, y_{<t})\big)$ normalized over candidate responses $y \in \mathcal{A}(x)$.
3. **Phase 3 — Individual User-Wise BT Models & Diagnostics**
   - Fit purely user-ID low-rank embeddings $w_i \in \mathbb{R}^d$ ($d \in \{2, 8, 32\}$) with unconstrained bilinear logits $s_i(x, y) = \langle w_i, \phi(x, y) \rangle$ under pairwise BT loss.
   - Normalize fitted scores to $[0, 1]$ via **global affine scaling** $\tilde{u}_i(y \mid x) = \frac{s_i(x, y) - s_{\min}}{s_{\max} - s_{\min}}$, preserving all BT probabilities identically with implied $\beta_{\mathrm{BT}}^{\mathrm{nat}} = s_{\max} - s_{\min}$.
   - Audit fit quality, calibration, and inter-user preference heterogeneity before moving to policy evaluation.
4. **Phase 4A — Mythical-User BT Policy & Theorem 6 Distortion Analysis**
   - Fit a single shared $\bar{w}$ BT-MLE reward $r_{\mathrm{MLE}}(x, y)$ (on observed pairs and on resampled preference datasets drawn from $\tilde{u}_i$ at higher $\beta_{\mathrm{BT}}$ values).
   - Compute closed-form KL-regularized policy $\pi_\lambda(y \mid x) \propto \pi_{\mathrm{ref}}(y \mid x) \exp(r_{\mathrm{MLE}}(x, y)/\lambda)$, realized $\tau(x) = D_{\mathrm{KL}}(\pi_\lambda(\cdot \mid x) \| \pi_{\mathrm{ref}}(\cdot \mid x))$, exact KL-ball optimal $\mathrm{AvgUtil}(\pi^\star \mid x)$, and multi-panel $(\tau, \mathrm{distortion})$ scatter plots.
5. **Phase 4B (Next) — DPO Finetuning (`trl.DPOTrainer`)**
   - Fine-tune $\pi_0$ (`Qwen/Qwen3.5-0.8B` initially, then a 4B model; `SmolLM2-135M` for local CPU tests) with TRL's `DPOTrainer` (multi-seed, multi-$\beta_{\mathrm{DPO}}$) and compare empirical DPO distortion against Phase 4A and Theorem 6.

---

## 4. Repository Structure

```text
LHP-Distortion/
├── README.md                                   # Project overview & formulation
├── LOG.md                                      # Experiment log & design-decision tracker
├── analyze_prism_users.py                      # PRISM per-user count analysis
├── prism_dataset_exploration.ipynb             # Executed PRISM exploration notebook
├── download_community_alignment_en.py          # Stream & filter EN split of Community Alignment
├── analyze_community_alignment_en.py           # EN split user query counts & prompt overlap
├── data/                                       # Local datasets (gitignored)
│   ├── prism-alignment/
│   └── community-alignment/
└── results/                                    # Summary CSVs and figures
```

See [LOG.md](LOG.md) for detailed progress tracking and open design decisions.
