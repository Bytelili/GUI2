# PAPO Offline Implementation Status

This document maps the proposed PAPO architecture to the current offline
implementation. "Proxy" means the module exists but replaces an unavailable
online/model signal with an offline-data estimate.

| PAPO module | Status | Current implementation | Remaining gap |
|---|---|---|---|
| GUI action canonicalization | Implemented | Raw coordinates are grounded to XML nodes and converted into semantic actions in `src/papo/raw_builder.py`. | Repair source-text encoding and improve cross-screen semantic alignment. |
| User history retrieval | Proxy | Strict-past Top-K same-user episodes are retrieved and included in model prompts. | No learned `f_hist` plus attention encoder; the VLM currently encodes the textual history directly. |
| Positive/negative reference retrieval | Implemented | Strict-past same-user and cross-user Top-K references with intent similarity and age-group filtering. | Intent similarity is lightweight rather than embedding-based. |
| Executable counterfactual tree | Proxy | Logged transitions from observed, same-user, and cross-user episodes form an offline transition tree. | No frozen base-policy candidates, UI-affordance expansion, or real environment execution. Executability is supported-transition validity, not emulator verification. |
| FingerTip-specific trajectory evidence | Implemented with offline task proxy | Compiles positive/negative trajectory LevSim into zero-centered `tanh(log(same/cross)/T)` evidence and applies `R_u = R_0(1 + eta R_pref)`. | This is a dataset-specific evidence compiler, not the core PAPO contribution. `R_0` feasibility should be strengthened after text encoding is repaired. |
| Leaf reward propagation | Implemented | Weighted reachable-leaf averages produce `Q_u`, `Q_0`, and `Q_pref`. | Leaf weights are support-based proxies rather than base-policy rollout probabilities. |
| Conservative value correction | Implemented | Penalizes `Q_u` using support and leaf-reward dispersion. | Uncertainty calibration has not been tuned on held-out data. |
| Residual personalized utility | Implemented | Computes `D_u = conservative_Q_u - Q_0` as `a_delta`. | None for the offline formulation. |
| KL closed-form target policy | Proxy | Produces normalized listwise targets using `support_prior * exp(D_u / beta)`. | Support prior approximates frozen `pi_0`; real base-policy probabilities are not yet scored. |
| Listwise target generation and training | Implemented | Expands each state target distribution into weighted candidate sequences and trains target-policy weighted sequence NLL through LLaMA-Factory. | Must be run on the server to verify a real GPU training step. |
| Pairwise PAPO surrogate | Implemented | Generates `p*=sigmoid(gap/beta)`, confidence weights, and trains soft-label reference-ratio BCE through the bundled LLaMA-Factory. | Must be run on the server to verify a real GPU training step. |
| Mixed listwise/pairwise/BC objective | Partial | SFT warm-up, listwise PAPO, and pairwise PAPO can be run sequentially. | No single Trainer jointly optimizing listwise, pairwise, and BC losses. |
| Coverage estimation | Implemented offline | Computes support-based `C(n,a)`. | Coverage-gated model inference and action selection are not implemented. |
| Proactive suggestion | Partial | Proactive task construction and SFT export are implemented. | PAPO preference optimization is currently only applied to personalized execution. |

## Current Offline Training Path

```text
FingerTip raw data
  -> semantic PAPO steps
  -> strict-past personalized references
  -> logged-transition counterfactual trees
  -> FingerTip-specific trajectory evidence
  -> Q_u, Q_0, conservative residual D_u
  -> closed-form listwise targets
  -> target-policy weighted listwise distillation
  -> soft-label weighted PAPO pairwise refinement
```

The current system directly trains the offline PAPO listwise target policy and
can then apply pairwise PAPO refinement. It is still an offline proxy because
the base policy prior is support-estimated rather than scored by a frozen
model.
