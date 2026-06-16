# Proactive Personalized Preference Pipeline

This directory adds a new research line after the clean-v2 Proactive SFT model.
It does not modify the existing SFT datasets, checkpoints, official evaluation,
or personalized-execution pipeline.

## Research Question

Can a Proactive suggestion model learn to rank a target user's likely intent
above plausible same-user alternatives, cross-user hard negatives, and sampled
SFT alternatives?

The pipeline deliberately does **not** claim to solve suggestion triggering or
abstention. FingerTip-20K contains positive intent episodes but no reliable
negative-trigger labels. Abstention requires a separately designed and audited
annotation protocol.

## Training Chain

```text
strict temporal Proactive train/eval tasks
  -> optional sampled candidates from clean-v2 SFT
  -> same-user and cross-user hard candidates from train partition only
  -> decomposed, auditable reward
  -> target-policy weighted Listwise training
  -> weighted soft-target PAPO-DPO training
  -> strict-holdout official Proactive evaluation at Levels 0/1/2/3
```

The decomposed reward is:

```text
R = 0.55 R_task + 0.20 R_user + 0.15 R_context + 0.10 R_specificity
```

- `R_task`: supervised similarity to the hidden train/eval target.
- `R_user`: same-user history similarity relative to cross-user candidates.
- `R_context`: scenario and time-of-day compatibility.
- `R_specificity`: deterministic penalty for empty, vague, or overly long text.

Every component, source episode, final reward, listwise probability, DPO gap,
weight, and soft preference target is stored in the generated artifacts.

Before export, the candidate-quality gate labels every non-oracle candidate as
`valid_hard_negative`, `easy_negative`, `pseudo_negative`, or `invalid`.
Structural failures block training; proxy semantic concerns are reported as
warnings for sampled human review. Invalid candidates are excluded from
Listwise export, and pseudo/invalid candidates can never become DPO rejected
responses. Reports are written to:

- `data/proactive_preference/candidate_quality_report.json`
- `data/proactive_preference/candidate_quality_flags.jsonl`
- `data/proactive_preference/candidate_quality_review_sample.jsonl`
- `data/proactive_preference/candidate_quality_excluded_targets.jsonl`

The default proxy tiers are explicit and configurable:

- `invalid`: empty/too-short, corrupted, control-character, or repeated-character output;
- `pseudo_negative`: lexical target similarity at least `0.92`, conservatively excluded from DPO rejection;
- `easy_negative`: both target and same-user-history similarity at most `0.20`;
- `valid_hard_negative`: structurally valid remaining alternatives.

These labels are not semantic ground truth. The balanced review sample must be
inspected before a main paper run, especially for near-target candidates whose
small textual differences may still change the task.

If a strict-train oracle target is structurally invalid, the pipeline excludes
that target from preference optimization and records it in
`candidate_quality_excluded_targets.jsonl`. Eval oracle targets are never
silently excluded.

## Safety Properties

- Train and temporal eval targets remain disjoint.
- Candidate references for both partitions come only from the strict Proactive
  training partition and are strictly earlier than the target.
- Official `test_suggestion.csv` is never used for training or model selection.
- Current clean-v2 SFT adapter provenance is required before candidate sampling.
- DPO cannot start until the finalized Listwise best adapter has passed strict
  provenance validation.
- Every training launch re-hashes preference datasets against the audited
  `preference_manifest.json`.
- Existing datasets and checkpoints use different names and are not overwritten.

## Server Run Order

After pulling the code on the server:

```bash
cd /home/dumike/zyy/GUI2
source server_env.sh

# Recommended: sample four diverse candidates from the clean SFT model.
bash proactive_preference_pipeline/run_pipeline.sh generate

# Build and audit preference datasets, render configs, and run preflight.
bash proactive_preference_pipeline/run_pipeline.sh prepare

# Train and finalize the best Listwise checkpoint.
bash proactive_preference_pipeline/run_pipeline.sh listwise

# Train and finalize PAPO-DPO from the best Listwise checkpoint.
bash proactive_preference_pipeline/run_pipeline.sh dpo

# Run the same official strict-holdout Level 0/1/2/3 evaluation for both models.
bash proactive_preference_pipeline/evaluate_models.sh

# Display manifests, configs, checkpoints, and GPU status.
bash proactive_preference_pipeline/run_pipeline.sh audit
```

`prepare` also works before `generate`; it then builds a valid first experiment
using oracle targets plus same-user and cross-user historical candidates. The
recommended main method includes sampled SFT candidates.

## Required Experiments

Main comparison:

1. `Qwen2.5-VL-3B-Instruct`
2. `clean-v2 Proactive SFT`
3. `SFT + personalized Listwise`
4. `SFT + personalized Listwise + PAPO-DPO`

Required ablations:

- remove `R_user`;
- remove cross-user hard negatives;
- remove sampled SFT candidates;
- use only `R_task`;
- evaluate every trained checkpoint at screenshot Levels 0, 1, 2, and 3.

Report official similarity together with user-cluster bootstrap confidence
intervals, per-user variance, wrong-user preference rate, reward margin, and
inference cost. Treat official-online history as supplementary only; the main
result must use strict-holdout history.
