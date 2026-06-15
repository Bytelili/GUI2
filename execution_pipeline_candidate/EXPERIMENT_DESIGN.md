# Personalized-Execution Experiment Design

This candidate directory treats the existing strict training chain as a
read-only upstream dependency and completes the missing online evaluation
chain. No existing formal script is replaced.

## Research questions

1. Does execution SFT improve online task completion over the base model?
2. Does personalized Listwise training improve over ordinary execution SFT?
3. Does DPO improve verified success and trajectory quality over SFT/Listwise?
4. Is improvement caused by correct user history rather than extra context?
5. Are gains consistent across users instead of concentrated in a few users?

## Upstream checkpoints

- Base: Qwen2.5-VL-3B-Instruct.
- Execution SFT best checkpoint.
- Execution Listwise best checkpoint initialized from Execution SFT.
- Execution DPO best checkpoint initialized from Execution Listwise.

Every adapter must contain a passed `papo_training_provenance.json` whose
protocol ID and execution dataset names match the experiment. Adapter and
provenance hashes are frozen in the prepared manifest.

## Evaluation data

- Targets: complete episodes from official `test_execution.csv`.
- Personalization references: strict temporal
  `data/papo_protocol/execution_references.csv`.
- Same-track official-test episodes are excluded from references.
- Target actions are retained only for metrics and the replay smoke backend.
  They are never included in a real model prompt.

## Primary model comparison

Run Base, SFT, Listwise, and DPO with `correct_full_history` on exactly the same
task-ID intersection. Report each model separately and use paired comparisons:

- SFT versus Base.
- Listwise versus Base and SFT.
- DPO versus Base, SFT, and Listwise.

The example configuration includes comparisons to Base. Add the adjacent-stage
comparisons before the final run when those are needed in the paper.

## Personalization controls

Evaluate the strongest checkpoint with:

- `correct_full_history`: all selected same-user references.
- `correct_recent_history`: most recent same-user reference.
- `no_history`: removes same-user and cross-user history.
- `cross_user_history`: replaces same-user history with a cross-user reference.
- `shuffled_user_history`: random strictly earlier history from another user.
- `stale_history`: earliest available same-user reference.
- `truncated_history`: newest half of the available same-user references for a
  context-length control.

All runs connected by declared comparisons are automatically restricted to the
same eligible task IDs. The example uses a separate duplicate DPO full-history
run for the ablation component, preventing restrictive ablation eligibility
from shrinking the primary Base/SFT/Listwise/DPO comparison set.

## Online execution protocol

1. Run `prepare_official_tasks.py` and `prepare_experiment.py`.
2. Run `preflight_experiment.py` before loading a model.
3. Require one deterministic reset/cleanup hook for every app.
4. Execute one model action at a time through ADB.
5. Save every screenshot, UI XML, action, latency, token count, and error.
6. Stop at the configured limit or the official 2.5x golden-step limit.
7. Resume only when experiment identity and task hashes are unchanged.

Model errors, device errors, cleanup failures, duplicate IDs, and stale hashes
are hard failures. Replay runs are never paper eligible.

## Success verification

- `finished()` is not success.
- An automatic rule must have a source and at least one non-empty final-XML
  predicate.
- Tasks without a valid rule require manual review with annotator and evidence.
- A run is paper eligible only when every task has a verified success label and
  no retryable failure exists.

## Metrics

Official-format execution metrics:

- Verified success rate.
- Step ratio.
- Same-user trajectory similarity (`up_sim`).
- Cross-user trajectory similarity (`down_sim`).
- Personalized similarity (`up_sim / down_sim`).
- Time and token cost.

The standard similarity columns reproduce the official complete-action-text
`fuzzywuzzy` formula, including its `down_sim == 0 -> 0.4` fallback. Parsed
action-list Levenshtein metrics are exported separately as strict diagnostics.
Official `real_step` and similarity use every raw model output. When
`invalid_action_policy` is `official_wait`, malformed outputs execute `wait()`
as in the official parser while retaining the raw output for official scoring.

Additional paired statistics:

- Mean paired improvement and 95% bootstrap confidence interval.
- Task-level win rate.
- Macro-user mean improvement.
- User-cluster bootstrap confidence interval for macro-user improvement.
- Worst-user improvement.
- Fraction of users improved.

## Reporting rule

Only numbers from runs that pass:

1. strict task preparation,
2. experiment preflight,
3. complete online execution,
4. success verification,
5. scoring hash binding,
6. `audit_experiment.py --require-paper-eligible`,

may enter the main paper table. The bundled official evaluator is called once
per run so Base/SFT/Listwise/DPO metrics are never accidentally pooled. The
final audit also requires a bound, non-source-only official-reference audit.
