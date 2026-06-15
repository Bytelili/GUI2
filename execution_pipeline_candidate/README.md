# Isolated Personalized-Execution Pipeline

This directory is an isolated candidate implementation of the complete
personalized-execution experiment chain. It does not modify or replace the
existing project scripts.

## Evidence levels

- `replay`: deterministic local integration testing. It may use golden actions
  and is never paper eligible.
- `adb` with explicit final-state rules: online execution with automatically
  verified success labels.
- `adb` without an explicit rule: online execution whose success remains
  unverified until a human annotation is supplied.

The model emitting `finished()` is never treated as verified task success.

## Pipeline

1. `prepare_experiment.py`: validate strict execution tasks and build a
   model/condition run matrix.
2. `preflight_experiment.py`: verify task/model hashes, ADB connection, app
   hooks, and success-rule coverage before loading the model.
3. `run_matrix.py`: execute all selected runs with resumable per-task records.
4. `prepare_success_annotations.py`: create a human-review CSV for unverified
   ADB episodes.
5. `score_results.py`: merge verified labels and produce official-format
   execution result CSV files.
6. `audit_experiment.py`: validate coverage, hashes, verification status, and
   paper eligibility.
7. `evaluate_official.py`: summarize scored CSV files with the bundled
   FingerTip execution evaluator.
8. `audit_official_reference.py`: hash and contract-check the local project
   against the original FingerTip-20K source and CSV files.
9. `build_runtime_templates.py`: generate reviewable per-app reset hooks and
   per-task success-rule templates from the strict task file.

## Local end-to-end smoke test

```bash
python -m unittest discover -s execution_pipeline_candidate/tests -v
```

The tests use the replay backend and do not load a model or call ADB.

## Server preparation

First prepare strict official `test_execution.csv` tasks using only the formal
execution-reference partition:

```bash
python execution_pipeline_candidate/prepare_official_tasks.py \
  --project-config config.yaml \
  --output data/papo_tasks/execution_official_test_strict.jsonl
```

Copy `experiment.example.json` to a runtime location outside this source
directory, point `tasks_path` at that file, and update the model/device paths.
Then:

```bash
python execution_pipeline_candidate/prepare_experiment.py \
  --config /path/to/execution_experiment.json

python execution_pipeline_candidate/preflight_experiment.py \
  --manifest /path/to/output/experiment_manifest.json

python execution_pipeline_candidate/run_matrix.py \
  --manifest /path/to/output/experiment_manifest.json
```

For real ADB runs, every app should have a deterministic reset hook in
`device.app_hooks`. A run without a reset hook is rejected by default.

After execution:

```bash
python execution_pipeline_candidate/prepare_success_annotations.py \
  --manifest /path/to/output/experiment_manifest.json \
  --output /path/to/output/success_annotations.csv

python execution_pipeline_candidate/score_results.py \
  --manifest /path/to/output/experiment_manifest.json \
  --annotations /path/to/output/success_annotations.csv

python execution_pipeline_candidate/audit_experiment.py \
  --manifest /path/to/output/experiment_manifest.json \
  --require-paper-eligible
```

The scored CSV files contain the columns consumed by the existing FingerTip
execution evaluator:

`success, origin_step, real_step, step_ratio, up_sim, down_sim, similarity,
time, token`

They also preserve `success_verified`, trajectories, termination reason,
checkpoint identity, provenance, and strict action-list diagnostic metrics.

See `SERVER_RUNBOOK.md` for the ordered server commands and
`PIPELINE_STATUS.md` for the implemented boundary. See `EXPERIMENT_DESIGN.md`
for the research questions, comparison matrix, controls, and reporting rules.
See `OFFICIAL_COMPATIBILITY.md` for the exact official-reference boundary.
