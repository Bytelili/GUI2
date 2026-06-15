# Server Runbook

All generated runtime data should remain outside the source folder.

Install the optional official-reference dependencies into the existing project
environment:

```bash
python -m pip install -r execution_pipeline_candidate/requirements-official-compat.txt
```

## 1. Audit the official reference

When the original FingerTip repository is available on the server:

```bash
python execution_pipeline_candidate/audit_official_reference.py \
  --reference-root /path/to/FingerTip-20K-main \
  --project-official-root data/official/fingertip20k \
  --output reports/execution_pipeline/official_reference_audit.json
```

Use `--source-only` only for a code snapshot that intentionally excludes
runtime official CSV files. Do not use it for the final server audit.

## 2. Prepare strict official-test tasks

```bash
cd /home/dumike/zyy/GUI2
source server_env.sh

python execution_pipeline_candidate/prepare_official_tasks.py \
  --project-config config.yaml \
  --output data/papo_tasks/execution_official_test_strict.jsonl

python execution_pipeline_candidate/build_runtime_templates.py \
  --tasks data/papo_tasks/execution_official_test_strict.jsonl \
  --output-dir reports/execution_pipeline/runtime_templates
```

This uses `test_execution.csv` only as targets and
`data/papo_protocol/execution_references.csv` only as historical references.
For the small official sample, copy `sampled_test_execution.csv` into the
project official-data directory and add
`--target-split sampled_test_execution.csv`.
Review every generated reset hook and author the success predicates before
copying them into the runtime experiment configuration.

## 3. Create runtime configuration

```bash
mkdir -p reports/execution_pipeline/config
cp execution_pipeline_candidate/experiment.example.json \
  reports/execution_pipeline/config/execution_strict_v1.json
```

Review the copied JSON before execution:

- Adapter paths must point to finalized clean-v2 best directories.
- Each adapter must contain `papo_training_provenance.json`.
- `device.app_hooks` must provide deterministic reset commands for every app.
- Success rules must be explicitly authored; missing rules remain unverified.
- `official_reference_audit_path` must point to the passed non-source-only
  server audit generated in step 1.

## 4. Preflight only

```bash
python execution_pipeline_candidate/prepare_experiment.py \
  --config reports/execution_pipeline/config/execution_strict_v1.json

python execution_pipeline_candidate/preflight_experiment.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json
```

Preparation refuses mismatched protocols, non-evaluation target actions,
unclean adapters, stale provenance, and comparison groups with no shared tasks.
Preflight then checks hashes again, verifies the ADB connection, requires reset
hooks for all task apps, confirms each task package is installed, records the
device fingerprint/display properties, and reports success-rule coverage before
model loading.

## 5. Small ADB checkpoint-selection run

Use a separate runtime config whose task file contains the fixed,
user-balanced subset. Then:

```bash
python execution_pipeline_candidate/run_matrix.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json \
  --limit 20

python execution_pipeline_candidate/score_results.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json

python execution_pipeline_candidate/audit_experiment.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json
```

Do not run full ADB evaluation until the small run has no retryable failures.

## 6. Full execution and human review

```bash
python execution_pipeline_candidate/run_matrix.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json

python execution_pipeline_candidate/prepare_success_annotations.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json \
  --output reports/execution_pipeline/strict_v1/success_annotations.csv
```

Fill only the `success`, `annotator`, and `evidence` columns for unverified
episodes. Then:

```bash
python execution_pipeline_candidate/score_results.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json \
  --annotations reports/execution_pipeline/strict_v1/success_annotations.csv

python execution_pipeline_candidate/analyze_experiment.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json

python execution_pipeline_candidate/evaluate_official.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json \
  --output-dir reports/execution_pipeline/strict_v1/official_metrics

python execution_pipeline_candidate/audit_experiment.py \
  --manifest reports/execution_pipeline/strict_v1/experiment_manifest.json \
  --require-paper-eligible
```

The final audit must pass before any number is copied into a paper.
The official evaluator writes separate metrics for every run and refuses
non-paper-eligible runs by default.
