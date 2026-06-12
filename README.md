# PAPO Experiment Pipeline

Paper-aligned non-PAPO baselines now live under [`baselines/`](baselines/).
Start by building the complete offline personalized-execution baseline set:

```powershell
python baselines/build_execution_baselines.py `
  --raw_root "D:\0608DataSet\Raw" `
  --out_dir data/baselines/execution
```

See [`baselines/README.md`](baselines/README.md) for retrieval experiments,
teacher-forced model inference, evaluation, and Qwen-2.5-VL-7B SFT export.

Detailed artifact schemas and JSON/JSONL field definitions are documented in
[`DATA_FORMAT.md`](DATA_FORMAT.md).

The repository includes a pinned LLaMA-Factory checkout under
`LLaMA-Factory/`. Upload the entire `GUI` directory to the training server,
then follow [`server/README.md`](server/README.md).

`config.yaml` is the single entry point for dataset paths, PAPO tree/reward/
value parameters, model selection, and training hyperparameters. Environment
variables such as `PAPO_RAW_ROOT` and `QWEN_MODEL_PATH` override
machine-specific paths without editing the file.

Check all configured dataset/model/output paths before building:

```bash
python scripts/12_validate_config_paths.py --config config.yaml
```

The simplest server workflow is:

```bash
bash server/setup_server.sh
RAW_ROOT=/datasets/FingerTip/Raw bash server/prepare_train_data.sh
bash server/train_auto_resume.sh configs/llamafactory/generated/proactive_sft.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_sft.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_listwise.yaml
bash server/train_auto_resume.sh configs/llamafactory/generated/execution_dpo.yaml
```

To prepare and validate only the Proactive Suggestion track before any
Execution work:

```bash
bash server/prepare_proactive_data.sh
bash server/train_auto_resume.sh configs/llamafactory/generated/proactive_sft.yaml
```

The preparation command builds only Proactive train/eval exports, validates
their images and provenance, and runs the strict training preflight. It does
not build Execution artifacts or start training.

`prepare_train_data.sh` first creates deterministic per-user temporal
train/eval partitions. Proactive histories and Execution references come only
from the corresponding train partition, and same-track official test episode
keys are hard-excluded. Formal configs use explicit eval datasets and new
`*_clean_v2` output directories. `train_auto_resume.sh` verifies protocol and
dataset hashes before every run, refuses stale resumes, and copies the
lowest-eval-loss checkpoint to a stable `*_best` directory after training.

For a local smoke run:

```powershell
python scripts/14_build_data_protocol.py --config config.yaml
python scripts/09_run_config_pipeline.py --config config.yaml --limit 3
python scripts/10_render_training_configs.py --config config.yaml
python scripts/08_validate_llamafactory_data.py --dataset_dir LLaMA-Factory/data/papo --check_images
python scripts/11_validate_papo_artifacts.py --work_dir data/papo_config_run
```

The standalone `proactive_suggestion.py` and `personalized_execution.py`
defaults reproduce official test-task construction for evaluation. They are
not formal training-data entry points; formal training must go through
`scripts/14_build_data_protocol.py` and `scripts/09_run_config_pipeline.py`.

The config pipeline writes PAPO residual action values, listwise target
distributions, and pairwise DPO approximations to `data/papo_config_run/`.
FingerTip-specific same-user versus cross-user trajectory similarity is treated
as leaf-level personalization evidence. By default, `evidence_transform:
tanh_log_ratio` makes this evidence zero-centered: positive values favor the
target user's history and negative values indicate stronger cross-user
similarity.
The bundled LLaMA-Factory has a project-local `pref_loss: papo` extension that
passes each exported `papo_weight` and soft preference target through its data
pipeline and minimizes normalized weighted pairwise BCE.

The bundled LLaMA-Factory also supports direct offline PAPO target-policy
distillation with `use_papo_listwise: true`. Each candidate action is weighted
by its closed-form `target_policy_probability`, giving the listwise objective
`-sum_a pi_star(a|n) log pi_theta(a|n,H_u)`.

PAPO DPO loss:

```text
target_i = sigmoid(advantage_gap_i / beta)
model_i = sigmoid((log pi_theta(chosen) - log pi_0(chosen))
                  - (log pi_theta(rejected) - log pi_0(rejected)))
L_PAPO = sum_i(weight_i * BCE(target_i, model_i)) / sum_i(weight_i)
weight_i = clip(advantage_gap_i / tau_m, 0, max_weight)
```

This repository contains only the PAPO experiment line. It builds PAPO records
directly from raw FingerTip-20K trajectories, constructs offline
counterfactual trees, propagates task and user rewards, and exports DPO
preference pairs.

## Layout

```text
src/papo/       PAPO implementation
scripts/        Pipeline entry points
data/raw/       Raw FingerTip-20K trajectories
data/papo_raw/  PAPO records and generated artifacts
```

## Run Order

```powershell
python scripts/00_build_papo_from_raw.py `
  --raw_root data/raw/fingertip20k `
  --out_dir data/papo_raw

python scripts/01_build_papo_trees.py `
  --steps data/papo_raw/papo_steps.jsonl `
  --out data/papo_raw/papo_trees.jsonl `
  --mode offline `
  --max_depth 3

python scripts/04_summarize_papo_trees.py `
  --trees data/papo_raw/papo_trees.jsonl `
  --out data/papo_raw/papo_tree_summary.json

python scripts/02_propagate_papo_advantages.py `
  --trees data/papo_raw/papo_trees.jsonl `
  --out data/papo_raw/papo_advantages.jsonl

python scripts/03_export_papo_dpo_pairs.py `
  --advantages data/papo_raw/papo_advantages.jsonl `
  --out data/papo_raw/papo_dpo_pairs.jsonl `
  --summary_out data/papo_raw/papo_dpo_summary.json
```

The pipeline uses raw `action.jsonl`, `survey_result.json`, screenshots, and
XML trees. It does not require intermediate records from another experiment
line.

## FingerTip Task Tracks

The official FingerTip evaluation flow is exposed as two offline task builders:

```powershell
python proactive_suggestion.py --screenshot_level 0
python personalized_execution.py
```

`proactive_suggestion.py` builds intent-prediction records from the user
profile, current time/scenario, strictly earlier same-user intents, and zero to
three initial screenshots.

`personalized_execution.py` builds execution records from the instruction,
profile, initial observation, the most similar strictly earlier same-user
action trajectory, and a cross-user counterfactual trajectory. Ground-truth
actions are kept in the `target` section for evaluation and are never included
in the model input.

The intended end-to-end flow is:

```text
Official CSV metadata + raw episodes
  -> proactive suggestion task records
  -> personalized execution task records
  -> PAPO steps and counterfactual trees
  -> propagated advantages
  -> DPO preference pairs
```

Execution task metadata contains `papo_root_step_id` and `papo_tree_id` so task
records can be joined directly with the existing PAPO artifacts.

## Full FingerTip Data Flow

While the archive is still extracting, inspect coverage without consuming
incomplete episodes:

```powershell
python scripts/05_audit_fingertip_dataset.py `
  --raw_root "D:\0608DataSet\Raw" `
  --out "D:\0608DataSet\papo\dataset_audit.json"
```

For personalized execution, build a retrieval pool from both the official
training split and execution targets. Then build trees only for the first step
of execution-test episodes:

```powershell
python scripts/00_build_papo_from_raw.py `
  --raw_root "D:\0608DataSet\Raw" `
  --catalog data/official/fingertip20k/train_set.csv `
  --catalog data/official/fingertip20k/test_execution.csv `
  --out_dir "D:\0608DataSet\papo\execution_pool"

python scripts/01_build_papo_trees.py `
  --steps "D:\0608DataSet\papo\execution_pool\papo_steps.jsonl" `
  --out "D:\0608DataSet\papo\execution_trees.jsonl" `
  --root_catalog data/official/fingertip20k/test_execution.csv `
  --root_only `
  --mode offline
```

Compute paper-aligned metrics and offline PAPO proxy metrics:

```powershell
python scripts/06_evaluate_paper_metrics.py `
  --suggestion_tasks data/papo_tasks/proactive_global_100.jsonl `
  --execution_tasks data/papo_tasks/execution_global_100_age_group.jsonl `
  --trees data/papo_balanced_smoke/papo_trees.jsonl `
  --out data/papo_tasks/paper_metrics_report.json
```

All raw and task builders skip incomplete extracted episodes by default.
Cross-user and same-user references are restricted to data strictly earlier
than the target episode.

For a balanced smoke test during extraction, sample a small number of episodes
from every currently available user:

```powershell
python scripts/00_build_papo_from_raw.py `
  --raw_root "D:\0608DataSet\Raw" `
  --max_episodes_per_user 2 `
  --out_dir "D:\0608DataSet\papo\balanced_smoke"
```
