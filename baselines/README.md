# FingerTip-20K Baselines

This directory contains baseline-only experiment code. PAPO-specific tree,
reward, and preference optimization code remains under `src/papo/`.

## Implemented baselines

- `no_history`: instruction and current screen only.
- `profile_only`: adds the user profile.
- `cross_user_icl`: adds the most similar different-age-group trajectory.
- `official_icl`: reproduces the official same-user Top-1 historical trajectory.
- `official_icl_no_same_intent`: excludes exactly matching historical intents.

All variants use the official personalized-execution prompt and the same
teacher-forced current observation and previous actions.

## Build the first offline experiment

```powershell
python baselines/build_execution_baselines.py `
  --raw_root "D:\0608DataSet\Raw" `
  --out_dir data/baselines/execution
```

The command writes:

- `execution_tasks.jsonl`: target episodes and all retrieval variants.
- `execution_steps.jsonl`: normalized test transitions.
- `teacher_forced_prompts.jsonl`: model-ready rows for every baseline variant.
- `retrieval_report.json`: same-user/cross-user retrieval quality.
- `retrieval_summary.csv`: compact retrieval table.
- `raw_test_audit.json`: test episode parsing quality.

For a quick smoke test:

```powershell
python baselines/build_execution_baselines.py --limit 3
```

After model inference, fill the `prediction` field in the prompt rows and run:

```powershell
python baselines/evaluate_predictions.py `
  --predictions data/baselines/execution/predictions.jsonl `
  --out data/baselines/execution/prediction_report.json
```

The evaluator reports parse validity, action-type accuracy, exact action
accuracy, coordinate distance, sequence-level user similarity, cross-user
similarity, and Sim2.

## SFT export

Build baseline prompts with `--test_split train_set.csv`, then export the
official-ICL rows for LLaMA-Factory:

```powershell
python baselines/export_llamafactory_sft.py `
  --prompts data/baselines/train/teacher_forced_prompts.jsonl `
  --variant official_icl `
  --out LLaMA-Factory/data/baselines/fingertip_execution_sft_all.json `
  --dataset_name fingertip_execution_sft_all `
  --dataset_info LLaMA-Factory/data/baselines/dataset_info.json
```

Training configurations matching the paper's Qwen-2.5-VL-7B LoRA rank-4
settings are under `baselines/configs/`.

## Immediate reference-copy lower bound

This policy copies the action at the same step from each variant's retrieved
trajectory. It is not an agent, but it reveals how much signal the retrieval
itself contains.

```powershell
python baselines/run_reference_copy.py `
  --prompts data/baselines/execution/teacher_forced_prompts.jsonl `
  --out data/baselines/execution/reference_copy_predictions.jsonl

python baselines/evaluate_predictions.py `
  --predictions data/baselines/execution/reference_copy_predictions.jsonl `
  --out data/baselines/execution/reference_copy_report.json
```

## OpenAI-compatible model inference

```powershell
$env:OPENAI_BASE_URL="http://localhost:8000/v1"
$env:OPENAI_API_KEY="EMPTY"
python baselines/run_openai_compatible.py `
  --prompts data/baselines/execution/teacher_forced_prompts.jsonl `
  --out data/baselines/execution/qwen_predictions.jsonl `
  --model Qwen2.5-VL-7B-Instruct `
  --variants no_history,profile_only,cross_user_icl,official_icl `
  --resume
```

## Visualizations

```powershell
python baselines/visualize_results.py `
  --input_dir data/baselines/execution `
  --out_dir outputs/visualizations/baseline
```

The script exports paper-ready PNG/PDF figures and their source CSV files.
