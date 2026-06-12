# FingerTip-20K Baseline Status

## Completed

The baseline experiment layer is implemented under `baselines/` and was
validated against the official FingerTip-20K personalized-execution code.

Complete offline test artifacts were generated from:

```text
Raw data: D:\0608DataSet\Raw
Official split: data/official/fingertip20k/test_execution.csv
Output: data/baselines/execution
```

Artifact counts:

```text
Available execution-test episodes: 195 / 200
Parsed test steps: 2,097
Valid teacher-forced steps: 2,046
Baseline prompt variants: 5
Teacher-forced prompt rows: 10,230
```

## Retrieval Result

| Retrieval mode | Intent similarity | Whole-sequence similarity | Action Levenshtein similarity |
|---|---:|---:|---:|
| Same-user Top-1 | 0.8420 | 0.4263 | 0.2344 |
| Same-user Top-1, excluding same intent | 0.7949 | 0.3854 | 0.2275 |
| Different-type cross-user Top-1 | 0.5914 | 0.2474 | 0.1412 |
| Random same-user | 0.2658 | 0.2531 | 0.1456 |
| Random cross-user | 0.1375 | 0.2259 | 0.1181 |

Same-user Top-1 improves whole-sequence similarity over cross-user Top-1 by
`0.1789`. Excluding exactly matching intents retains most of the gain. This
confirms that the dataset contains a measurable personalized execution signal.

## Implemented Variants

```text
no_history
profile_only
cross_user_icl
official_icl
official_icl_no_same_intent
```

## Next Experiment

Run the five variants through the same base VLM using
`baselines/run_openai_compatible.py`, then evaluate with
`baselines/evaluate_predictions.py`.

The first trained reproduction should use:

```text
Model: Qwen-2.5-VL-7B-Instruct
Data: stratified 1,000 training episodes
Method: LoRA rank 4
Prompt variant: official_icl
```

Build the 1,000-episode training sample:

```powershell
python baselines/build_execution_baselines.py `
  --raw_root "D:\0608DataSet\Raw" `
  --test_split train_set.csv `
  --sample_size 1000 `
  --variants official_icl `
  --retrieval_modes same_user_top1,cross_user_top1 `
  --out_dir data/baselines/train_1k
```

