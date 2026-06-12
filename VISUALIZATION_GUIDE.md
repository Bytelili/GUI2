# Visualization Guide

Generate the current baseline visualization suite:

```powershell
python baselines/visualize_results.py `
  --input_dir data/baselines/execution `
  --out_dir outputs/visualizations/baseline
```

Every quantitative figure is exported as PNG, PDF, and CSV.

Generate dataset-level Introduction figures from all complete `total.csv`
episodes:

```powershell
python baselines/visualize_total_dataset.py `
  --raw_root "D:\0608DataSet\Raw" `
  --out_dir outputs/visualizations/total_dataset
```

These figures describe the dataset and personalized-behavior evidence. They
must not be presented as model-performance results.

Generate Introduction-focused personalization motivation figures:

```powershell
python baselines/visualize_intro_personalization.py `
  --episode_cache outputs/visualizations/total_dataset/total_episode_statistics.jsonl `
  --out_dir outputs/visualizations/introduction
```

These figures intentionally avoid general dataset statistics and focus on why
one-size-fits-all GUI execution is insufficient.

Generate the dark, teaser-style Introduction figures from the exported real
personalization statistics:

```powershell
python baselines/visualize_intro_fancy.py `
  --input_dir outputs/visualizations/introduction `
  --out_dir outputs/visualizations/introduction_fancy
```

The teaser's colored curves are a schematic visual encoding. All displayed
percentages, gains, category values, and landscape cells are computed from
real complete episodes.

Generate restrained, white-background academic figures using density plots,
ECDFs, violin distributions, raincloud plots, and scatter plots:

```powershell
python baselines/visualize_intro_academic.py `
  --input_dir outputs/visualizations/introduction `
  --out_dir outputs/visualizations/introduction_academic
```

Use `academic_fig06_introduction_composite` as the primary quantitative
Introduction figure. The remaining panels are exported separately for
analysis sections and appendices.

## Recommended Paper Use

### Introduction

- `fig01_retrieval_signal_overview`: primary quantitative motivation. It shows
  that same-user history contains substantially stronger action-path signal
  than cross-user or random history.
- `fig02_paired_personalization_gain`: stronger statistical evidence that the
  gain is visible episode by episode rather than being caused by a few outliers.
- `fig07_personalized_trajectory_case_study`: qualitative teaser illustrating
  different ways users execute a similar task.

### Dataset and Experimental Setup

- `fig04_action_space_distribution`: describes the action space represented in
  the execution test set.
- `fig05_task_length_distribution`: shows task difficulty and motivates
  length-stratified evaluation.

### Motivation and Analysis

- `fig03_intent_vs_trajectory_similarity`: shows that retrieving a similar
  intent does not fully determine a personalized action trajectory, motivating
  a preference-aware method.
- `fig06_reference_copy_tradeoff`: diagnostic only. It demonstrates that
  copying same-user history contains useful signal but remains an inadequate
  execution policy. Label it clearly as a retrieval-only lower bound.

## Suggested Introduction Narrative

1. Similar tasks admit different user-specific execution paths.
2. Same-user histories are markedly closer to target-user trajectories than
   cross-user histories.
3. Intent similarity alone does not reliably recover the preferred path.
4. Therefore, history should induce a personalized policy rather than merely
   being copied into the prompt.
