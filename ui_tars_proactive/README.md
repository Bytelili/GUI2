# UI-TARS Proactive Evaluation

This directory is an isolated experiment line for UI-TARS-7B on the clean-v2
Proactive suggestion protocol. It does not modify the existing Qwen results.

The chain compares:

1. `ui_tars_7b_base`: `/home/dumike/zyy/GUI/backbone/UI-TARS-7B` without tuning.
2. `ui_tars_7b_sft`: the same backbone after clean-v2 Proactive SFT.

Both models are evaluated with the same official FingerTip Proactive similarity
implementation at screenshot levels 0, 1, 2, and 3.

Outputs:

- `reports/ui_tars_proactive/<model>/<mode>/level_<n>/...`
- `reports/ui_tars_proactive/summary/<mode>_ui_tars_level_results.csv`
- `reports/ui_tars_proactive/summary/<mode>_ui_tars_level_results.md`
- `reports/ui_tars_proactive/summary/<mode>_ui_tars_sft_minus_base.csv`

The default training config is conservative for four 96GB GPUs:
`per_device_train_batch_size=4`, `gradient_accumulation_steps=4`,
`gradient_checkpointing=true`, effective global batch size 64.
