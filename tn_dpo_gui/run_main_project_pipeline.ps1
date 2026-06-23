$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$RootConfig = if ($args.Count -gt 0) { $args[0] } else { "..\\config.yaml" }

function Run-Step {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Run-Step "python -m tn_dpo_gui.scripts.preprocess_data --root-config `"$RootConfig`""
Run-Step "python -m tn_dpo_gui.scripts.build_user_index --root-config `"$RootConfig`""
Run-Step "python -m tn_dpo_gui.scripts.build_pairs --config configs/build_pairs.yaml --root-config `"$RootConfig`""
Run-Step "python -m tn_dpo_gui.scripts.train_ranker --config configs/train_ranker.yaml --root-config `"$RootConfig`""
Run-Step "python -m tn_dpo_gui.scripts.train_gate --config configs/train_gate.yaml --root-config `"$RootConfig`""
Run-Step "python -m tn_dpo_gui.scripts.eval --config configs/eval.yaml --root-config `"$RootConfig`""
