$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-Step {
    param([string]$Command)
    Invoke-Expression $Command
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Run-Step "python -m tn_dpo_gui.scripts.preprocess_data --demo --output-dir data/demo"
Run-Step "python -m tn_dpo_gui.scripts.build_user_index --trajectories data/demo/trajectories.jsonl --output data/demo/user_index.json"
Run-Step "python -m tn_dpo_gui.scripts.build_pairs --config configs/build_pairs.demo.yaml"
Run-Step "python -m tn_dpo_gui.scripts.train_ranker --config configs/train_ranker.demo.yaml"
Run-Step "python -m tn_dpo_gui.scripts.train_gate --config configs/train_gate.demo.yaml"
Run-Step "python -m tn_dpo_gui.scripts.eval --config configs/eval.demo.yaml"
