from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the lowest-eval-loss checkpoint without loading it on GPU.")
    parser.add_argument("--training-config", required=True)
    args = parser.parse_args()

    training_path = Path(args.training_config).resolve()
    training = yaml.safe_load(training_path.read_text(encoding="utf-8"))
    output_dir = Path(str(training["output_dir"])).resolve()
    gate_path = output_dir / "papo_preflight.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Cannot finalize an ungated run: {gate_path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    selected, metric, step = select_best_checkpoint(output_dir)
    target = output_dir.with_name(output_dir.name + "_best")
    _replace_stable_directory(selected, target, output_dir)
    provenance = {
        **gate,
        "status": "passed",
        "source_checkpoint": str(selected),
        "best_eval_loss": metric,
        "best_step": step,
        "stable_model_dir": str(target),
    }
    (target / "papo_training_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    print("BEST CHECKPOINT FINALIZATION PASSED")


def select_best_checkpoint(output_dir: Path) -> tuple[Path, float, int]:
    records: dict[int, float] = {}
    candidates = sorted(output_dir.glob("checkpoint-*"), key=_checkpoint_step)
    for checkpoint in candidates:
        state_path = checkpoint / "trainer_state.json"
        if not state_path.exists():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for item in state.get("log_history", []):
            if "eval_loss" in item and "step" in item:
                records[int(item["step"])] = float(item["eval_loss"])
    root_state = output_dir / "trainer_state.json"
    root_global_step = -1
    if root_state.exists():
        state = json.loads(root_state.read_text(encoding="utf-8"))
        root_global_step = int(state.get("global_step", -1) or -1)
        for item in state.get("log_history", []):
            if "eval_loss" in item and "step" in item:
                records[int(item["step"])] = float(item["eval_loss"])
    if not records:
        raise ValueError(f"No eval_loss records were found under {output_dir}")

    best_step, best_metric = min(records.items(), key=lambda item: (item[1], item[0]))
    selected = output_dir / f"checkpoint-{best_step}"
    if not selected.is_dir() and root_global_step == best_step and (output_dir / "adapter_model.safetensors").exists():
        selected = output_dir
    if not selected.is_dir():
        raise FileNotFoundError(
            f"Best evaluated checkpoint-{best_step} was not preserved. "
            "save_steps must equal eval_steps and save_total_limit must remain unset."
        )
    if not (selected / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Best checkpoint has no adapter_model.safetensors: {selected}")
    return selected, best_metric, best_step


def _replace_stable_directory(source: Path, target: Path, output_dir: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    output_dir = output_dir.resolve()
    expected_target = output_dir.with_name(output_dir.name + "_best")
    source_is_output = source == output_dir
    source_is_checkpoint = source.parent == output_dir and _checkpoint_step(source) >= 0
    if (
        not source.is_dir()
        or not (source_is_output or source_is_checkpoint)
        or target != expected_target
        or target == source
    ):
        raise ValueError(f"Unsafe stable model target: {target}")
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for path in source.iterdir():
        if path.is_file() and path.name not in {"optimizer.pt", "scheduler.pt", "rng_state.pth"}:
            shutil.copy2(path, temporary / path.name)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


if __name__ == "__main__":
    main()
