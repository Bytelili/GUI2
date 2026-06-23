from __future__ import annotations

from pathlib import Path

from tn_dpo_gui.scripts import apply_main_project_layout, override_main_project_root_config, resolve_config_paths, resolve_path
from tn_dpo_gui.utils.config import load_config


def test_resolve_path_keeps_posix_absolute_paths_absolute() -> None:
    assert resolve_path("/home/dumike/zyy/GUI2").as_posix() == "/home/dumike/zyy/GUI2"


def test_resolve_config_paths_preserves_posix_absolute_strings() -> None:
    config = {"data": {"pairs_path": "/home/dumike/zyy/GUI2/data/pairs/pairs.jsonl"}}
    resolved = resolve_config_paths(config, {"data": ["pairs_path"]})
    assert resolved["data"]["pairs_path"] == "/home/dumike/zyy/GUI2/data/pairs/pairs.jsonl"


def test_override_main_project_root_config_updates_config() -> None:
    config = {"main_project": {"enabled": True, "root_config": "../config.yaml"}}
    override_main_project_root_config(config, "/tmp/server-config.yaml")
    assert config["main_project"]["root_config"] == "/tmp/server-config.yaml"


def test_resolve_config_paths_uses_config_file_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    config_path = config_dir / "train.yaml"
    config_path.write_text("data:\n  pairs_path: ../data/pairs.jsonl\n", encoding="utf-8")

    config = load_config(config_path)
    resolved = resolve_config_paths(config, {"data": ["pairs_path"]})
    assert resolved["data"]["pairs_path"] == (tmp_path / "data" / "pairs.jsonl").resolve().as_posix()


def test_apply_main_project_layout_resolves_root_config_relative_to_config_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / "nested" / "configs"
    config_dir.mkdir(parents=True)
    root_config = workspace / "config.yaml"
    root_config.write_text(
        "\n".join(
            [
                "paths:",
                "  work_dir: ./work",
                "  task_dir: ./tasks",
                "  checkpoint_root: ./checkpoints",
                "  logging_root: ./runs",
                "  raw_root: ./raw",
                "  official_root: ./official",
                "  protocol_dir: ./protocol",
                "  llamafactory_dir: ./LLaMA-Factory",
                "  llamafactory_data_dir: ./LLaMA-Factory/data",
                "training:",
                "  model_name_or_path: ./models/qwen",
            ]
        ),
        encoding="utf-8",
    )
    sub_config = config_dir / "build_pairs.yaml"
    sub_config.write_text("main_project:\n  enabled: true\n  root_config: ../../config.yaml\ninput: {}\noutput: {}\n", encoding="utf-8")

    config = load_config(sub_config)
    resolved = apply_main_project_layout(config, {"input": {"steps_path": "steps_path"}, "output": {"pairs_path": "pairs_path"}})
    assert root_config.resolve().as_posix() == resolved["_main_project_layout"]["root_config_path"]
