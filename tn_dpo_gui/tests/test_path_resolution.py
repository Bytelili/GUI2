from __future__ import annotations

from tn_dpo_gui.scripts import override_main_project_root_config, resolve_config_paths, resolve_path


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
