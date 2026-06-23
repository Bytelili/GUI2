from __future__ import annotations

from tn_dpo_gui.scripts import resolve_path


def test_resolve_path_keeps_posix_absolute_paths_absolute() -> None:
    assert resolve_path("/home/dumike/zyy/GUI2").as_posix() == "/home/dumike/zyy/GUI2"
