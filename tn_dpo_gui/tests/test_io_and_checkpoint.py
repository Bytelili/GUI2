from __future__ import annotations

from pathlib import Path

import torch

from tn_dpo_gui.training.trainer_utils import load_checkpoint, save_checkpoint
from tn_dpo_gui.utils.io import read_json, read_jsonl, read_yaml, write_json, write_jsonl, write_yaml


def test_write_json_overwrites_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    write_json(path, {"step": 1, "text": "第一次"})
    write_json(path, {"step": 2, "text": "第二次"})
    assert read_json(path) == {"step": 2, "text": "第二次"}
    assert not list(tmp_path.glob(".data.json.*.tmp"))


def test_write_jsonl_roundtrip_unicode(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    rows = [{"id": 1, "text": "Taylor Swift"}, {"id": 2, "text": "周杰伦"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert not list(tmp_path.glob(".pairs.jsonl.*.tmp"))


def test_write_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    payload = {"main_project": {"enabled": True, "root_config": "../../config.yaml"}}
    write_yaml(path, payload)
    assert read_yaml(path) == payload
    assert not list(tmp_path.glob(".config.yaml.*.tmp"))


def test_save_checkpoint_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ranker.pt"
    payload = {"tensor": torch.tensor([1.0, 2.0, 3.0]), "meta": {"epoch": 3}}
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert torch.equal(loaded["tensor"], payload["tensor"])
    assert loaded["meta"] == payload["meta"]
    assert not list(tmp_path.glob(".ranker.pt.*.tmp"))
