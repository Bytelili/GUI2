from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import verify_release  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def oracle_only_rows(groups: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        candidates = group.get("candidates")
        oracle_index = group.get("oracle_index")
        if not isinstance(candidates, list) or not isinstance(oracle_index, int) or not 0 <= oracle_index < len(candidates):
            raise ValueError(f"Invalid oracle group: {group.get('group_id')}")
        oracle = candidates[oracle_index]
        if oracle.get("source") != "oracle_target":
            raise ValueError(f"Oracle source mismatch: {group.get('group_id')}")
        messages = copy.deepcopy(group.get("messages"))
        if not isinstance(messages, list) or any(item.get("role") == "assistant" for item in messages):
            raise ValueError(f"Control source messages must be prompt-only: {group.get('group_id')}")
        messages.append({"role": "assistant", "content": str(oracle.get("text") or "")})
        metadata = copy.deepcopy(group.get("metadata") or {})
        metadata.update(
            {
                "partition": split,
                "control_kind": "oracle_only_continuation",
                "source_group_id": group.get("group_id"),
                "source_task_id": group.get("task_id"),
                "release_claim": "retrieval-only smoke experiment; not full-v4",
            }
        )
        rows.append({"messages": messages, "images": list(group.get("images") or []), "metadata": metadata})
    return rows


def verify_control(output: Path, source_manifest_path: Path) -> dict[str, Any]:
    manifest_path = output / "oracle_control_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing oracle control manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8", errors="strict"))
    if manifest.get("formal_full_v4_complete") is not False:
        raise ValueError("Oracle control incorrectly claims full-v4 completion")
    expected_source = sha256_file(source_manifest_path)
    if manifest.get("source_release_manifest_sha256") != expected_source:
        raise ValueError("Oracle control source release manifest changed")
    for filename, expected in (manifest.get("dataset_hashes") or {}).items():
        path = output / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Oracle control SHA256 mismatch: {filename}")
    return {
        "status": "passed",
        "output_dir": str(output),
        "group_counts": manifest.get("group_counts"),
        "source_release_manifest_sha256": expected_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an oracle-only continuation control from the same v4 smoke groups.")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    release = args.release_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"CONTROL BUILD REFUSED: output directory is not empty: {output}")
    verification = verify_release(release)
    if verification["status"] != "passed":
        raise SystemExit(f"CONTROL BUILD REFUSED: source release verification failed: {verification}")
    source_manifest_path = release / "listwise_v4_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("release_kind") != "smoke_v4" or source_manifest.get("formal_full_v4_complete") is not False:
        raise SystemExit("CONTROL BUILD REFUSED: expected the unchanged non-formal smoke release")

    if args.verify_only:
        print(json.dumps(verify_control(output, source_manifest_path), ensure_ascii=False, indent=2))
        print("PAPO ORACLE-ONLY SMOKE CONTROL VERIFICATION PASSED")
        return

    output.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in ("train", "eval"):
        source = release / f"papo_proactive_{split}_listwise_v4.json"
        groups = json.loads(source.read_text(encoding="utf-8", errors="strict"))
        rows = oracle_only_rows(groups, split)
        filename = f"papo_proactive_{split}_sft_v4_smoke_control.json"
        destination = output / filename
        write_json(destination, rows)
        files[filename] = sha256_file(destination)
        counts[split] = len(rows)

    dataset_info = {
        "papo_proactive_train_sft_v4_smoke_control": {
            "file_name": "papo_proactive_train_sft_v4_smoke_control.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
        "papo_proactive_eval_sft_v4_smoke_control": {
            "file_name": "papo_proactive_eval_sft_v4_smoke_control.json",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        },
    }
    write_json(output / "dataset_info.json", dataset_info)
    files["dataset_info.json"] = sha256_file(output / "dataset_info.json")
    manifest = {
        "schema_version": "papo_listwise_v4_oracle_control_manifest",
        "experiment_kind": "retrieval-only smoke experiment",
        "formal_full_v4_complete": False,
        "source_release": str(release),
        "source_release_manifest_sha256": sha256_file(source_manifest_path),
        "group_counts": counts,
        "dataset_hashes": files,
        "claim_boundary": "engineering control only; not full-v4 and not a formal effect claim",
    }
    write_json(output / "oracle_control_manifest.json", manifest)
    with (output / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for filename, digest in files.items():
            handle.write(f"{digest}  {filename}\n")
        handle.write(f"{sha256_file(output / 'oracle_control_manifest.json')}  oracle_control_manifest.json\n")
    print(json.dumps(verify_control(output, source_manifest_path), ensure_ascii=False, indent=2))
    print("PAPO ORACLE-ONLY SMOKE CONTROL BUILD PASSED")


if __name__ == "__main__":
    main()
