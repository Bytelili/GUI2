from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import sha256_file, verify_release, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely register a verified v4 release without touching v2/v3 entries.")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--allow-synthetic-smoke", action="store_true")
    args = parser.parse_args()
    verification = verify_release(args.release_dir)
    if verification["status"] != "passed":
        print(json.dumps(verification, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit("RELEASE REGISTRATION FAILED: SHA256 verification failed")
    manifest = json.loads((args.release_dir / "listwise_v4_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("release_status") != "formal_candidate_release" and not args.allow_synthetic_smoke:
        raise SystemExit("RELEASE REGISTRATION FAILED: synthetic smoke requires --allow-synthetic-smoke")
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in manifest["dataset_hashes"]:
        source = args.release_dir / name
        destination = args.dataset_dir / name
        if destination.exists() and sha256_file(destination) != sha256_file(source):
            raise SystemExit(f"RELEASE REGISTRATION FAILED: refusing to overwrite different v4 artifact: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
            copied.append(str(destination.resolve()))
    fragment = json.loads((args.release_dir / "dataset_info_v4.json").read_text(encoding="utf-8"))
    dataset_info_path = args.dataset_dir / "dataset_info.json"
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8")) if dataset_info_path.exists() else {}
    for name, entry in fragment.items():
        if name in dataset_info and dataset_info[name] != entry:
            raise SystemExit(f"RELEASE REGISTRATION FAILED: refusing to replace existing dataset entry: {name}")
        dataset_info[name] = entry
    write_json(dataset_info_path, dataset_info)
    result = {
        "status": "passed",
        "release_status": manifest.get("release_status"),
        "copied": copied,
        "dataset_info": str(dataset_info_path.resolve()),
        "v2_v3_entries_preserved": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
