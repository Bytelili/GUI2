from __future__ import annotations

import argparse
from pathlib import Path

from epipeline.io_utils import read_json, read_jsonl, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a manual success-review sheet for unverified execution episodes.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    rows = []
    for entry in manifest["runs"]:
        for result in read_jsonl(Path(str(entry["run_dir"])) / "raw_results.jsonl"):
            if result.get("success_verified"):
                continue
            final = result.get("final_observation") if isinstance(result.get("final_observation"), dict) else {}
            rows.append(
                {
                    "run_id": entry["id"],
                    "task_id": result.get("task_id", ""),
                    "user_id": result.get("user_id", ""),
                    "app": result.get("app", ""),
                    "termination_reason": result.get("termination_reason", ""),
                    "final_screenshot": final.get("screenshot", ""),
                    "final_xml": final.get("xml_path", ""),
                    "success": "",
                    "annotator": "",
                    "evidence": "",
                }
            )
    write_csv(args.output, rows)
    print(f"Unverified rows requiring review: {len(rows)}")
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
