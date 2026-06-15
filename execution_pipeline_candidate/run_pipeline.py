from __future__ import annotations

import argparse
import json
from pathlib import Path

from epipeline.analysis import analyze_manifest
from epipeline.audit import audit_manifest
from epipeline.io_utils import read_json
from epipeline.preflight import preflight_manifest
from epipeline.prepare import prepare_experiment
from epipeline.runner import run_entry
from epipeline.scoring import score_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated personalized-execution experiment pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stages", default="prepare,preflight,run,score,analyze,audit")
    parser.add_argument("--annotations", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-device-connection", action="store_true")
    parser.add_argument("--require-paper-eligible", action="store_true")
    args = parser.parse_args()
    stages = {value.strip() for value in args.stages.split(",") if value.strip()}
    allowed = {"prepare", "preflight", "run", "score", "analyze", "audit"}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    config = read_json(args.config)
    manifest_path = Path(str(config["output_root"])).resolve() / "experiment_manifest.json"
    manifest = prepare_experiment(args.config) if "prepare" in stages else read_json(manifest_path)
    output: dict[str, object] = {"manifest": str(manifest_path)}
    if "preflight" in stages:
        preflight = preflight_manifest(manifest, check_device_connection=not args.skip_device_connection)
        output["preflight"] = preflight
        if preflight["status"] != "passed":
            print(json.dumps(output, ensure_ascii=False, indent=2))
            raise RuntimeError("Execution experiment preflight failed")
    if "run" in stages:
        output["runs"] = [run_entry(manifest, entry, limit=args.limit) for entry in manifest["runs"]]
        if any(report["status"] != "completed" for report in output["runs"]):
            print(json.dumps(output, ensure_ascii=False, indent=2))
            raise RuntimeError("At least one execution run has retryable task failures")
    if "score" in stages:
        output["score"] = score_manifest(manifest, args.annotations or None)
    if "analyze" in stages:
        output["analysis"] = analyze_manifest(manifest)
    if "audit" in stages:
        audit = audit_manifest(manifest, require_paper_eligible=args.require_paper_eligible)
        output["audit"] = audit
        if audit["status"] != "passed":
            print(json.dumps(output, ensure_ascii=False, indent=2))
            raise RuntimeError("EXECUTION PIPELINE AUDIT FAILED")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("EXECUTION PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
