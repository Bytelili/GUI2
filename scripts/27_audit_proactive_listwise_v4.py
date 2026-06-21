from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_listwise_v4 import verify_release  # noqa: E402
from papo.proactive_quality_gate_v4 import audit_v4_groups, write_quality_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-audit a PAPO Listwise-v4 release and verify all SHA256 bindings.")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, action="append", default=[])
    parser.add_argument("--allow-unavailable-images", action="store_true")
    args = parser.parse_args()
    train = json.loads((args.release_dir / "papo_proactive_train_listwise_v4.json").read_text(encoding="utf-8"))
    evaluation = json.loads((args.release_dir / "papo_proactive_eval_listwise_v4.json").read_text(encoding="utf-8"))
    quality, issues = audit_v4_groups(
        train,
        evaluation,
        image_roots=args.image_root,
        allow_unavailable_images=args.allow_unavailable_images,
    )
    paths = write_quality_outputs(quality, issues, args.report_dir)
    hashes = verify_release(args.release_dir)
    result = {"quality": quality, "sha256_and_manifest": hashes, "reports": paths}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if quality["status"] == "failed" or hashes["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
