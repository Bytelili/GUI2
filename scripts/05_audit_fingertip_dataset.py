from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.dataset_audit import audit_dataset  # noqa: E402
from papo.io import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extracted FingerTip data before PAPO processing.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--official_root", default=str(PROJECT_ROOT / "data/official/fingertip20k"))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    audit = audit_dataset(args.raw_root, args.official_root)
    if args.out:
        write_json(args.out, audit)
    print(f"complete episodes: {audit['complete_episodes']}")
    print(f"complete users: {audit['num_complete_users']}")
    print(f"ready for full build: {audit['ready_for_full_build']}")
    for name, coverage in audit["coverage"].items():
        print(f"{name}: {coverage['covered']}/{coverage['total']} ({coverage['rate']:.1%})")
    for warning in audit["warnings"]:
        print(f"warning: {warning}")


if __name__ == "__main__":
    main()
