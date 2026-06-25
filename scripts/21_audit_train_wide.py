from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.proactive_fixed_export import audit_wide_rows, read_wide_csv, write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train_wide.csv for proactive-fixed exports.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_wide_csv(args.input)
    report = audit_wide_rows(rows)
    write_report(Path(args.out), report)
    print(f"rows: {report['row_count']}")
    print(f"users: {report['user_count']}")
    print(f"warnings: {len(report['warnings'])}")
    print(f"written: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
