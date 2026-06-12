from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.dpo import build_pairs  # noqa: E402
from papo.io import read_jsonl, write_json, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--advantages", default=str(PROJECT_ROOT / "data/papo_raw/papo_advantages.jsonl"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data/papo_raw/papo_dpo_pairs.jsonl"))
    parser.add_argument("--summary_out", default=str(PROJECT_ROOT / "data/papo_raw/papo_dpo_summary.json"))
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--tau_m", type=float, default=0.2)
    parser.add_argument("--w_max", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    rows = read_jsonl(args.advantages)
    pairs = build_pairs(rows, margin=args.margin, tau_m=args.tau_m, w_max=args.w_max, beta=args.beta)
    write_jsonl(args.out, pairs)
    summary = {
        "num_advantage_rows": len(rows),
        "num_pairs": len(pairs),
        "margin": args.margin,
        "tau_m": args.tau_m,
        "w_max": args.w_max,
        "beta": args.beta,
    }
    write_json(args.summary_out, summary)
    print(f"pairs: {len(pairs)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
