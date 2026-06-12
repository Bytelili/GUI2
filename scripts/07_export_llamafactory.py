from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.llamafactory_export import (  # noqa: E402
    attach_prior_actions,
    dataset_info,
    export_execution_dpo,
    export_execution_sft,
    export_proactive_sft,
    load_rows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PAPO tasks to LLaMA-Factory datasets.")
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--suggestion_tasks", required=True)
    parser.add_argument("--execution_tasks", required=True)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dpo_pairs", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--asset_prefix", default="RawDataset")
    args = parser.parse_args()

    suggestion_tasks = load_rows(args.suggestion_tasks)
    execution_tasks = load_rows(args.execution_tasks)
    steps = load_rows(args.steps)
    attach_prior_actions(execution_tasks, steps)
    out_dir = Path(args.out_dir)

    proactive = export_proactive_sft(suggestion_tasks, args.raw_root, args.asset_prefix)
    execution = export_execution_sft(execution_tasks, steps, args.raw_root, args.asset_prefix)
    dpo = (
        export_execution_dpo(execution_tasks, steps, load_rows(args.dpo_pairs), args.raw_root, args.asset_prefix)
        if args.dpo_pairs
        else []
    )
    write_json(out_dir / "papo_proactive_sft.json", proactive)
    write_json(out_dir / "papo_execution_sft.json", execution)
    write_json(out_dir / "papo_execution_dpo.json", dpo)
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info(), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"proactive SFT: {len(proactive)}")
    print(f"execution SFT: {len(execution)}")
    print(f"execution DPO: {len(dpo)}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
