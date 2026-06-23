from __future__ import annotations

import argparse
from pathlib import Path

from tn_dpo_gui.data.dataset import load_trajectory_records
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.utils.io import ensure_dir, write_json
from tn_dpo_gui.utils.main_project import derive_tn_dpo_layout
from tn_dpo_gui.utils.provenance import runtime_provenance

from . import PROJECT_ROOT, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a user history index from trajectory records.")
    parser.add_argument("--trajectories", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--root-config", default="")
    args = parser.parse_args()

    if args.trajectories and args.output:
        trajectories_path = resolve_path(args.trajectories)
        output_path = resolve_path(args.output)
    else:
        layout = derive_tn_dpo_layout(resolve_path(args.root_config) if args.root_config else None)
        trajectories_path = layout["trajectories_path"]
        output_path = layout["user_index_path"]

    trajectories = load_trajectory_records(trajectories_path)
    index = UserHistoryIndex.build(trajectories)
    ensure_dir(output_path.parent)
    index.save(output_path)
    summary = {
        "users": len(index.users()),
        "trajectories_path": str(trajectories_path),
        "output": str(output_path),
        "provenance": runtime_provenance(PROJECT_ROOT),
    }
    if args.root_config:
        summary["root_config_path"] = str(resolve_path(args.root_config))
    write_json(Path(output_path).with_name(Path(output_path).stem + "_summary.json"), summary)
    print(summary)


if __name__ == "__main__":
    main()
