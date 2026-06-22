from __future__ import annotations

import argparse

from tn_dpo_gui.data.dataset import load_trajectory_records
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.utils.io import ensure_dir
from tn_dpo_gui.utils.main_project import derive_tn_dpo_layout

from . import resolve_path


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
    print({"users": len(index.users()), "output": str(output_path)})


if __name__ == "__main__":
    main()
