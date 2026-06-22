from __future__ import annotations

import argparse

from tn_dpo_gui.data.dataset import load_trajectory_records
from tn_dpo_gui.retrieval.user_history_index import UserHistoryIndex
from tn_dpo_gui.utils.io import ensure_dir

from . import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a user history index from trajectory records.")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    trajectories = load_trajectory_records(resolve_path(args.trajectories))
    index = UserHistoryIndex.build(trajectories)
    output_path = resolve_path(args.output)
    ensure_dir(output_path.parent)
    index.save(output_path)
    print({"users": len(index.users()), "output": str(output_path)})


if __name__ == "__main__":
    main()
