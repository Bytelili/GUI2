from __future__ import annotations

import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from papo.papo_objective import _personalization_evidence, score_tree_leaves  # noqa: E402


def main() -> None:
    temperature = 0.2
    assert _personalization_evidence(math.log(2.0), temperature, "tanh_log_ratio") > 0.0
    assert _personalization_evidence(0.0, temperature, "tanh_log_ratio") == 0.0
    assert _personalization_evidence(math.log(0.5), temperature, "tanh_log_ratio") < 0.0
    assert 0.5 < _personalization_evidence(math.log(2.0), temperature, "sigmoid_log_ratio") < 1.0

    try:
        _personalization_evidence(0.0, temperature, "unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown evidence transforms must fail.")

    tree = {
        "target_actions": ["select:preferred", "finish"],
        "leaves": [
            {"leaf_id": "same", "actions": ["select:preferred", "finish"]},
            {"leaf_id": "cross", "actions": ["select:generic", "finish"]},
        ],
    }
    task = {
        "input": {
            "same_user_action_references": [{"actions": ["select:preferred", "finish"]}],
            "cross_user_action_references": [{"actions": ["select:generic", "finish"]}],
        }
    }
    scored = score_tree_leaves(
        tree,
        task,
        {
            "eta": 0.5,
            "evidence_transform": "tanh_log_ratio",
            "fingertip_temperature": temperature,
            "epsilon": 1e-6,
            "min_same_user_references": 1,
            "task_similarity_threshold": 0.0,
            "require_finish": False,
        },
    )
    assert scored["leaves"][0]["r_pref"] > 0.0
    assert scored["leaves"][1]["r_pref"] < 0.0
    assert scored["metadata"]["personalization_evidence_transform"] == "tanh_log_ratio"

    print("PAPO objective smoke test passed")


if __name__ == "__main__":
    main()
