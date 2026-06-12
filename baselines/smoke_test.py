from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.common import action_type, official_prompt, parse_action, sequence_similarity  # noqa: E402
from baselines.metrics import prediction_report  # noqa: E402


def main() -> None:
    click = "click(coordinates=(12, 34), content='搜索')"
    typed = "type(text='咖啡')"
    assert parse_action(click)["coordinates"] == [12, 34]
    assert parse_action(typed)["text"] == "咖啡"
    assert action_type("finished()") == "finished"
    assert not parse_action("I would click the button")["valid"]
    assert sequence_similarity([click], [click]) == 1.0

    prompt = official_prompt("搜索咖啡", "用户画像", "1080x2400", "[]", [click], [])
    assert "Actions_reference" in prompt
    assert "Previous_actions" in prompt

    report = prediction_report(
        [
            {
                "variant": "official_icl",
                "episode_id": "episode",
                "prediction": click,
                "target_action": click,
                "cross_user_actions": ["finished()"],
            }
        ]
    )
    assert report["parse_valid_rate"] == 1.0
    assert report["action_type_accuracy"] == 1.0
    assert report["exact_action_accuracy"] == 1.0
    print("baseline smoke test passed")


if __name__ == "__main__":
    main()
