from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The local data-only test runtime may intentionally omit torch.
    torch = None


ROOT = Path(__file__).resolve().parents[1]
def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


collator_module = _load(
    "papo_group_collator_standalone",
    ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "data" / "papo_group.py",
)
flatten_papo_group_features = collator_module.flatten_papo_group_features
IGNORE_INDEX = -100
if torch is not None:
    loss_module = _load(
        "papo_group_listwise_standalone",
        ROOT / "LLaMA-Factory" / "src" / "llamafactory" / "train" / "sft" / "papo_group_listwise.py",
    )
    papo_group_listwise_loss = loss_module.papo_group_listwise_loss
    papo_listwise_loss = loss_module.papo_listwise_loss


def _batch(candidate_logits: list[float], lengths: list[int] | None = None):
    lengths = lengths or [2] * len(candidate_logits)
    sequence_length, vocabulary = 4, 3
    logits = torch.zeros((len(candidate_logits), sequence_length, vocabulary), dtype=torch.float32)
    labels = torch.full((len(candidate_logits), sequence_length), IGNORE_INDEX, dtype=torch.long)
    for index, (value, length) in enumerate(zip(candidate_logits, lengths)):
        labels[index, 1 : 1 + length] = 1
        logits[index, :length, 1] = value
    return logits, labels


@unittest.skipIf(torch is None, "torch is not installed in this local data-only runtime")
class GroupListwiseLossTest(unittest.TestCase):
    def _loss(self, values: list[float], q=(0.9, 0.1)) -> float:
        logits, labels = _batch(values)
        return float(
            papo_group_listwise_loss(
                logits,
                labels,
                torch.tensor([0, 0]),
                torch.tensor(q),
                torch.tensor([True, False]),
            )
        )

    def test_oracle_log_probability_improvement_lowers_loss(self) -> None:
        self.assertLess(self._loss([3.0, 0.0]), self._loss([0.0, 0.0]))

    def test_wrong_candidate_improvement_raises_loss(self) -> None:
        self.assertGreater(self._loss([0.0, 3.0]), self._loss([0.0, 0.0]))

    def test_groups_do_not_share_softmax(self) -> None:
        first_logits, first_labels = _batch([2.0, 0.0])
        second_logits, second_labels = _batch([-1.0, 1.0, 0.0])
        first = papo_group_listwise_loss(
            first_logits, first_labels, torch.tensor([0, 0]), torch.tensor([0.8, 0.2]), torch.tensor([True, False])
        )
        second = papo_group_listwise_loss(
            second_logits,
            second_labels,
            torch.tensor([0, 0, 0]),
            torch.tensor([0.6, 0.3, 0.1]),
            torch.tensor([True, False, False]),
        )
        combined = papo_group_listwise_loss(
            torch.cat([first_logits, second_logits]),
            torch.cat([first_labels, second_labels]),
            torch.tensor([0, 0, 1, 1, 1]),
            torch.tensor([0.8, 0.2, 0.6, 0.3, 0.1]),
            torch.tensor([True, False, True, False, False]),
        )
        self.assertTrue(torch.allclose(combined, (first + second) / 2, atol=1e-6))

    def test_length_normalization_removes_token_count_advantage(self) -> None:
        logits, labels = _batch([1.5, 1.5], lengths=[1, 3])
        loss = papo_group_listwise_loss(
            logits, labels, torch.tensor([0, 0]), torch.tensor([0.5, 0.5]), torch.tensor([True, False])
        )
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))), places=6)

    def test_variable_group_size_padding_and_metrics(self) -> None:
        logits, labels = _batch([2.0, 0.0, 1.0, -1.0, 0.5], lengths=[1, 3, 2, 1, 3])
        loss, metrics = papo_group_listwise_loss(
            logits,
            labels,
            torch.tensor([0, 0, 1, 1, 1]),
            torch.tensor([0.8, 0.2, 0.7, 0.2, 0.1]),
            torch.tensor([True, False, True, False, False]),
            return_metrics=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(metrics), {"group_loss", "oracle_top1_accuracy", "oracle_margin", "target_entropy", "policy_entropy"}
        )

    def test_illegal_target_distribution_and_oracle_are_rejected(self) -> None:
        logits, labels = _batch([1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            papo_group_listwise_loss(
                logits, labels, torch.tensor([0, 0]), torch.tensor([0.8, 0.3]), torch.tensor([True, False])
            )
        with self.assertRaisesRegex(ValueError, "highest"):
            papo_group_listwise_loss(
                logits, labels, torch.tensor([0, 0]), torch.tensor([0.4, 0.6]), torch.tensor([True, False])
            )

    def test_collator_flatten_keeps_complete_variable_groups(self) -> None:
        features = [
            {
                "input_ids": [[1, 2], [1, 3]],
                "labels": [[-100, 2], [-100, 3]],
                "attention_mask": [[1, 1], [1, 1]],
                "images": ["a.jpg"],
                "videos": None,
                "audios": None,
                "papo_group_target": [0.8, 0.2],
                "papo_group_oracle_index": 0,
                "papo_group_id": "g0",
            },
            {
                "input_ids": [[1], [2], [3]],
                "labels": [[1], [2], [3]],
                "attention_mask": [[1], [1], [1]],
                "images": None,
                "videos": None,
                "audios": None,
                "papo_group_target": [0.7, 0.2, 0.1],
                "papo_group_oracle_index": 0,
                "papo_group_id": "g1",
            },
        ]
        flat, groups, probabilities, oracle = flatten_papo_group_features(features)
        self.assertEqual(len(flat), 5)
        self.assertEqual(groups, [0, 0, 1, 1, 1])
        self.assertEqual(probabilities, [0.8, 0.2, 0.7, 0.2, 0.1])
        self.assertEqual(oracle, [True, False, True, False, False])

    def test_legacy_weighted_nll_remains_available(self) -> None:
        logits, labels = _batch([1.0, 0.0])
        loss = papo_listwise_loss(logits, labels, torch.tensor([0.8, 0.2]))
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
