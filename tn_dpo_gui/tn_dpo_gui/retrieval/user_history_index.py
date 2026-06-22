from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tn_dpo_gui.data.schema import TrajectoryRecord
from tn_dpo_gui.utils.io import read_json, write_json


class UserHistoryIndex:
    def __init__(self, records_by_user: dict[str, list[TrajectoryRecord]] | None = None) -> None:
        self.records_by_user = records_by_user or {}

    @classmethod
    def build(cls, records: list[TrajectoryRecord]) -> "UserHistoryIndex":
        grouped: dict[str, list[TrajectoryRecord]] = defaultdict(list)
        for record in records:
            grouped[record.user_id].append(record)
        return cls(dict(grouped))

    def get(self, user_id: str) -> list[TrajectoryRecord]:
        return list(self.records_by_user.get(user_id, []))

    def users(self) -> list[str]:
        return sorted(self.records_by_user)

    def to_dict(self) -> dict[str, list[dict]]:
        return {user_id: [record.to_dict() for record in records] for user_id, records in self.records_by_user.items()}

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "UserHistoryIndex":
        payload = read_json(path)
        return cls({user_id: [TrajectoryRecord.from_dict(row) for row in rows] for user_id, rows in payload.items()})
