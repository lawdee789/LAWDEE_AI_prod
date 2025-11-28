from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .schemas import LawyerPayload


class FileLawyerRepository:
    """JSON-file backed storage for lawyer profiles."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_raw(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        text = self.path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        data = json.loads(text)
        if isinstance(data, list):
            return {item["lawyer_id"]: item for item in data}
        return data

    def _save_raw(self, items: Dict[str, dict]) -> None:
        self.path.write_text(
            json.dumps(list(items.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all(self) -> List[LawyerPayload]:
        data = self._load_raw()
        return [LawyerPayload(**item) for item in data.values()]

    def upsert(self, lawyers: Iterable[LawyerPayload]) -> List[LawyerPayload]:
        items = self._load_raw()
        for lawyer in lawyers:
            items[lawyer.lawyer_id] = lawyer.model_dump()
        self._save_raw(items)
        return [LawyerPayload(**item) for item in items.values()]
