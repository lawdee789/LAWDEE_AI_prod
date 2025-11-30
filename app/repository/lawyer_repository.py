from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class LawyerRepository:
    """Load and cache lawyer data from disk."""

    def __init__(self, data_path: str) -> None:
        self.data_path = Path(data_path)
        self._lawyers: List[Dict[str, Any]] = []
        self.reload()

    def _load_from_disk(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Lawyer dataset not found at {self.data_path.resolve()}."
                " Create the file or update LAWYERS_DATA_PATH."
            )

        with self.data_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("Lawyer dataset must be a list of lawyer records")

        return data

    def reload(self) -> None:
        self._lawyers = self._load_from_disk()

    def get_all(self) -> List[Dict[str, Any]]:
        return self._lawyers
