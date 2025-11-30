"""Lightweight persistence layer for lawyer embeddings."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional

from . import schemas


class LawyerStore:
    """Persist lawyer profiles plus their embeddings in JSON."""

    def __init__(self, data_dir: Path) -> None:
        self._lock = RLock()
        self._file_path = data_dir / "lawyers.json"
        self._records: Dict[str, schemas.LawyerRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return

        with self._file_path.open("r", encoding="utf-8") as fp:
            raw_records = json.load(fp)

        for payload in raw_records:
            record = schemas.LawyerRecord(**payload)
            self._records[record.lawyer_id] = record

    def _persist(self) -> None:
        with self._file_path.open("w", encoding="utf-8") as fp:
            json.dump(
                [record.model_dump() for record in self._records.values()],
                fp,
                ensure_ascii=False,
                indent=2,
            )

    def upsert(
        self, profile: schemas.LawyerProfile, embedding: List[float]
    ) -> schemas.LawyerRecord:
        """Create or update a lawyer record."""

        with self._lock:
            now = datetime.now(timezone.utc)
            existing = self._records.get(profile.lawyer_id)
            created_at = existing.created_at if existing else now

            record = schemas.LawyerRecord(
                **profile.model_dump(),
                embedding=list(map(float, embedding)),
                created_at=created_at,
                updated_at=now,
            )
            self._records[record.lawyer_id] = record
            self._persist()
            return record

    def list(self) -> List[schemas.LawyerRecord]:
        """Return a snapshot of all records."""

        with self._lock:
            return list(self._records.values())

    def get(self, lawyer_id: str) -> Optional[schemas.LawyerRecord]:
        with self._lock:
            return self._records.get(lawyer_id)

    def count(self) -> int:
        with self._lock:
            return len(self._records)

    def iter_embeddings(self) -> Iterable[schemas.LawyerRecord]:
        """Yield records, useful for read-only streaming."""

        with self._lock:
            for record in self._records.values():
                yield record

