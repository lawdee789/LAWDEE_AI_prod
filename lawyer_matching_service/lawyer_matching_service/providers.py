"""Data providers for lawyer records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import httpx

from .config import Settings, get_settings
from .logger import get_logger
from .schemas import LawyerProfile

logger = get_logger(__name__)


class LawyerProvider:
    """Loads lawyer records from the Express API or a fallback file."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._fallback_path = Path(self._settings.fallback_dataset_path)

    async def fetch(self) -> List[LawyerProfile]:
        """Return lawyer profiles from the configured data source."""

        records: List[LawyerProfile] | None = None

        if self._settings.lawyer_source_url:
            try:
                records = await self._fetch_from_remote()
                logger.debug("Fetched %s lawyers from remote source", len(records))
            except Exception as exc:  # noqa: BLE001 (we want to log any failure)
                logger.warning("Falling back to local dataset: %s", exc)

        if records is None:
            records = self._load_from_file()
            logger.debug("Loaded %s lawyers from %s", len(records), self._fallback_path)

        return records

    async def _fetch_from_remote(self) -> List[LawyerProfile]:
        headers = {}
        if self._settings.lawyer_source_token:
            headers["Authorization"] = f"Bearer {self._settings.lawyer_source_token}"

        async with httpx.AsyncClient(timeout=self._settings.request_timeout_seconds) as client:
            response = await client.get(str(self._settings.lawyer_source_url), headers=headers)
            response.raise_for_status()
            payload = response.json()

        return self._parse_payload(payload)

    def _load_from_file(self) -> List[LawyerProfile]:
        if not self._fallback_path.exists():
            raise FileNotFoundError(
                f"Fallback dataset not found at {self._fallback_path}. Provide LAWYER_SOURCE_URL or a valid dataset."
            )

        with self._fallback_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return self._parse_payload(payload)

    def _parse_payload(self, payload: object) -> List[LawyerProfile]:
        data: Iterable[object]
        if isinstance(payload, dict) and "data" in payload:
            data = payload["data"]  # type: ignore[assignment]
        elif isinstance(payload, list):
            data = payload
        else:
            raise ValueError("Unexpected payload shape received from lawyer source")

        parsed: List[LawyerProfile] = []
        for raw in data:
            parsed.append(LawyerProfile.model_validate(raw))

        if not parsed:
            raise ValueError("No lawyer data available after parsing")

        return parsed
