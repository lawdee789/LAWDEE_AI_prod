from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import httpx

from .models import LawyerProfile


class LawyerRepository(ABC):
    @abstractmethod
    async def load(self) -> list[LawyerProfile]:
        ...


class FileLawyerRepository(LawyerRepository):
    """Loads lawyers from a local JSON file."""

    def __init__(self, path: Path):
        self.path = path

    async def load(self) -> list[LawyerProfile]:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self.path.read_text, "utf-8")
        payload = json.loads(raw)
        return _parse_lawyers(payload)


class HttpLawyerRepository(LawyerRepository):
    """
    Fetches lawyer profiles from an HTTP endpoint exposed by the Express backend.
    Expected response shape: {"success": true, "data": [ ...lawyer objects... ]}
    """

    def __init__(self, base_url: str, timeout_seconds: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def load(self) -> list[LawyerProfile]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(self.base_url)
            response.raise_for_status()
            payload = response.json()
        data = payload["data"] if isinstance(payload, dict) else payload
        return _parse_lawyers(data)


def _parse_lawyers(items: Iterable[dict]) -> list[LawyerProfile]:
    lawyers: list[LawyerProfile] = []
    for item in items:
        if not item:
            continue
        lawyer_id = (
            item.get("lawyer_id")
            or item.get("id")
            or item.get("_id")
            or item.get("user_id")
        )
        if not lawyer_id:
            continue
        lawyers.append(
            LawyerProfile(
                lawyer_id=str(lawyer_id),
                name=item.get("name")
                or item.get("user", {}).get("name")
                or "Unnamed Lawyer",
                slogan=item.get("slogan"),
                summary=item.get("summary") or item.get("description") or "",
                description=item.get("description"),
                lawfirm_name=item.get("lawfirm_name"),
                consult_min_price=item.get("consult_min_price")
                or item.get("consultationRate", {})
                .get("min")
                if isinstance(item.get("consultationRate"), dict)
                else None,
                consult_max_price=item.get("consult_max_price")
                or item.get("consultationRate", {})
                .get("max")
                if isinstance(item.get("consultationRate"), dict)
                else None,
                document_delivery_min_price=item.get("document_delivery_min_price")
                or item.get("documentDeliveryRate", {})
                .get("min")
                if isinstance(item.get("documentDeliveryRate"), dict)
                else None,
                document_delivery_max_price=item.get("document_delivery_max_price")
                or item.get("documentDeliveryRate", {})
                .get("max")
                if isinstance(item.get("documentDeliveryRate"), dict)
                else None,
                civil_case_specialization=item.get("civil_case_specialization")
                or item.get("civilCase_specialized")
                or [],
                criminal_case_specialization=item.get("criminal_case_specialization")
                or item.get("criminalCase_specialized")
                or [],
                services=item.get("services") or item.get("service") or [],
                has_lawyer_license=item.get("has_lawyer_license")
                or item.get("has_law_license"),
                is_verified_by_court=item.get("is_verified_by_court")
                or item.get("is_verified_by_council"),
                languages=item.get("languages") or [],
                education=item.get("education") or item.get("educations") or [],
                notable_cases=item.get("notable_cases") or [],
                avg_rating=item.get("avgRating") or item.get("avg_rating"),
                review_count=item.get("reviewCount") or item.get("review_count"),
                raw_payload=item,
            )
        )
    return lawyers
