import asyncio

import numpy as np

from app.config import Settings
from app.matching import MatchingEngine
from app.models import CasePayload, LawyerProfile
from app.repository import LawyerRepository


class InMemoryRepository(LawyerRepository):
    def __init__(self, lawyers):
        self._lawyers = lawyers

    async def load(self):
        return self._lawyers


class FakeEncoder:
    """Simple keyword-based encoder used to keep the unit test deterministic."""

    KEYWORDS = ["criminal", "civil", "family", "assault", "contract"]

    def encode(self, texts):
        vectors = []
        for text in texts:
            tokens = text.lower()
            vector = [tokens.count(keyword) for keyword in self.KEYWORDS]
            vectors.append(np.asarray(vector, dtype="float32"))
        return np.vstack(vectors)


def test_matches_prioritize_relevant_specialists():
    lawyers = [
        LawyerProfile(
            lawyer_id="criminal-expert",
            name="Criminal Expert",
            summary="criminal defense assault battery criminal",
            criminal_case_specialization=["Assault"],
        ),
        LawyerProfile(
            lawyer_id="civil-expert",
            name="Civil Expert",
            summary="civil contract contract law",
            civil_case_specialization=["Contract"],
        ),
    ]
    repo = InMemoryRepository(lawyers)
    encoder = FakeEncoder()
    settings = Settings(default_top_k=1, min_score_threshold=0.0)
    engine = MatchingEngine(settings=settings, encoder=encoder, repository=repo)

    asyncio.run(engine.refresh())

    case = CasePayload(
        case_id="case-1",
        title="โดนข่มขู่และทำร้ายร่างกาย",
        description="criminal assault battery at convenience store",
        category="CRIMINAL",
    )

    response = engine.match(case, top_k=1)

    assert response.lawyers[0].lawyer.lawyer_id == "criminal-expert"
