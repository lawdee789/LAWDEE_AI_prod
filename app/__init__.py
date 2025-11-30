from flask import Flask
from flask_cors import CORS

from .config import Settings, get_settings
from .repository.lawyer_repository import LawyerRepository
from .routes import register_routes
from .services.embedding_service import EmbeddingService
from .services.lawyer_ranker import LawyerRanker


def create_app(settings: Settings | None = None) -> Flask:
    """Flask application factory used by both tests and runtime."""

    settings = settings or get_settings()

    app = Flask(__name__)
    CORS(app)

    repository = LawyerRepository(settings.data_path)
    embedder = EmbeddingService(settings.model_name)
    ranker = LawyerRanker(
        repository=repository,
        embedder=embedder,
        min_score=settings.min_similarity_score,
    )

    register_routes(app, ranker, settings)

    return app
