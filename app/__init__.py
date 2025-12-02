from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from .config import AppConfig
from .routes import api_bp
from .services.lawyer_ranker import LawyerRanker


def create_app(config: AppConfig | None = None) -> Flask:
    """Application factory used by both the CLI and the WSGI server."""

    cfg = config or AppConfig.from_env()
    app = Flask(__name__)
    CORS(app, origins=cfg.cors_origins)

    app.config["APP_CONFIG"] = cfg
    app.extensions["lawyer_ranker"] = LawyerRanker(
        model_name=cfg.model_name,
        lawyer_data_source=cfg.lawyer_data_source,
        default_top_k=cfg.default_top_k,
    )

    app.register_blueprint(api_bp)

    return app


__all__ = ["create_app"]
