from __future__ import annotations

from dotenv import load_dotenv

from app import create_app
from app.config import AppConfig

load_dotenv()
config = AppConfig.from_env()
app = create_app(config)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.port, debug=config.debug)
