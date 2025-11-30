from . import create_app
from .config import get_settings


def main() -> None:
    settings = get_settings()
    app = create_app(settings)
    app.run(host=settings.flask_host, port=settings.flask_port, debug=settings.flask_debug)


if __name__ == "__main__":
    main()
