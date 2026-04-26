from __future__ import annotations

from .web_app import app


def serve(host: str, port: int) -> None:
    app.run(host=host, port=port, debug=False)
