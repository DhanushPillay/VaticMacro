"""App package — lazy re-export to avoid circular imports."""

try:
    from app.factory import create_app

    __all__ = ["create_app"]
except Exception:  # pragma: no cover
    __all__ = []
