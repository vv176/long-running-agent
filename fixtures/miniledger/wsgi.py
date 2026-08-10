"""Gunicorn entry point. Module-level `application` is the signature."""
from app import create_app

application = create_app()
