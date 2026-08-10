"""App factory. A package __init__ that imports its own submodules."""
from flask import Flask

from .routes import health, invoices


def create_app():
    app = Flask(__name__)
    app.register_blueprint(invoices.bp)
    app.register_blueprint(health.bp)
    return app
