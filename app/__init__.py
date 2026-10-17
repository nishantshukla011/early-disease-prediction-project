"""
Flask application factory.

Original code: pmk456 / AI-Powered-Early-Disease-Prediction-System (GPL-3.0).
Modified for academic submission by nishantshukla011 on 17 October 2026:
secret key and upload limit are configured here, and friendly error pages are
registered instead of raw tracebacks.
"""

import os

from flask import Flask, render_template

from app.config import DEMO_CREDENTIALS, MAX_UPLOAD_BYTES
from app.routes import routes


def create_app():
    app_ = Flask(__name__)
    # Demo only: set the SECRET_KEY environment variable to keep sessions across restarts.
    app_.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32)
    app_.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app_.register_blueprint(routes)

    @app_.context_processor
    def inject_demo_credentials():
        # Shown on the login pages; these are demo values, not real authentication.
        return {"demo_credentials": DEMO_CREDENTIALS}

    @app_.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", title="Page not found",
                               message="That page does not exist."), 404

    @app_.errorhandler(405)
    def method_not_allowed(_error):
        return render_template("error.html", title="Not allowed",
                               message="That page cannot be opened this way. Please start from the home page."), 405

    @app_.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", title="File too large",
                               message="The uploaded file is larger than %d MB." % (MAX_UPLOAD_BYTES // (1024 * 1024))), 413

    @app_.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", title="Something went wrong",
                               message="An unexpected error occurred. Please try again."), 500

    return app_
