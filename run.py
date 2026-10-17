"""
Modified by Nishant Shukla, based on pmk456's project.
Upstream repository: https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System (GPL-3.0)
Modification date: 17-10-2026
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=3000, debug=debug)
