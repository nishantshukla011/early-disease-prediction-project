"""
Original author: Patan Musthakheem
Original date & time: 20-03-2025 03:14 AM
Original project: https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System (GPL-3.0)

Modified for academic submission by nishantshukla011 on 17 October 2026:
the secret key now lives in create_app(), and the server listens on
localhost only with debug mode off unless FLASK_DEBUG=1 is set.
"""

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=3000, debug=debug)
