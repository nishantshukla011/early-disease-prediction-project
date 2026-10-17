"""
Original author: Nishant Shukla
Original date & time: 17-10-2026 01:14 AM




import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=3000, debug=debug)
