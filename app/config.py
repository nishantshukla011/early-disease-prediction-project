"""
Application settings for the Early Disease Prediction System (educational demo).

Modified for academic submission by nishantshukla011 on 17 October 2026:
this module is new. It centralises settings that were previously hard-coded
inside routes.py.

DEMO ONLY: the credentials below are public demo values so the project can be
shown locally. They are NOT real authentication. Do not reuse them anywhere and
do not deploy this application to the internet with them.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "chest_xray_model.keras")
HEATMAP_DIR = os.path.join(BASE_DIR, "static", "heatmaps")
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")

# --- DEMO CREDENTIALS (not production authentication) -----------------------
# Same demo account for all three roles, taken from the original project.
DEMO_CREDENTIALS = {
    "admin": ("mahima", "mahima"),
    "doctor": ("mahima", "mahima"),
    "patient": ("mahima", "mahima"),
}

# --- Upload limits ----------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

APP_TITLE = "AI-Powered Early Disease Prediction System"
