"""
Original author: Patan Musthakheem
Original date & time: 20-03-2025 1:59 AM
Original project: https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System (GPL-3.0)

Modified for academic submission by nishantshukla011 on 17 October 2026:
- demo credentials moved to app/config.py and clearly labelled as demo only
- fixed the admin login route, the missing patient "check report" route and the
  broken MRI route (MRI analysis is not implemented in this project)
- safe uploads (extension check, size limit, image validation, temp files are
  deleted after use) and clear error pages instead of tracebacks
- the OpenAI-based report was replaced by the local template in app/report.py
- TensorFlow is loaded on first use, so the login pages work even if the model
  cannot be loaded
"""

import functools
import hmac
import os
import uuid
from io import BytesIO

from flask import (Blueprint, Response, current_app, redirect, render_template,
                   request, session, url_for)
from PIL import Image

from app import report, utils
from app.config import ALLOWED_EXTENSIONS, DEMO_CREDENTIALS, HEATMAP_DIR, UPLOAD_DIR

routes = Blueprint("routes", __name__)


# ---------------------------------------------------------------- helpers

def _error(message, status=400, title="Error"):
    return render_template("error.html", title=title, message=message), status


def _credentials_ok(role, user_id, password):
    """Demo-only check against app/config.py. This is NOT production authentication."""
    expected_id, expected_pass = DEMO_CREDENTIALS[role]
    return (hmac.compare_digest((user_id or "").encode("utf-8"), expected_id.encode("utf-8"))
            and hmac.compare_digest((password or "").encode("utf-8"), expected_pass.encode("utf-8")))


def _logged_in(role):
    return session.get("role") == role


def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@functools.lru_cache(maxsize=1)
def _get_models():
    """Load the predictor and heatmap generator once (this imports TensorFlow)."""
    from app import api, generate_heatmap
    return api.XRayPredictor(), generate_heatmap.XRayHeatmapGenerator()


def _remove(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


# ------------------------------------------------------------------- home

@routes.route("/", methods=["GET"])
def index():
    return render_template("home.html")


# ------------------------------------------------------------------ admin

@routes.route("/admin-login")
def admin_login():
    return render_template("admin-login.html")


@routes.route("/admin-login/admin-dashboard", methods=["POST"])
def admin_login_submit():
    if _credentials_ok("admin", request.form.get("adminID"), request.form.get("adminPASS")):
        session["role"] = "admin"
        return redirect(url_for("routes.admin_dashboard"))
    return _error("Incorrect Admin ID or password.", 401, "Login failed")


@routes.route("/admin-dashboard", methods=["GET"])
def admin_dashboard():
    if not _logged_in("admin"):
        return redirect(url_for("routes.admin_login"))
    return render_template("admin-dash.html")


# ----------------------------------------------------------------- doctor

@routes.route("/doctor-login")
def doctor_login():
    return render_template("doc-login.html")


@routes.route("/doctor-login/doctor-dashboard", methods=["GET", "POST"])
def doctor_dashboard():
    if request.method == "POST":
        if _credentials_ok("doctor", request.form.get("doctorId"), request.form.get("doctorPass")):
            session["role"] = "doctor"
            return render_template("doc-dash.html")
        return _error("Incorrect Doctor ID or password.", 401, "Login failed")
    if not _logged_in("doctor"):
        return redirect(url_for("routes.doctor_login"))
    return render_template("doc-dash.html")


@routes.route("/doctor-login/doctor-dashboard/doctor-upload-xray", methods=["GET", "POST"])
def doctor_upload_xray():
    if not _logged_in("doctor"):
        return _error("Please log in first.", 401, "Login required")
    if request.method == "GET":
        return render_template("doc-upload.html")

    upload = request.files.get("file")
    if upload is None or upload.filename == "":
        return _error("Please choose an X-ray image to upload.")
    if not _allowed_file(upload.filename):
        return _error("Only PNG or JPG images are accepted.")

    raw = upload.read()
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        image = image.convert("RGB")
    except Exception:
        return _error("The uploaded file is not a valid image.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex
    xray_path = os.path.join(UPLOAD_DIR, job_id + ".jpg")  # RGB copy used only inside the PDF
    overlay_name = job_id + ".png"
    overlay_path = os.path.join(HEATMAP_DIR, overlay_name)

    try:
        image.save(xray_path, "JPEG", quality=95)

        try:
            predictor, heatmap_generator = _get_models()
        except Exception:
            current_app.logger.exception("Model could not be loaded")
            return _error("The prediction model could not be loaded. Check the server console for details.",
                          500, "Model unavailable")

        result = predictor.predict(raw)
        if "error" in result:
            return _error(result["error"])

        heatmap = heatmap_generator.generate_heatmap(raw)
        if heatmap is None:
            return _error("The heatmap could not be generated for this image.", 500)
        heatmap_generator.overlay_heatmap(heatmap, raw, output_filename=overlay_name)

        report_text = report.build_report_text(result["prediction"], result.get("scores"))
        pdf_bytes = utils.generate_pdf(xray_path, overlay_path, report_text)

        response = Response(pdf_bytes, content_type="application/pdf")
        response.headers["Content-Disposition"] = "attachment; filename=xray_report.pdf"
        return response
    except Exception:
        current_app.logger.exception("Report generation failed")
        return _error("The report could not be generated. Check the server console for details.",
                      500, "Report failed")
    finally:
        _remove(xray_path, overlay_path)


@routes.route("/doctor-login/doctor-dashboard/doctor-upload-mri", methods=["GET", "POST"])
def doctor_upload_mri():
    if not _logged_in("doctor"):
        return _error("Please log in first.", 401, "Login required")
    return _error("MRI analysis is not implemented in this demo. Please use X-Ray Insight.",
                  501, "Not available")


# ---------------------------------------------------------------- patient

@routes.route("/patient-login")
def patient_login():
    return render_template("patient-login.html")


@routes.route("/patient-login/patient-dashboard", methods=["GET", "POST"])
def patient_dashboard():
    if request.method == "POST":
        if _credentials_ok("patient", request.form.get("patientID"), request.form.get("patientPASS")):
            session["role"] = "patient"
            return render_template("patient-dash.html")
        return _error("Incorrect Patient ID or password.", 401, "Login failed")
    if not _logged_in("patient"):
        return redirect(url_for("routes.patient_login"))
    return render_template("patient-dash.html")


@routes.route("/patient-login/check-report")
def patient_check_report():
    if not _logged_in("patient"):
        return redirect(url_for("routes.patient_login"))
    return _error("Patient records are not part of this demo.", 501, "Not available")
