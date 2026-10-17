"""
Tests for the Early Disease Prediction System (educational demo).

Added for academic submission by nishantshukla011 on 17 October 2026.

Run from the project root:
    python -m unittest discover -s app/tests -p test.py -v

The TensorFlow model is replaced by a small fake so the tests run in seconds.
Everything else (routes, upload validation, report text and, when the fpdf
package is installed, the real PDF builder) is the real code.
"""

import importlib
import os
import sys
import types
import unittest
from io import BytesIO
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import fpdf  # noqa: F401
    FPDF_IS_REAL = True
except ImportError:  # keeps the route tests runnable on a machine without fpdf
    FPDF_IS_REAL = False

    class _FakeFPDF:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

        def output(self, dest="S"):
            return "%PDF-fake"

    sys.modules["fpdf"] = types.SimpleNamespace(FPDF=_FakeFPDF)

from PIL import Image  # noqa: E402

from app import create_app, report  # noqa: E402
from app.config import DEMO_CREDENTIALS  # noqa: E402

# app/__init__.py does `from app.routes import routes`, so the attribute `app.routes`
# is the Blueprint; import the module explicitly.
routes_module = importlib.import_module("app.routes")


def make_png(mode="RGB", size=(64, 64)):
    buffer = BytesIO()
    Image.new(mode, size, "gray" if mode != "RGBA" else (128, 128, 128, 255)).save(buffer, "PNG")
    buffer.seek(0)
    return buffer


class FakePredictor:
    def __init__(self, labels):
        self.labels = labels

    def predict(self, _image_bytes):
        scores = {l: 0.91 for l in self.labels if l != report.NO_FINDING}
        return {"prediction": self.labels, "scores": scores}


class FakeHeatmap:
    def generate_heatmap(self, _image_bytes):
        return [[0.0]]

    def overlay_heatmap(self, _heatmap, _image_bytes, output_filename="x.png", **_kw):
        from app.config import HEATMAP_DIR
        os.makedirs(HEATMAP_DIR, exist_ok=True)
        path = os.path.join(HEATMAP_DIR, output_filename)
        Image.new("RGB", (224, 224), "red").save(path)
        return path


class ReportTextTests(unittest.TestCase):
    def test_uses_only_returned_labels(self):
        text = report.build_report_text(["Edema"], {"Edema": 0.73})
        self.assertIn("Edema", text)
        self.assertIn("0.73", text)
        self.assertNotIn("Pneumonia", text)
        self.assertIn("not a medical diagnosis", text)

    def test_no_finding_does_not_claim_healthy(self):
        text = report.build_report_text([report.NO_FINDING])
        self.assertIn("does not rule out disease", text)

    def test_is_deterministic_and_ascii(self):
        a = report.build_report_text(["Mass", "Nodule"], {"Mass": 0.6, "Nodule": 0.55})
        b = report.build_report_text(["Mass", "Nodule"], {"Mass": 0.6, "Nodule": 0.55})
        self.assertEqual(a, b)
        a.encode("ascii")

    def test_never_says_confirmed(self):
        text = report.build_report_text(["Pneumonia"], {"Pneumonia": 0.99}).lower()
        self.assertNotIn("confirmed", text.replace("not a confirmed", ""))


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def login_doctor(self):
        user, password = DEMO_CREDENTIALS["doctor"]
        return self.client.post("/doctor-login/doctor-dashboard",
                                data={"doctorId": user, "doctorPass": password})

    def use_fake_models(self, labels):
        patcher = mock.patch.object(
            routes_module, "_get_models", lambda: (FakePredictor(labels), FakeHeatmap()))
        patcher.start()
        self.addCleanup(patcher.stop)

    # --- pages
    def test_public_pages_load(self):
        for url in ("/", "/doctor-login", "/patient-login", "/admin-login"):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_unknown_page_is_friendly(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        self.assertIn(b"Page not found", response.data)

    def test_no_openai_or_old_branding_in_pages(self):
        page = self.client.get("/").data.lower()
        self.assertNotIn(b"openai", page)
        self.assertNotIn(b"alliance", page)

    # --- login
    def test_doctor_login_ok_and_bad(self):
        self.assertEqual(self.login_doctor().status_code, 200)
        bad = self.client.post("/doctor-login/doctor-dashboard",
                               data={"doctorId": "x", "doctorPass": "y"})
        self.assertEqual(bad.status_code, 401)

    def test_patient_login_ok_and_bad(self):
        user, password = DEMO_CREDENTIALS["patient"]
        ok = self.client.post("/patient-login/patient-dashboard",
                              data={"patientID": user, "patientPASS": password})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post("/patient-login/patient-dashboard",
                               data={"patientID": user, "patientPASS": "wrong"})
        self.assertEqual(bad.status_code, 401)

    def test_admin_login_form_target_exists(self):
        user, password = DEMO_CREDENTIALS["admin"]
        response = self.client.post("/admin-login/admin-dashboard",
                                    data={"adminID": user, "adminPASS": password},
                                    follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Admin Dashboard", response.data)

    def test_non_ascii_password_does_not_crash(self):
        response = self.client.post("/doctor-login/doctor-dashboard",
                                    data={"doctorId": "mahima", "doctorPass": "päss"})
        self.assertEqual(response.status_code, 401)

    # --- protected pages
    def test_upload_requires_login(self):
        self.assertEqual(self.client.get("/doctor-login/doctor-dashboard/doctor-upload-xray").status_code, 401)

    def test_mri_is_a_clear_message_not_a_crash(self):
        self.login_doctor()
        response = self.client.get("/doctor-login/doctor-dashboard/doctor-upload-mri")
        self.assertEqual(response.status_code, 501)
        self.assertIn(b"not implemented", response.data)

    # --- upload validation
    def post_file(self, data, name):
        return self.client.post("/doctor-login/doctor-dashboard/doctor-upload-xray",
                                data={"file": (data, name)}, content_type="multipart/form-data")

    def test_rejects_missing_wrong_type_and_fake_images(self):
        self.login_doctor()
        no_file = self.client.post("/doctor-login/doctor-dashboard/doctor-upload-xray", data={})
        self.assertEqual(no_file.status_code, 400)
        self.assertEqual(self.post_file(BytesIO(b"hello"), "notes.txt").status_code, 400)
        self.assertEqual(self.post_file(BytesIO(b"not an image"), "fake.png").status_code, 400)

    def test_path_traversal_filename_is_harmless(self):
        self.login_doctor()
        self.use_fake_models([report.NO_FINDING])
        response = self.post_file(make_png(), "../../evil.png")
        self.assertEqual(response.status_code, 200)

    def test_rejects_oversized_upload(self):
        self.login_doctor()
        self.app.config["MAX_CONTENT_LENGTH"] = 1024
        response = self.post_file(BytesIO(b"0" * 5000), "big.png")
        self.assertEqual(response.status_code, 413)

    # --- end to end (model faked)
    def test_upload_returns_pdf_and_cleans_up(self):
        from app.config import UPLOAD_DIR, HEATMAP_DIR
        self.login_doctor()
        self.use_fake_models(["Edema"])
        before_up = set(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else set()
        before_hm = set(os.listdir(HEATMAP_DIR))
        response = self.post_file(make_png("RGBA"), "xray.png")  # alpha PNG must not break the PDF
        self.assertEqual(response.status_code, 200, response.data[:200])
        self.assertEqual(response.mimetype, "application/pdf")
        self.assertTrue(response.data.startswith(b"%PDF"))
        after_up = set(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else set()
        self.assertEqual(before_up, after_up)
        self.assertEqual(before_hm, set(os.listdir(HEATMAP_DIR)))

    def test_model_load_failure_is_reported_not_raised(self):
        self.login_doctor()

        def broken():
            raise RuntimeError("no model")
        patcher = mock.patch.object(routes_module, "_get_models", broken)
        patcher.start()
        self.addCleanup(patcher.stop)
        response = self.post_file(make_png(), "xray.png")
        self.assertEqual(response.status_code, 500)
        self.assertIn(b"could not be loaded", response.data)


@unittest.skipUnless(FPDF_IS_REAL, "fpdf not installed; real PDF builder not exercised")
class RealPdfTests(unittest.TestCase):
    def test_pdf_builder_with_real_fpdf(self):
        import tempfile
        from app import utils
        with tempfile.TemporaryDirectory() as tmp:
            xray = os.path.join(tmp, "x.jpg")
            heat = os.path.join(tmp, "h.png")
            Image.new("RGB", (300, 300), "gray").save(xray)
            Image.new("RGB", (224, 224), "red").save(heat)
            pdf = utils.generate_pdf(xray, heat, report.build_report_text(["Mass"], {"Mass": 0.7}))
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
