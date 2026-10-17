# AI-Powered Early Disease Prediction System

A Python Flask web application that demonstrates how a deep-learning model can assist with chest X-ray analysis. A doctor uploads a chest X-ray, the model returns its predicted labels, a Grad-CAM heatmap is drawn, and a PDF report is produced.

> **Educational demonstration only.** This is not a medical device and its output is not a diagnosis. See [Limitations and disclaimer](#limitations-and-disclaimer).

## Credits and Attribution

This project is an academic fork and technical enhancement of:
- **Upstream Project:** [AI-Powered-Early-Disease-Prediction-System](https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System)
- **Original Author:** Patan Musthakheem ([pmk456](https://github.com/pmk456))
- **Original Release:** March 2025
- **License:** [GNU General Public License v3.0 (GPL-3.0)](LICENSE)

For complete academic attribution and integrity details, please see [ACADEMIC_DISCLOSURE.md](ACADEMIC_DISCLOSURE.md).

### Summary of Modifications & Original Contributions (by Nishant Shukla):
- **Centralized Configuration (`app/config.py`):** Replaced hardcoded paths, magic numbers, and credentials with centralized configuration.
- **Offline Report Engine (`app/report.py`):** Replaced external paid OpenAI API dependency with a local, deterministic medical report generator that complies with patient privacy.
- **Clinical Risk Stratification:** Upgraded binary thresholding to multi-tier confidence levels (High, Moderate, Borderline) and added disease co-occurrence syndrome analysis.
- **Medical Imaging Pipeline (`app/medical_imaging.py`):** Added support for radiological DICOM (`.dcm`) ingestion and 16-bit medical radiography window/level normalization.
- **Dataset Epidemiology & Error Analysis (`app/model_analysis.py`):** Added offline statistical analysis for class imbalance, label sparsity, and multi-label metrics on NIH ChestX-ray14.
- **Upload Security & Error Handling:** Implemented file validation, automatic deletion of temporary medical uploads, and custom error pages (`app/templates/error.html`).
- **Automated Testing Suite (`app/tests/test.py`):** Added unit test suite verifying routing, security bounds, report formatting, and mock model pipelines.

## Features

- Home page with separate Admin, Doctor and Patient login pages.
- Doctor login (demo credentials) and patient login (demo credentials).
- Chest X-ray upload (PNG, JPG, or DICOM) with validation; uploaded files are deleted after the report is built.
- Prediction with the trained Keras model `app/models/chest_xray_model.keras` (14 NIH ChestX-ray14 labels).
- Grad-CAM heatmap overlay.
- PDF report built with a local, deterministic template with risk stratification and co-occurrence insights.
- Friendly error pages instead of tracebacks.

Not implemented in this demo (the buttons say so): MRI analysis, patient records, doctor assignment, admin statistics, enquiries.

## Technology stack

- Python 3.11 (recommended), Flask 3.1
- TensorFlow 2.19 / Keras 3.9 (EfficientNetB0-based model)
- OpenCV, Pillow, NumPy, Pandas
- fpdf 1.7.2 for PDF output
- HTML, CSS and a little JavaScript

## Project structure

```
.
├── run.py                      # starts the web server
├── requirements.txt
├── LICENSE                     # GNU GPL v3
├── ACADEMIC_DISCLOSURE.md      # academic integrity and fork disclosure
├── README.md
└── app/
    ├── __init__.py             # Flask app factory, error pages
    ├── config.py               # paths, upload limits, DEMO credentials
    ├── routes.py               # all web routes
    ├── api.py                  # X-ray prediction (XRayPredictor)
    ├── medical_imaging.py      # DICOM & medical radiography pipeline
    ├── generate_heatmap.py     # Grad-CAM heatmap
    ├── report.py               # local report template with risk stratification
    ├── model_analysis.py       # dataset epidemiology & error analysis tool
    ├── utils.py                # PDF builder
    ├── models/chest_xray_model.keras
    ├── model_train/            # original training scripts and NIH label CSVs
    ├── templates/              # HTML pages
    ├── static/                 # CSS, JS, images, heatmap output folder
    └── tests/test.py           # automated tests
```

## Installation and running (Windows PowerShell)

Install **Python 3.11** from python.org first. TensorFlow 2.19 supports Python 3.9 to 3.12 and does **not** support Python 3.13.

```powershell
git clone https://github.com/nishantshukla011/early-disease-prediction-project.git
cd early-disease-prediction-project
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py
```

Open <http://127.0.0.1:3000> in your browser.

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Run the automated tests (from the project folder, with the virtual environment active):

```powershell
python -m unittest discover -s app/tests -p test.py -v
```

Troubleshooting on Windows: if TensorFlow reports `DLL load failed`, install the "Microsoft Visual C++ Redistributable 2015-2022". If `pip` fails with a path-too-long error, clone the project to a short folder such as `C:\edp`.

## Demo login

The login pages show the demo credentials. They are defined in `app/config.py`:

| Role    | User ID  | Password |
|---------|----------|----------|
| Admin   | `nishant` | `nishant` |
| Doctor  | `nishant` | `nishant` |
| Patient | `nishant` | `nishant` |

These are **demo credentials, not real authentication**. There is no database, no password hashing and no user management. Never deploy this application publicly with them.

## Using the application

1. Log in as a doctor and choose **X-RAY Insight**.
2. Upload a chest X-ray image (PNG or JPG).
3. The browser downloads `xray_report.pdf` containing the uploaded image, the model output, the heatmap and a disclaimer.

## How the local report generation works

`app/report.py` builds the report text from a fixed template. It makes no network calls and uses no language model.

- It only uses the labels (and raw scores) that the model actually returned. A label is returned when the model's sigmoid output for it is above 0.5.
- If no label passes the threshold, the report says so and states that this does not rule out disease.
- For each returned label it adds one line of generic background about the condition. This is textbook-style context, not a finding about the patient.
- Every report ends with a disclaimer that the output is not a diagnosis.
- The same input always produces the same text.







