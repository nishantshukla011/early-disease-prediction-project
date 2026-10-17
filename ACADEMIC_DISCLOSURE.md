# Academic Integrity & Attribution Disclosure

**Project Title:** AI-Powered Early Disease Prediction System  
**Author / Contributor:** Nishant Shukla (`nishantshukla011`)  
**Submission Context:** Academic Coursework / Educational Project  
**Date:** October 2026  
**License:** GNU General Public License v3.0 (GPL-3.0)

---

## 1. Upstream Base Project Attribution

This project is an academic fork and technical enhancement of the open-source repository:
- **Project Name:** AI-Powered Early Disease Prediction System
- **Original Author:** Patan Musthakheem (`pmk456`)
- **Upstream Repository:** [https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System](https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System)
- **Original Date:** March 2025
- **License:** GNU General Public License v3.0 (GPL-3.0)

Under the terms of the GNU General Public License v3.0, downstream users are legally permitted to study, modify, and redistribute the work provided that the original copyright notice, attribution, and license text are preserved and that modified files carry prominent notices stating that they have been changed.

---

## 2. Statement of Contributions and Modifications

In accordance with academic integrity guidelines, the following breakdown specifies the original upstream code versus the original contributions developed by Nishant Shukla:

### A. Inherited Upstream Baseline (`pmk456`)
- Base Flask application structure, initial HTML templates, and basic CSS stylesheets.
- Core deep learning model architecture and weights (`app/models/chest_xray_model.keras`), trained on the NIH ChestX-ray14 dataset.
- Baseline Grad-CAM activation algorithm structure (`app/generate_heatmap.py`).
- Initial model training and evaluation scripts (`app/model_train/`).

### B. Original Modifications & Technical Enhancements (by Nishant Shukla)
1. **Architectural Modularization & Centralized Configuration (`app/config.py`):**
   - Centralized application settings, absolute directory paths, upload thresholds, and demo credential constants to remove scattered magic numbers and working-directory dependencies.
2. **Deterministic Offline Report Generator (`app/report.py`):**
   - Replaced the external, paid, cloud-dependent OpenAI API call (`app/chat.py`) with a local, deterministic medical report generation template.
   - Eliminates third-party API keys, network latency, and privacy issues with transmitting patient medical data over external APIs.
   - Ensured pure ASCII/Latin-1 encoding compatibility with `fpdf`.
3. **Medical Imaging Pipeline & Extended Format Ingestion (`app/medical_imaging.py`):**
   - Introduced a clinical radiology ingestion engine supporting 16-bit medical radiographs and DICOM (`.dcm`) standard files.
   - Implemented radiological window/level normalization (Hounsfield windowing) and metadata extraction (view position, modality, photometric interpretation).
4. **Clinical Risk Stratification & Pathology Co-occurrence Engine (`app/report.py`):**
   - Upgraded flat binary thresholding into calibrated clinical confidence bands: High Suspicion ($\ge 0.70$), Moderate Suspicion ($0.50 - 0.70$), and Borderline ($0.35 - 0.50$).
   - Implemented a clinical co-occurrence correlation analyzer identifying established medical syndrome pairings in the NIH dataset (e.g., Cardiomegaly + Edema suggesting heart failure; Consolidation + Effusion indicating pneumonia).
5. **Epidemiological & Error Analysis Suite (`app/model_analysis.py`):**
   - Implemented an offline analytical tool parsing NIH ChestX-ray14 metadata (`Data_Entry_2017.csv` and `BBox_List_2017.csv`).
   - Computes multi-label disease prevalence, co-occurrence matrices, and classification metric evaluations (Precision, Recall, F1, multi-label Hamming loss).
6. **Input Validation, File Sanitization, and Security:**
   - Implemented secure file upload handling: file size limiting, MIME/magic byte validation using PIL, path traversal sanitization, and automatic deletion of temporary medical uploads immediately following report generation.
   - Built custom error handling templates (`app/templates/error.html`) to suppress raw application stack traces and sensitive server information.
7. **Automated Unit Testing Suite (`app/tests/test.py`):**
   - Developed comprehensive unit tests verifying routes, session authentication boundaries, file validation, deterministic report generation, medical imaging ingestion, and mock model pipelines.

---

## 3. Transparency & Academic Integrity Declaration

I declare that:
1. I have openly acknowledged the foundational work of Patan Musthakheem (`pmk456`).
2. All modifications, refactorings, and newly created modules documented in Section 2(B) represent my own work.
3. This project complies fully with the open-source GNU General Public License v3.0 under which the upstream software was distributed.
4. This disclosure has been provided in the root repository to ensure complete transparency during academic evaluation.
