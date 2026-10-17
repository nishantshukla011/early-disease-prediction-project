# AI-Powered Early Disease Prediction System

## Overview
The AI-Powered Early Disease Prediction System is a Python Flask web application designed to demonstrate how machine learning can assist with medical image analysis. It includes an X-ray analysis workflow, heatmap visualization, and a web interface with separate login areas for patients, doctors, and administrators.

## Features
- Chest X-ray image upload and model-based prediction.
- Heatmap generation to visualize image regions associated with the model's output.
- Doctor dashboard with patient data and emergency-related sections.
- Flask-based web interface with HTML templates and static assets.
- Automated report generation, subject to API configuration and availability.

## Technology Stack
- Python and Flask
- TensorFlow/Keras
- OpenAI API for the original report-generation implementation
- HTML, CSS, and JavaScript

## Project Structure
- `run.py` — application entry point
- `app/routes.py` — web routes and application workflows
- `app/api.py` — X-ray prediction logic
- `app/chat.py` — report text generation
- `app/generate_heatmap.py` — heatmap generation
- `app/models/` — trained model file
- `app/templates/` — HTML pages
- `app/static/` — static assets
- `requirements.txt` — Python dependencies

## Setup
1. Install Python compatible with the dependencies in `requirements.txt`.
2. Clone this repository:
   ```bash
   git clone https://github.com/nishantshukla011/early-disease-prediction-project.git
   cd early-disease-prediction-project
   ```
3. Create and activate a virtual environment.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Configure any required API credentials using environment variables. Never commit API keys or passwords.
6. Start the application:
   ```bash
   python run.py
   ```

## Limitations
This project is an educational demonstration, not a medical diagnostic device. Predictions can be inaccurate and must not be used to make clinical decisions. Model performance depends on its training data and implementation. Some report-generation features may require external API access.

## Attribution and License
This repository is based on the original project by [pmk456](https://github.com/pmk456/AI-Powered-Early-Disease-Prediction-System). The original project includes a GPL-3.0 license; retain the license and comply with its terms when redistributing or modifying the code.

## Disclaimer
This application is intended for educational and demonstration purposes only. It is not a substitute for professional medical evaluation.
