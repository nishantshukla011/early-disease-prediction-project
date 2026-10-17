"""
Chest X-ray prediction using the trained model in app/models/.

Original code: pmk456 / AI-Powered-Early-Disease-Prediction-System (GPL-3.0).
Modified for academic submission by nishantshukla011 on 17 October 2026:
model path made independent of the working directory, raw model scores
returned alongside the labels, and the manual test block takes an image path.
"""

import sys
from io import BytesIO

import numpy as np
import tensorflow as tf
from PIL import Image

from app.config import MODEL_PATH

IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Same order as the training script (sorted NIH ChestX-ray14 labels, without "No Finding").
ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]


class XRayPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)

    def predict(self, image_bytes):
        transformed_image = self._transform_image(image_bytes)
        if transformed_image is None:
            return {"error": "The file could not be read as an image."}

        predictions = self.model.predict(transformed_image, verbose=0)[0]
        flagged = np.where(predictions > THRESHOLD)[0]
        predicted_labels = [ALL_LABELS[i] for i in flagged]

        return {
            "prediction": predicted_labels if predicted_labels else ["No Disease Detected"],
            "scores": {ALL_LABELS[i]: float(predictions[i]) for i in flagged},
        }

    def _transform_image(self, image_bytes):
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")  # Convert to RGB
            img = img.resize(IMG_SIZE)  # Resize to match model input
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            return img_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None


if __name__ == "__main__":
    # Manual check:  python -m app.api path/to/xray.png
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.api path/to/xray.png")
    with open(sys.argv[1], "rb") as f:
        print("Prediction:", XRayPredictor().predict(f.read()))
