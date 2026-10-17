"""
Grad-CAM heatmap generation for the chest X-ray model.

Original code: pmk456 / AI-Powered-Early-Disease-Prediction-System (GPL-3.0).
Modified for academic submission by nishantshukla011 on 17 October 2026:
model and output paths made independent of the working directory, guard
against a division by zero when the heatmap is all zeros, and the manual test
block takes an image path.
"""

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
import sys

from app.config import MODEL_PATH, HEATMAP_DIR

IMG_SIZE = (224, 224)
OUTPUT_DIR = HEATMAP_DIR  # Directory to save heatmap images

class XRayHeatmapGenerator:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure directory exists
    def _transform_image(self, image_bytes):
        """Preprocess image like in XRayPredictor"""
        try:
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img, dtype=np.float32)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def generate_heatmap(self, image_bytes, class_index=None):
        """Generates a Grad-CAM heatmap for the input image."""
        img_array = self._transform_image(image_bytes)
        if img_array is None:
            return None
        # Get the last convolutional layer
        last_conv_layer = self.model.get_layer("top_conv")  # Adjust layer name if different
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [last_conv_layer.output, self.model.output]
        )
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = np.argmax(predictions[0])  # Use most probable class
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        # Multiply each channel by its gradient importance
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.math.reduce_max(heatmap)
        heatmap = heatmap / max_value if float(max_value) > 0 else heatmap  # Normalize
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, image_bytes, output_filename="heatmap_overlay.png", alpha=0.5):
        """Overlays the Grad-CAM heatmap on the original image and saves it."""
        img = Image.open(BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
        img = np.array(img, dtype=np.uint8)
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        # Apply colormap
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Blend heatmap with original image
        overlay = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
        # Save the output image
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        Image.fromarray(overlay).save(output_path)

        return output_path  # Return the path of the saved heatmap image

if __name__ == "__main__":
    # Manual check:  python -m app.generate_heatmap path/to/xray.png
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.generate_heatmap path/to/xray.png")
    heatmap_generator = XRayHeatmapGenerator()
    with open(sys.argv[1], "rb") as f:
        image_bytes = f.read()
    heatmap = heatmap_generator.generate_heatmap(image_bytes)
    if heatmap is not None:
        output_path = heatmap_generator.overlay_heatmap(heatmap, image_bytes)
        print(f"Heatmap saved at: {output_path}")
    else:
        print("Error generating heatmap.")
