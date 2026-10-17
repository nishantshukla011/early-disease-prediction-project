"""
Medical Imaging and DICOM Processing Module.

Added by Nishant Shukla for academic submission in October 2026.

Provides clinical radiography ingestion, supporting standard 8-bit images
(PNG, JPEG), high-dynamic-range 16-bit medical images (TIFF/PNG), and
radiological DICOM (.dcm) files with Hounsfield windowing/leveling.
"""

from io import BytesIO
import struct
import numpy as np
from PIL import Image

# Common Chest X-ray Windowing presets (Center, Width)
CHEST_WINDOW_PRESETS = {
    "lung": (-600, 1500),
    "mediastinum": (50, 350),
    "bone": (300, 1500),
    "standard": (127, 255),
}


def is_dicom_bytes(data: bytes) -> bool:
    """Return True if the byte string begins with a DICOM header."""
    if len(data) >= 132 and data[128:132] == b"DICM":
        return True
    return False


def _parse_dicom_header_tags(data: bytes) -> dict:
    """
    Lightweight, dependency-free DICOM header parser.
    Extracts essential radiological tags (Modality, View Position, Dimensions, Photometric).
    """
    metadata = {
        "modality": "CR/DX",
        "view_position": "PA",
        "photometric_interpretation": "MONOCHROME2",
        "rows": None,
        "columns": None,
        "bits_allocated": 16,
        "window_center": None,
        "window_width": None,
    }

    if not is_dicom_bytes(data):
        return metadata

    # Search for common DICOM standard tags within the first 4KB of header
    header_chunk = data[:4096]
    
    # Tag (0008, 0060) Modality: e.g. "DX" or "CR"
    idx = header_chunk.find(b"\x08\x00\x60\x00")
    if idx != -1 and idx + 8 < len(header_chunk):
        vl = struct.unpack("<H", header_chunk[idx+6:idx+8])[0]
        if vl <= 16:
            metadata["modality"] = header_chunk[idx+8:idx+8+vl].decode("latin-1", errors="ignore").strip()

    # Tag (0018, 5101) View Position: "PA", "AP", "LATERAL"
    idx = header_chunk.find(b"\x18\x00\x01\x51")
    if idx != -1 and idx + 8 < len(header_chunk):
        vl = struct.unpack("<H", header_chunk[idx+6:idx+8])[0]
        if vl <= 16:
            metadata["view_position"] = header_chunk[idx+8:idx+8+vl].decode("latin-1", errors="ignore").strip()

    # Tag (0028, 0004) Photometric Interpretation
    idx = header_chunk.find(b"\x28\x00\x04\x00")
    if idx != -1 and idx + 8 < len(header_chunk):
        vl = struct.unpack("<H", header_chunk[idx+6:idx+8])[0]
        if vl <= 32:
            metadata["photometric_interpretation"] = header_chunk[idx+8:idx+8+vl].decode("latin-1", errors="ignore").strip()

    return metadata


def apply_window_level(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Applies linear radiological window/level transformation (VOI LUT).
    Maps high-dynamic-range radiological intensities to 8-bit [0, 255].
    """
    lower = center - (width / 2.0)
    upper = center + (width / 2.0)
    clipped = np.clip(pixel_array, lower, upper)
    normalized = ((clipped - lower) / max(width, 1e-5)) * 255.0
    return normalized.astype(np.uint8)


def load_medical_image(image_input) -> tuple[Image.Image, dict]:
    """
    Safely load a medical image from raw bytes, file-like object, or file path.
    Supports DICOM (.dcm), 16-bit medical PNG/TIFF, and standard 8-bit images.

    Returns:
        (PIL.Image.Image in RGB mode, metadata dictionary)
    """
    if isinstance(image_input, (str, bytes)):
        if isinstance(image_input, str):
            with open(image_input, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = image_input
    elif hasattr(image_input, "read"):
        raw_bytes = image_input.read()
    else:
        raise ValueError("Invalid image input type. Expected filepath, bytes, or file stream.")

    if not raw_bytes:
        raise ValueError("Empty image data provided.")

    metadata = {
        "format": "STANDARD",
        "modality": "Chest Radiograph",
        "view_position": "Unknown",
        "channels": 3,
        "original_size": None,
    }

    # 1. Check if input is a DICOM stream
    if is_dicom_bytes(raw_bytes):
        metadata["format"] = "DICOM"
        dicom_tags = _parse_dicom_header_tags(raw_bytes)
        metadata.update(dicom_tags)

        # Try pydicom if available for full pixel extraction
        try:
            import pydicom
            ds = pydicom.dcmread(BytesIO(raw_bytes))
            pixel_array = ds.pixel_array.astype(np.float32)

            # Check Rescale Slope and Intercept
            slope = getattr(ds, "RescaleSlope", 1.0)
            intercept = getattr(ds, "RescaleIntercept", 0.0)
            pixel_array = pixel_array * slope + intercept

            # Invert MONOCHROME1 (where minimum pixel value is white)
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                pixel_array = np.max(pixel_array) - pixel_array

            # Window leveling
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            if isinstance(wc, (list, tuple)):
                wc = wc[0]
            if isinstance(ww, (list, tuple)):
                ww = ww[0]

            if wc is not None and ww is not None:
                uint8_img = apply_window_level(pixel_array, float(wc), float(ww))
            else:
                p_min, p_max = np.min(pixel_array), np.max(pixel_array)
                uint8_img = (((pixel_array - p_min) / max(p_max - p_min, 1e-5)) * 255.0).astype(np.uint8)

            image = Image.fromarray(uint8_img).convert("RGB")
            metadata["original_size"] = image.size
            return image, metadata
        except ImportError:
            # Fallback: if pydicom not installed, try PIL or raise descriptive warning
            pass

    # 2. Standard image loading with PIL (handles PNG, JPEG, TIFF, 16-bit Grayscale)
    try:
        pil_img = Image.open(BytesIO(raw_bytes))
        pil_img.load()
        metadata["original_size"] = pil_img.size
        metadata["mode"] = pil_img.mode

        # If 16-bit integer image (e.g. Mode 'I' or 'I;16')
        if pil_img.mode in ("I", "I;16", "I;16B", "I;16L"):
            arr = np.array(pil_img, dtype=np.float32)
            amin, amax = arr.min(), arr.max()
            norm = (((arr - amin) / max(amax - amin, 1e-5)) * 255.0).astype(np.uint8)
            pil_img = Image.fromarray(norm).convert("RGB")
        else:
            pil_img = pil_img.convert("RGB")

        return pil_img, metadata
    except Exception as exc:
        raise ValueError(f"Failed to process image: {exc}") from exc


def convert_image_to_jpeg_bytes(image: Image.Image, quality: int = 95) -> bytes:
    """Converts a PIL Image to JPEG bytes for model pipeline and PDF inclusion."""
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return buffer.getvalue()
