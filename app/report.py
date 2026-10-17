"""
Local, deterministic report text for the Early Disease Prediction System.

Modified for academic submission by nishantshukla011 on 17 October 2026:
this module replaces the original app/chat.py, which called the paid OpenAI
API. It makes no network calls. The text is produced from a fixed template and
only uses the labels (and raw scores) that the trained model actually returned.
Nothing here is a diagnosis.

All text is plain ASCII so that it can be written into the PDF with the
built-in Latin-1 fonts of the fpdf library.
"""

NO_FINDING = "No Disease Detected"
THRESHOLD = 0.5

# Generic, textbook-style background for each label the model can output.
# This is NOT a finding about the patient.
LABEL_BACKGROUND = {
    "Atelectasis": "partial or complete collapse of lung tissue.",
    "Cardiomegaly": "an enlarged heart.",
    "Consolidation": "lung air spaces filled with fluid, pus or other material.",
    "Edema": "excess fluid in the lungs.",
    "Effusion": "fluid collecting in the space around the lungs.",
    "Emphysema": "long-term damage to the air sacs of the lungs.",
    "Fibrosis": "scarring of lung tissue.",
    "Hernia": "an organ or tissue pushing through an opening in the body wall or diaphragm.",
    "Infiltration": "a non-specific area where something denser than air is present in lung tissue.",
    "Mass": "a lung abnormality larger than 3 cm.",
    "Nodule": "a small, rounded lung abnormality (3 cm or smaller).",
    "Pleural_Thickening": "thickening of the lining around the lungs.",
    "Pneumonia": "an infection that inflames the air sacs of the lungs.",
    "Pneumothorax": "air in the space around the lung, which can cause the lung to collapse.",
}

DISCLAIMER = (
    "This report was produced automatically by an educational demonstration "
    "model. It is not a medical diagnosis and must not be used for clinical "
    "decisions. Any result must be reviewed by a qualified radiologist or "
    "medical professional."
)


def _ascii(text):
    """Return text that is safe for the Latin-1 PDF fonts."""
    return str(text).encode("ascii", errors="replace").decode("ascii")


def build_report_text(labels, scores=None):
    """Build the analysis paragraph(s) for the PDF.

    labels: list of label strings returned by the model, e.g. ["Edema"] or
            ["No Disease Detected"].
    scores: optional {label: float} with the raw model output for each label.
    """
    scores = scores or {}
    labels = [str(label).strip() for label in (labels or []) if str(label).strip()]
    flagged = [label for label in labels if label != NO_FINDING]

    if not flagged:
        body = (
            "The model did not score any of its 14 labels above its decision "
            "threshold of %.1f. This does not rule out disease: the model can "
            "miss findings, and a normal-looking result is not a clean bill of "
            "health." % THRESHOLD
        )
        return _ascii(body + "\n\n" + DISCLAIMER)

    lines = [
        "The model scored the following label(s) above its decision threshold "
        "of %.1f:" % THRESHOLD,
        "",
    ]
    for label in flagged:
        pretty = label.replace("_", " ")
        if label in scores:
            lines.append("- %s (raw model output: %.2f)" % (pretty, float(scores[label])))
        else:
            lines.append("- %s" % pretty)
    lines += [
        "",
        "Raw model outputs are not calibrated probabilities and should not be "
        "read as the chance that a condition is present.",
    ]

    known = [label for label in flagged if label in LABEL_BACKGROUND]
    if known:
        lines += ["", "General background (not specific to this patient):"]
        for label in known:
            lines.append(
                "- %s: %s" % (label.replace("_", " "), LABEL_BACKGROUND[label])
            )

    lines += ["", DISCLAIMER]
    return _ascii("\n".join(lines))
