"""
Local, deterministic report text and clinical risk stratification.

Modified and extended by Nishant Shukla for academic submission in October 2026:
- Replaces original app/chat.py (which called the paid external OpenAI API) with
  a deterministic offline report engine.
- Added multi-tier clinical risk stratification (High, Moderate, Borderline).
- Added multi-label syndrome co-occurrence correlation analysis based on known
  NIH ChestX-ray14 pathology associations.
- Strictly deterministic, offline, and safe for Latin-1/ASCII PDF generation.
- Never outputs a clinical diagnosis or claims healthiness.
"""

NO_FINDING = "No Disease Detected"
THRESHOLD = 0.5
HIGH_THRESHOLD = 0.70
BORDERLINE_THRESHOLD = 0.35

# Generic, textbook-style background for each label the model can output.
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

# Established multi-label co-occurrence patterns in chest radiography
COOCCURRENCE_PATTERNS = [
    (
        {"Cardiomegaly", "Edema"},
        "Cardiomegaly and Edema: Frequent dual-presentation commonly associated with congestive cardiac insufficiency or fluid overload.",
    ),
    (
        {"Consolidation", "Effusion"},
        "Consolidation and Effusion: Co-presentation frequently observed in acute pneumonia and parapneumonic effusions.",
    ),
    (
        {"Atelectasis", "Effusion"},
        "Atelectasis and Effusion: Pleural fluid accumulation may cause secondary compressive loss of adjacent lung volume.",
    ),
    (
        {"Infiltration", "Consolidation"},
        "Infiltration and Consolidation: Overlapping radiographic density patterns characteristic of acute inflammatory or infectious processes.",
    ),
    (
        {"Emphysema", "Pneumothorax"},
        "Emphysema and Pneumothorax: Chronic alveolar hyperinflation and bullous changes may heighten susceptibility to spontaneous pleural air entry.",
    ),
]

DISCLAIMER = (
    "This report was produced automatically by an educational demonstration "
    "model. It is not a medical diagnosis and must not be used for clinical "
    "decisions. Any result must be reviewed by a qualified radiologist or "
    "medical professional."
)


def _ascii(text):
    """Return text that is safe for the Latin-1 PDF fonts."""
    return str(text).encode("ascii", errors="replace").decode("ascii")


def get_confidence_tier(score: float) -> str:
    """Categorizes model confidence score into qualitative clinical tiers."""
    if score >= HIGH_THRESHOLD:
        return "High Suspicion (score >= 0.70)"
    elif score >= THRESHOLD:
        return "Moderate Suspicion (0.50 <= score < 0.70)"
    elif score >= BORDERLINE_THRESHOLD:
        return "Borderline / Low Suspicion (0.35 <= score < 0.50)"
    else:
        return "Below Threshold"


def find_cooccurrence_patterns(flagged_labels: list[str]) -> list[str]:
    """Identifies known pathology co-occurrence patterns among flagged labels."""
    label_set = set(flagged_labels)
    identified = []
    for pair_set, note in COOCCURRENCE_PATTERNS:
        if pair_set.issubset(label_set):
            identified.append(note)
    return identified


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
            score_val = float(scores[label])
            tier = get_confidence_tier(score_val)
            lines.append("- %s (raw model output: %.2f - %s)" % (pretty, score_val, tier))
        else:
            lines.append("- %s" % pretty)

    lines += [
        "",
        "Raw model outputs are not calibrated probabilities and should not be "
        "read as the chance that a condition is present.",
    ]

    # Clinical Co-occurrence Insights
    cooccurrences = find_cooccurrence_patterns(flagged)
    if cooccurrences:
        lines += [
            "",
            "Clinical Co-occurrence Analysis (NIH ChestX-ray14 correlation):",
        ]
        for note in cooccurrences:
            lines.append("- %s" % note)

    # General condition background
    known = [label for label in flagged if label in LABEL_BACKGROUND]
    if known:
        lines += ["", "General background (not specific to this patient):"]
        for label in known:
            lines.append(
                "- %s: %s" % (label.replace("_", " "), LABEL_BACKGROUND[label])
            )

    lines += ["", DISCLAIMER]
    return _ascii("\n".join(lines))
