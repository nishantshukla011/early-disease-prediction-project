
class GPT:
    def __init__(self):
        pass

    def create_prompt(self, label):
        label = str(label).strip() if label is not None else ""

        if not label or label.lower() in {
            "none", "nothing", "normal", "no disease detected"
        }:
            return (
                "The model did not return a positive disease label. "
                "This does not rule out disease. Please consult a qualified "
                "radiologist or medical professional for interpretation."
            )

        return (
            f"The X-ray model returned the label: {label}. "
            "This is an automated model output, not a medical diagnosis. "
            "The result must be reviewed by a qualified radiologist or "
            "medical professional before any clinical decision is made."
        )
