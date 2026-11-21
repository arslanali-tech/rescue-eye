"""RescueEye: open-vocabulary and distilled object detection for aerial flood rescue imagery (CPU-friendly)."""
__version__ = "0.1.0"

CLASSES = ["person", "vehicle", "boat", "flooded_building"]
CAT_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
ID_CAT = {v: k for k, v in CAT_ID.items()}
