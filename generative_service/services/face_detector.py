"""Face localization only. OpenCV bundled cascade; no identity embeddings."""
import threading
from pathlib import Path
import cv2
import numpy as np

_detector = None
_lock = threading.Lock()

def initialize_face_detector():
    global _detector
    with _lock:
        if _detector is None:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(str(path))
            if detector.empty():
                raise RuntimeError("OpenCV face cascade is missing or invalid")
            _detector = detector
    return _detector

def get_face_bbox(image):
    detector = initialize_face_detector()
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    with _lock:
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=6,
                                         minSize=(60, 60))
    if len(faces) != 1:
        return None
    x, y, w, h = (int(v) for v in faces[0])
    height, width = gray.shape
    if w <= 0 or h <= 0:
        return None
    return [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
