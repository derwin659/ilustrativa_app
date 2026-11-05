import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from services import face_detector

class FaceDetectorTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (300, 300), "white")

    def test_bundled_detector_loads_without_external_weights(self):
        with patch.object(face_detector, "_detector", None):
            detector = face_detector.initialize_face_detector()
            self.assertFalse(detector.empty())
            self.assertIsNone(face_detector.get_face_bbox(self.image))

    def test_single_face_is_converted_and_clamped(self):
        detector = Mock()
        detector.detectMultiScale.return_value = np.array([[20, 30, 290, 280]])
        with patch.object(face_detector, "_detector", detector):
            self.assertEqual(face_detector.get_face_bbox(self.image), [20, 30, 300, 300])

    def test_no_or_multiple_faces_do_not_choose_another_person(self):
        for boxes in ([], [[10, 10, 80, 80], [130, 10, 90, 90]]):
            detector = Mock()
            detector.detectMultiScale.return_value = boxes
            with patch.object(face_detector, "_detector", detector):
                self.assertIsNone(face_detector.get_face_bbox(self.image))

    def test_missing_cascade_fails_before_accepting_work(self):
        with patch.object(face_detector, "_detector", None), patch.object(face_detector.cv2, "CascadeClassifier") as factory:
            factory.return_value.empty.return_value = True
            with self.assertRaisesRegex(RuntimeError, "cascade"):
                face_detector.initialize_face_detector()
