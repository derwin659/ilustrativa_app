import base64
import unittest
from io import BytesIO
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image, ImageDraw
from fastapi import HTTPException
from app import analizar, AnalizarRequest, face_geometry

class PortraitValidationTest(unittest.TestCase):
    def request(self, image):
        stream=BytesIO(); image.save(stream,'PNG')
        return AnalizarRequest(imagen_base64=base64.b64encode(stream.getvalue()).decode())
    def test_sales_screen_does_not_get_haircut_recommendations(self):
        image=Image.new('RGB',(480,720),'white'); d=ImageDraw.Draw(image)
        d.text((20,20),'VENTAS DEL DIA  TOTAL S/ 450',fill='black')
        for i in range(7): d.rectangle((20+i*60,500-i*35,50+i*60,650),fill='blue')
        with self.assertRaises(HTTPException) as error: analizar(self.request(image))
        self.assertEqual(error.exception.status_code,422)
    def test_blank_photo_is_rejected(self):
        with self.assertRaises(HTTPException): analizar(self.request(Image.new('RGB',(480,720),'white')))
    @patch('app.cv2.CascadeClassifier')
    def test_single_false_positive_without_agreement_is_rejected(self, classifiers):
        first=Mock(); second=Mock()
        first.detectMultiScale.return_value=np.array([[50,50,160,160]])
        second.detectMultiScale.return_value=[]
        classifiers.side_effect=[first,second]
        with self.assertRaisesRegex(ValueError,'confirmar'): face_geometry(np.zeros((400,400,3),np.uint8))
    @patch('app.cv2.CascadeClassifier')
    def test_multiple_people_are_rejected(self, classifiers):
        detector=Mock(); detector.detectMultiScale.return_value=np.array([[20,20,100,100],[200,20,100,100]])
        classifiers.return_value=detector
        with self.assertRaisesRegex(ValueError,'sola persona'): face_geometry(np.zeros((400,400,3),np.uint8))

if __name__ == '__main__': unittest.main()
