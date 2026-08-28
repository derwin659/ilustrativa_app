import base64
import unittest
from io import BytesIO

from PIL import Image

from app import decode_image
from recommendations import recommend


class AnalyticsContractTests(unittest.TestCase):
    def test_recommendations_use_generative_cut_codes(self):
        cuts, _ = recommend("ovalado", "media")
        self.assertEqual("MID_FADE", cuts[0]["nombre"])
        self.assertGreaterEqual(cuts[0]["score"], .75)

    def test_decodes_valid_image(self):
        image = Image.new("RGB", (300, 300), "white")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        decoded = decode_image(base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual((300, 300, 3), decoded.shape)

    def test_rejects_invalid_base64(self):
        with self.assertRaisesRegex(ValueError, "Base64"):
            decode_image("not-valid-base64")


if __name__ == "__main__":
    unittest.main()
