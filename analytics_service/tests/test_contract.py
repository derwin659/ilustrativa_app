import base64
import unittest
from io import BytesIO

from PIL import Image

from app import decode_image
from catalog import HAIRCUTS, TINT_SERVICES
from recommendations import recommend, recommend_detailed


class AnalyticsContractTests(unittest.TestCase):
    def test_recommendations_use_generative_cut_codes(self):
        cuts, _ = recommend("ovalado", "media")
        self.assertIn(cuts[0]["nombre"], {"MID_FADE", "LOW_FADE", "FADE_MODERNO", "TAPER", "BUZZ"})
        self.assertTrue(cuts[0]["vista_generativa_disponible"])
        self.assertGreaterEqual(cuts[0]["score"], .70)

    def test_catalog_has_meaningful_initial_coverage(self):
        self.assertGreaterEqual(len(HAIRCUTS), 25)
        self.assertGreaterEqual(len(TINT_SERVICES), 4)
        self.assertEqual(len({style.code for style in HAIRCUTS}), len(HAIRCUTS))

    def test_detailed_recommendations_are_explainable(self):
        cuts, services = recommend_detailed("redondo", "alta", "rizado", "medio")
        self.assertGreaterEqual(len(cuts), 5)
        self.assertTrue(cuts[0]["razones"])
        self.assertIn("nombre_visible", cuts[0])
        self.assertTrue(all(item["requires_professional_review"] for item in services))

    def test_top_scores_are_not_saturated_or_tied(self):
        cuts, _ = recommend_detailed("redondo", "baja", "lacio", "corto")
        top_scores = [item["score"] for item in cuts[:3]]
        self.assertEqual(3, len(set(top_scores)))
        self.assertLessEqual(top_scores[0], .95)
        self.assertGreater(top_scores[0], top_scores[1])

    def test_long_styles_are_penalized_when_hair_is_short(self):
        short_hair, _ = recommend_detailed("ovalado", "media", "lacio", "corto", limit=30)
        long_hair, _ = recommend_detailed("ovalado", "media", "lacio", "largo", limit=30)
        short_score = next(item["score"] for item in short_hair if item["nombre"] == "MAN_BUN")
        long_score = next(item["score"] for item in long_hair if item["nombre"] == "MAN_BUN")
        self.assertGreater(long_score, short_score)

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
