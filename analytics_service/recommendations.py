from __future__ import annotations

CATALOG = {
    "ovalado": [("MID_FADE", .94), ("TAPER", .90), ("FADE_MODERNO", .86)],
    "redondo": [("MID_FADE", .93), ("FADE_MODERNO", .89), ("TAPER", .82)],
    "cuadrado": [("LOW_FADE", .94), ("BUZZ", .88), ("TAPER", .84)],
    "alargado": [("LOW_FADE", .91), ("TAPER", .87), ("BUZZ", .80)],
}


def recommend(face_shape: str, density: str):
    rows = CATALOG.get(face_shape, CATALOG["ovalado"])
    adjustment = -.04 if density == "baja" else .0
    cuts = [
        {"nombre": name, "score": round(max(.75, score + adjustment), 2), "riesgo": "bajo" if score >= .88 else "medio"}
        for name, score in rows
    ]
    tints = ["Negro natural", "Castano oscuro", "Castano medio"]
    return cuts, tints
