"""Catalogo versionado de estilos y servicios para el motor analitico."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Haircut:
    code: str
    display_name: str
    face_shapes: tuple[str, ...]
    textures: tuple[str, ...]
    min_length: str
    densities: tuple[str, ...]
    maintenance: str
    generative_ready: bool = False


HAIRCUTS = (
    Haircut("MID_FADE", "Mid Fade", ("redondo", "ovalado", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("baja", "media", "alta"), "media", True),
    Haircut("LOW_FADE", "Low Fade", ("ovalado", "cuadrado", "alargado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("baja", "media", "alta"), "media", True),
    Haircut("HIGH_FADE", "High Fade", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("media", "alta"), "alta"),
    Haircut("SKIN_FADE", "Skin Fade", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("media", "alta"), "alta"),
    Haircut("DROP_FADE", "Drop Fade", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("media", "alta"), "alta"),
    Haircut("BURST_FADE", "Burst Fade", ("ovalado", "redondo", "cuadrado"), ("ondulado", "rizado", "afro"), "medio", ("media", "alta"), "alta"),
    Haircut("FADE_MODERNO", "Fade moderno", ("ovalado", "redondo"), ("lacio", "ondulado", "rizado"), "corto", ("media", "alta"), "media", True),
    Haircut("TAPER", "Taper clásico", ("ovalado", "redondo", "cuadrado", "alargado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("baja", "media", "alta"), "baja", True),
    Haircut("TAPER_FADE", "Taper Fade", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "corto", ("media", "alta"), "media"),
    Haircut("BUZZ", "Buzz Cut", ("ovalado", "cuadrado", "alargado"), ("lacio", "ondulado", "rizado", "afro"), "rapado", ("baja", "media", "alta"), "baja", True),
    Haircut("CREW_CUT", "Crew Cut", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado"), "corto", ("media", "alta"), "baja"),
    Haircut("FRENCH_CROP", "French Crop", ("ovalado", "alargado", "cuadrado"), ("lacio", "ondulado"), "corto", ("media", "alta"), "media"),
    Haircut("CAESAR", "Corte César", ("ovalado", "alargado", "cuadrado"), ("lacio", "ondulado"), "corto", ("baja", "media", "alta"), "baja"),
    Haircut("QUIFF", "Quiff", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado"), "medio", ("media", "alta"), "alta"),
    Haircut("POMPADOUR", "Pompadour", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado"), "medio", ("media", "alta"), "alta"),
    Haircut("SLICK_BACK", "Slick Back", ("ovalado", "cuadrado"), ("lacio", "ondulado"), "medio", ("media", "alta"), "media"),
    Haircut("SIDE_PART", "Raya lateral", ("ovalado", "redondo", "cuadrado", "alargado"), ("lacio", "ondulado"), "medio", ("baja", "media", "alta"), "media"),
    Haircut("COMB_OVER", "Comb Over", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado"), "medio", ("baja", "media", "alta"), "media"),
    Haircut("TEXTURED_CROP", "Crop texturizado", ("ovalado", "alargado", "cuadrado"), ("lacio", "ondulado", "rizado"), "corto", ("media", "alta"), "media"),
    Haircut("CURLY_TOP", "Curly Top", ("ovalado", "redondo", "cuadrado"), ("rizado", "afro"), "medio", ("media", "alta"), "media"),
    Haircut("AFRO_TAPER", "Afro Taper", ("ovalado", "redondo", "cuadrado", "alargado"), ("afro",), "medio", ("media", "alta"), "media"),
    Haircut("MULLET", "Mullet", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado"), "medio", ("media", "alta"), "alta"),
    Haircut("MOHAWK", "Mohawk", ("ovalado", "redondo", "cuadrado"), ("lacio", "ondulado", "rizado", "afro"), "medio", ("media", "alta"), "alta"),
    Haircut("BRO_FLOW", "Bro Flow", ("ovalado", "cuadrado", "alargado"), ("lacio", "ondulado", "rizado"), "largo", ("media", "alta"), "media"),
    Haircut("CURTAIN", "Curtain Hair", ("ovalado", "cuadrado", "alargado"), ("lacio", "ondulado"), "medio", ("media", "alta"), "media"),
    Haircut("MAN_BUN", "Man Bun", ("ovalado", "cuadrado"), ("lacio", "ondulado", "rizado"), "largo", ("media", "alta"), "media"),
)


TINT_SERVICES = (
    {"code": "NATURAL_DARK", "name": "Oscurecimiento natural", "requires_professional_review": True},
    {"code": "GRAY_BLEND", "name": "Camuflaje de canas", "requires_professional_review": True},
    {"code": "FASHION_COLOR", "name": "Color creativo", "requires_professional_review": True},
    {"code": "HIGHLIGHTS", "name": "Mechas o reflejos", "requires_professional_review": True},
)
