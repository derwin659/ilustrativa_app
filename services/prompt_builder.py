BASE_PROMPT = (
    "ultra realistic photo of the same person, "
    "real human skin texture, "
    "natural lighting, "
    "only change the hairstyle"
)


TINTE_PROMPTS = {
    "platino": (
        "platinum blonde hair, "
        "very light blonde hair, "
        "bleached platinum hair, "
        "uniform platinum tone, "
        "no dark roots"
    ),
    "marrón medio": (
        "medium brown hair, "
        "natural brown tone, "
        "even brown hair color"
    )
}


CORTE_PROMPTS = {
    "FADE_MODERNO": (
        "low fade haircut, "
        "clean low fade on sides, "
        "smooth fade transition, "
        "sharp lineup, "
        "modern barbershop style"
    ),
    "TAPER": "taper fade haircut, clean neckline",
    "BUZZ": "buzz cut haircut, uniform hair length",
    "LOW_FADE": (
    "professional low skin fade haircut, "
    "fade starts low near the ear, "
    "short fade transition, "
    "clean low fade"
    ),
    "MID_FADE": (
    "professional barber mid fade haircut, "
    "smooth gradual fade from skin to hair, "
    "clear skin fade at the bottom, "
    "natural hair density on top, "
    "real barbershop fade"
    )
}



ONDULADO_PROMPTS = {
    "ondulado_suave": (
        "soft wavy hair texture, "
        "subtle natural waves, "
        "not curly, not afro"
    ),
    "ondulado_marcado": "defined wave perm texture"
}



VISTA_PROMPTS = {
    "frontal": "front view, face clearly visible",
    "lateral": "side profile view, fade clearly visible",
    "atras": "back view of the head, nape fade visible"
}

NEGATIVE_ZERO = (
    "hair, stubble, beard, mustache, five o'clock shadow, "
    "dark dots, hair roots, hair follicles, "
    "gray shadow, dark patch, "
    "purple, violet, magenta, "
    "texture, noise, grain"
)

NEGATIVE_PROMPTL = (
    "hair design, shaved lines, patterns, symbols, artwork, "
    "zig zag, lightning, stripes, drawings, "
    "beard, mustache, stubble, "
    "face deformation, facial distortion, "
    "plastic skin, wax skin, ai face, "
    "buzz cut, crop cut, undercut, mohawk, "
    "hard part line, shaved part "
    "purple color, magenta, artifacts, color banding, noise blocks, corrupted texture"

)

NEGATIVE_PROMPT = (
    "beard, mustache, stubble, facial hair, "
    "five o'clock shadow, jaw shadow, "
    "face deformation, facial distortion, "
    "changed facial structure, "
    "altered jaw, altered cheeks, "
    "plastic skin, wax skin, ai face, "
    "face retouching, facial smoothing, "
    "modified eyes, modified nose, modified lips, "
    "beard, facial hair, sideburns, jawline beard, "
    "goatee, stubble on face, fake beard, "
    "painted beard, dark patch on cheek, "
    "purple color, violet color, magenta, neon, "
    "background artifacts, color noise, paint texture, "
    "blurred background, jpeg artifacts, watermark, "
    "deformed head, extra face, extra ear, "
    "neck distortion, melted skin"

)

NEGATIVE_PROMPTG = (
    # ❌ Cortes mal hechos
    "thick sides, no soft fade, no blurry fade, no weak fade, "

    # ❌ Barba / vello facial
    "beard, mustache, stubble, facial hair, "
    "five o'clock shadow, jaw shadow, "

    # ❌ Deformaciones de rostro
    "face deformation, facial deformation, "
    "changed facial structure, "
    "wide face, fat face, puffy cheeks, "
    "distorted jaw, altered jawline, "
    "altered cheekbones, "

    # ❌ Orejas
    "altered ear, ear deformation, "

    # ❌ Piel falsa / IA
    "face retouching, facial smoothing, "
    "plastic skin, wax skin, ai face, "
    "skin distortion, dark skin patches, "
    "fake shadows on face, "

    # ❌ Ojos / nariz / boca
    "modified eyes, modified nose, modified lips"
)




def build_prompt(req, vista: str) -> str:
    parts = [BASE_PROMPT]

    # 🔹 Corte (USAR TIPO, NO NOMBRE)
    corte_prompt = CORTE_PROMPTS.get(req.corte.tipo)
    if corte_prompt:
        parts.append(corte_prompt)

    # 🔹 Tinte
    if req.tinte and req.tinte.aplicar:
        tinte_prompt = TINTE_PROMPTS.get(req.tinte.color.lower())
        if tinte_prompt:
            parts.append(tinte_prompt)

    # 🔹 Ondulado
    if req.ondulado and req.ondulado.aplicar:
        ondulado_prompt = ONDULADO_PROMPTS.get(req.ondulado.tipo)
        if ondulado_prompt:
            parts.append(ondulado_prompt)

    # 🔹 Vista
    parts.append(VISTA_PROMPTS.get(vista, vista))

    return ", ".join(parts)

