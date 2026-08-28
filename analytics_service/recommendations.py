from __future__ import annotations

from catalog import HAIRCUTS, TINT_SERVICES, Haircut

LENGTH_ORDER = {"rapado": 0, "corto": 1, "medio": 2, "largo": 3}


def _affinity(value: str, options: tuple[str, ...], maximum: float) -> float:
    if value not in options:
        return 0.0
    position = options.index(value)
    position_factor = max(.45, 1 - position * .18)
    specificity_factor = 1 + max(0, 4 - len(options)) * .035
    return min(maximum, maximum * position_factor * specificity_factor)


def _score(style: Haircut, face_shape: str, density: str, texture: str, length: str) -> tuple[float, list[str]]:
    score = .18
    reasons: list[str] = []
    if face_shape in style.face_shapes:
        score += _affinity(face_shape, style.face_shapes, .30)
        reasons.append(f"equilibra un rostro {face_shape}")
    if texture in style.textures:
        score += _affinity(texture, style.textures, .18)
        reasons.append(f"funciona con cabello {texture}")
    if density in style.densities:
        score += _affinity(density, style.densities, .10)
        reasons.append(f"es compatible con densidad {density}")

    available = LENGTH_ORDER.get(length, 1)
    required = LENGTH_ORDER.get(style.min_length, 1)
    if available >= required:
        score += .12 if available == required else .09
        reasons.append("el largo actual permite trabajarlo")
    else:
        score -= .18 * (required - available)
        reasons.append(f"requiere dejar crecer hasta largo {style.min_length}")
    return round(min(.96, max(.35, score)), 2), reasons

def recommend_detailed(face_shape: str, density: str, texture: str = "lacio", length: str = "corto", limit: int = 8):
    ranked = []
    for style in HAIRCUTS:
        score, reasons = _score(style, face_shape, density, texture, length)
        ranked.append({
            "nombre": style.code,
            "nombre_visible": style.display_name,
            "score": score,
            "riesgo": "bajo" if score >= .82 else "medio" if score >= .68 else "alto",
            "razones": reasons,
            "mantenimiento": style.maintenance,
            "largo_minimo": style.min_length,
            "vista_generativa_disponible": style.generative_ready,
        })
    ranked.sort(key=lambda item: (item["score"], item["vista_generativa_disponible"]), reverse=True)
    return ranked[:limit], [dict(service) for service in TINT_SERVICES]


def recommend(face_shape: str, density: str):
    """Contrato v1 conservado para backend y pruebas existentes."""
    cuts, tint_services = recommend_detailed(face_shape, density, limit=3)
    return cuts, [service["name"] for service in tint_services[:3]]