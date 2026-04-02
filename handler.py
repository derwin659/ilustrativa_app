import runpod
from schemas_runpod import RequestGenerar


def handler(job):
    job_input = job.get("input", {})

    try:
        req = RequestGenerar(**job_input)
    except Exception as e:
        return {
            "success": False,
            "message": f"Payload inválido: {str(e)}"
        }

    if "frontal" not in req.imagenes:
        return {
            "success": False,
            "message": "Imagen frontal requerida"
        }

    return {
        "success": True,
        "message": "Runpod Serverless funcionando",
        "sesionId": req.sesionId,
        "imagenes_recibidas": list(req.imagenes.keys()),
        "corte": {
            "nombre": req.corte.nombre,
            "tipo": req.corte.tipo
        },
        "vistas": req.vistas
    }


runpod.serverless.start({"handler": handler})