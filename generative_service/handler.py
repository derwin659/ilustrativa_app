import logging

import runpod

from schemas_runpod import RequestGenerar
from services.image_transport import delete_output_assets, upload_output_image

logger = logging.getLogger(__name__)


def handler(job):
    job_input = job.get("input", {})

    try:
        req = RequestGenerar(**job_input)
    except Exception as error:
        return {"success": False, "message": f"Payload inválido: {error}"}

    if req.storage is None:
        return {"success": False, "message": "storage.tenantId es obligatorio en Serverless"}

    assets = {}
    try:
        from services.generation_runner import run_generation

        def store(image, view):
            uploaded = upload_output_image(image, req.storage.tenantId, req.sesionId, view)
            assets[view] = uploaded["publicId"]
            return uploaded["url"]

        result = run_generation(
            req=req,
            base_url=None,
            save_files=False,
            output_uploader=store,
        )
        return {
            "success": True,
            "message": "Generación completada",
            **result,
            "assets": assets,
        }
    except Exception as error:
        delete_output_assets(assets.values())
        logger.exception("Error interno en el worker para la sesión %s", req.sesionId)
        return {"success": False, "message": f"Error interno en worker: {error}"}


if __name__ == "__main__":
    from worker_startup import initialize_worker
    initialize_worker()
    runpod.serverless.start({"handler": handler})