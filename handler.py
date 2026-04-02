import runpod

from schemas_runpod import RequestGenerar
from services.generation_runner import run_generation


def handler(job):
    job_input = job.get("input", {})

    try:
        req = RequestGenerar(**job_input)
    except Exception as e:
        return {
            "success": False,
            "message": f"Payload inválido: {str(e)}"
        }

    try:
        result = run_generation(
            req=req,
            base_url=None,
            save_files=False
        )

        return {
            "success": True,
            "message": "Generación completada",
            **result
        }
    except ValueError as e:
        return {
            "success": False,
            "message": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error interno en worker: {str(e)}"
        }


runpod.serverless.start({"handler": handler})