import traceback
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

    try:
        from services.generation_runner import run_generation

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
    except Exception as e:
        return {
            "success": False,
            "message": f"Error interno en worker: {str(e)}",
            "traceback": traceback.format_exc()
        }


runpod.serverless.start({"handler": handler})