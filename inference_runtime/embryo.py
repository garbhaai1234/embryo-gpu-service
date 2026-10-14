"""Extra inference endpoints shared by GPU and CPU deployments."""
import asyncio
import base64
from io import BytesIO
import os
import re
import tempfile
from functools import lru_cache

from fastapi import Depends, File, Form, HTTPException, UploadFile


def extend(service):
    from PIL import Image
    import torch
    from inference_runtime import segmentation

    # Serialize model access, including GradCAM hooks on shared classifiers.
    gate = asyncio.Lock()

    @service.app.middleware("http")
    async def bound_inference(request, call_next):
        if request.method != "POST":
            return await call_next(request)
        if gate.locked():
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Inference capacity busy"}, status_code=503)
        async with gate:
            return await call_next(request)

    def decode(raw, square=False):
        image = Image.open(BytesIO(raw)).convert("RGB")
        if square:
            w, h = image.size
            side = min(w, h)
            image = image.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
        return image

    def grade(raw, gradcam):
        image = decode(raw, square=True)
        tensor = service._classifier_transform(image).unsqueeze(0).to(service._torch_device)
        with torch.no_grad():
            confidence, prediction = torch.softmax(service._classifier(tensor), dim=1).max(dim=1)
        index = prediction.item()
        result = {"ai_grade": service.class_names[index], "confidence": confidence.item()}
        if gradcam:
            cam = service.GradCAM(service._classifier, service._classifier.features[-1])
            try:
                heatmap = cam.generate_cam(tensor, index)
                overlay = service._overlay_grad_cam(image, heatmap, alpha=0.4)
                out = BytesIO()
                overlay.save(out, format="PNG")
                result["gradcam_image"] = base64.b64encode(out.getvalue()).decode()
            finally:
                cam.close()
        return result

    auth = [Depends(service.require_api_key)]

    @lru_cache(maxsize=1)
    def summarizer():
        from transformers import pipeline
        path = os.environ.get("FEEDBACK_SUMMARIZER_MODEL_PATH")
        if not path:
            raise HTTPException(503, "Feedback summarizer is not configured")
        return pipeline("summarization", model=path,
                        device=0 if service._torch_device.type == "cuda" else -1)

    @service.app.post("/v1/summarize", dependencies=auth)
    async def summarize_endpoint(text: str = Form(...)):
        def run():
            result = summarizer()(text, max_length=50, min_length=10, do_sample=False)
            return {"summary": result[0]["summary_text"] if result else ""}
        return await asyncio.to_thread(run)

    @service.app.post("/v1/grade", dependencies=auth)
    async def grade_endpoint(file: UploadFile = File(...), gradcam: bool = Form(False)):
        return await asyncio.to_thread(grade, await file.read(), gradcam)

    def morphology(raw, profile):
        if service._gardners is None:
            raise HTTPException(503, "Morphology model unavailable")
        image = decode(raw, square=profile == "standard")
        result = service._morphology_batch([image])[0]
        if profile == "tracking":
            # Preserve the existing tracking endpoint's naming and scoring.
            result["exp"] = result.pop("expansion")
            result["morphology_score"] = round(sum(result[k] for k in (
                "expansion_score", "icm_score", "te_score")) / 3, 2)
        return {"morphology": result}

    @service.app.post("/v1/morphology", dependencies=auth)
    async def morphology_endpoint(file: UploadFile = File(...), profile: str = Form("standard")):
        if profile not in {"standard", "tracking"}:
            raise HTTPException(422, "Unknown morphology profile")
        return await asyncio.to_thread(morphology, await file.read(), profile)

    def explain(raw):
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                path = temp.name
                temp.write(raw)
            return {"explainability": segmentation.get_explainability_data(path)}
        finally:
            if path:
                os.unlink(path)

    @service.app.post("/v1/explainability", dependencies=auth)
    async def explain_endpoint(file: UploadFile = File(...)):
        return await asyncio.to_thread(explain, await file.read())

    @service.app.post("/v1/image-analysis", dependencies=auth)
    async def image_analysis_endpoint(file: UploadFile = File(...), stage: str = Form(...)):
        if not re.fullmatch(r"day[1-9][0-9]*", stage):
            raise HTTPException(422, "stage must be a positive day, for example day3 or day4")
        raw = await file.read()
        result = await asyncio.to_thread(grade, raw, False)
        metrics, masks, colors = {}, {}, {}
        morph = None
        if stage in {"day3", "day4"}:
            output = (await asyncio.to_thread(explain, raw))["explainability"]
            allowed = {"cells", "zona_pellucida"}
            if stage == "day4":
                allowed.update({"morula", "compaction", "fragmentation"})
            masks = dict(output.get("masks") or {})
            colors = dict(output.get("color_map") or {})
            if not masks.get("zona_pellucida") and masks.get("zona"):
                masks["zona_pellucida"] = masks["zona"]
            if "zona_pellucida" not in colors and "zona" in colors:
                colors["zona_pellucida"] = colors["zona"]
            masks = {key: value for key, value in masks.items() if key in allowed}
            colors = {key: value for key, value in colors.items() if key in allowed}
            metrics = dict(output.get("metrics") or {})
            if stage == "day3":
                metrics = {key: value for key, value in metrics.items() if key not in {
                    "fragmentation_percentage", "fragmentation_pct", "fragmentation_area_px",
                    "compaction_percentage", "compaction_pct", "compaction_area_px",
                }}
            morph = dict(metrics)
        elif stage == "day5":
            morph = (await asyncio.to_thread(morphology, raw, "standard"))["morphology"]
        return {
            "requested_stage": stage,
            "predicted_grade": result["ai_grade"],
            "confidence_score": result["confidence"],
            "morphology": morph,
            "metrics": metrics,
            "masks": masks,
            "color_map": colors,
        }

    @service.app.on_event("startup")
    def warm_segmentation():
        # Fail startup when a required checkpoint is missing, rather than
        # reporting a healthy full service with only some capabilities loaded.
        for path in (segmentation.PB_WEIGHTS, segmentation.MORULA_WEIGHTS,
                     segmentation.CELL_WEIGHTS, segmentation.FRAG_WEIGHTS,
                     segmentation.EMBRYO_WEIGHTS):
            segmentation._get_model(path)

    @service.app.get("/ready", dependencies=auth)
    def ready():
        if service._gardners is None:
            raise HTTPException(503, "Morphology model unavailable")
        return {"ready": True, "contract": 1, "device": str(service._torch_device)}
