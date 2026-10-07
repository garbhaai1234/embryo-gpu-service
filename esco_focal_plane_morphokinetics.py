import asyncio
import os
from typing import Callable, List

import httpx
from fastapi import Depends, HTTPException, Request, status
from PIL import Image


VALID_STAGES = ["8 Cells", "Morula", "Blastocyst"]
STAGE_DAY = {"8 Cells": "Day 3", "Morula": "Day 4", "Blastocyst": "Day 5"}
GRADE_PRIORITY = {"A": 3, "B": 2, "C": 1}
DEFAULT_FOCAL_PLANES = [1, 2, 3, 4, 5, 6, 7]


def _stage_and_grade(class_name: str):
    if class_name == "Error Image":
        return None, None
    if "8 Cells" in class_name:
        return "8 Cells", class_name.split()[-1]
    if "Morula" in class_name:
        return "Morula", class_name.split()[-1]
    if "Blastocyst" in class_name:
        return "Blastocyst", class_name.split()[-1]
    return None, None


def _sort_key(item: dict):
    return (float(item.get("time", 0)), int(item.get("index", 0)), str(item.get("z", "")))


def _pick_best_frames(classified: list[dict]) -> dict:
    """Pick one best frame per stage across one or many focal planes.

    This mirrors timelapse.py's grade/confidence preference, but uses time
    ordering instead of raw frame index so candidates from multiple focal
    planes can be compared fairly.
    """
    best = {}
    last_time = float("-inf")
    for stage in VALID_STAGES:
        stage_items = [
            item for item in classified
            if item["stage"] == stage and float(item["time"]) > last_time
        ]
        if not stage_items:
            continue
        stage_items.sort(
            key=lambda item: (
                GRADE_PRIORITY.get(item["grade"], 0),
                float(item["confidence"]),
            ),
            reverse=True,
        )
        chosen = stage_items[0]
        best[stage] = chosen
        last_time = float(chosen["time"])
    return best


def _pick_best_frames_across_planes(classified: list[dict]) -> dict:
    """Pick the best Day 3, Day 4, and Day 5 frame across all focal planes.

    Unlike the per-plane/timelapse selector, this is intentionally independent
    per stage: the UI wants the best frame available for each day across z
    planes, even if another stage is missing or partially processed.
    """
    best = {}
    for stage in VALID_STAGES:
        stage_items = [item for item in classified if item["stage"] == stage]
        if not stage_items:
            continue
        stage_items.sort(
            key=lambda item: (
                GRADE_PRIORITY.get(item["grade"], 0),
                float(item["confidence"]),
            ),
            reverse=True,
        )
        best[stage] = stage_items[0]
    return best


def _calculate_morphokinetics(classified: list[dict]) -> dict:
    kinetics = {"8 Cells": None, "Morula": None, "Blastocyst": None}
    for item in sorted(classified, key=_sort_key):
        stage = item["stage"]
        time_value = float(item["time"])
        if kinetics[stage] is not None:
            continue
        if stage == "Morula" and kinetics["8 Cells"] is None:
            continue
        if stage == "Blastocyst" and kinetics["Morula"] is None:
            continue
        kinetics[stage] = time_value

    t8 = kinetics["8 Cells"]
    tM = kinetics["Morula"]
    tB = kinetics["Blastocyst"]
    return {
        "t8": t8,
        "tM": tM,
        "tB": tB,
        "d_8_M": (tM - t8) if t8 is not None and tM is not None else None,
        "d_M_B": (tB - tM) if tM is not None and tB is not None else None,
        "total_dev_time": (tB - t8) if t8 is not None and tB is not None else None,
    }


def _candidate_time(candidate: dict) -> float:
    for key in ("time", "elapsed_time", "elapsed_time_minutes", "hpi"):
        if candidate.get(key) is not None:
            return float(candidate[key])
    return float(candidate.get("index", 0))


def _candidate_image_url(candidate: dict) -> str:
    image_url = candidate.get("image_url") or candidate.get("url")
    if not image_url:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="ESCO candidate response is missing image_url.",
        )
    return str(image_url)


def _without_image(item: dict) -> dict:
    out = dict(item)
    out.pop("_image", None)
    return out


def _coverage_score(result: dict) -> int:
    kinetics = result.get("morphokinetics") or {}
    complete = sum(1 for key in ("t8", "tM", "tB") if kinetics.get(key) is not None)
    best_count = len(result.get("best_frames") or {})
    return complete * 10 + best_count


def _average_best_confidence(result: dict) -> float:
    frames = (result.get("best_frames") or {}).values()
    confidences = [float(frame.get("confidence", 0)) for frame in frames]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _choose_recommended_plane(results_by_plane: dict) -> str | None:
    successful = [
        (str(z), result) for z, result in results_by_plane.items()
        if result.get("success")
    ]
    if not successful:
        return None
    successful.sort(
        key=lambda pair: (
            _coverage_score(pair[1]),
            _average_best_confidence(pair[1]),
            int(pair[1].get("classified_count") or 0),
        ),
        reverse=True,
    )
    return successful[0][0]


def register_esco_focal_plane_morphokinetics(
    *,
    app,
    require_api_key,
    class_names: List[str],
    decode_image: Callable[[bytes], Image.Image],
    classify_batch,
    morphology_batch,
    classifier_transform,
    torch_device,
    classifier,
    gradcam_cls,
    overlay_grad_cam,
    pil_to_jpeg_b64,
    esco_bridge_base_url: str = "",
    esco_bridge_api_key: str = "",
):
    bridge_base_url = (
        os.environ.get("ESCO_BRIDGE_BASE_URL")
        or esco_bridge_base_url
        or ""
    ).rstrip("/")
    bridge_api_key = os.environ.get("ESCO_BRIDGE_API_KEY") or esco_bridge_api_key or ""
    fetch_concurrency = int(os.environ.get("ESCO_FETCH_CONCURRENCY", "24"))
    fetch_timeout = float(os.environ.get("ESCO_FETCH_TIMEOUT", "240"))
    default_parallel_planes = int(os.environ.get("ESCO_PARALLEL_PLANES", "2"))

    def bridge_headers(accept: str = "application/json") -> dict:
        headers = {"Accept": accept}
        if bridge_api_key:
            headers["X-API-Key"] = bridge_api_key
        return headers

    def bridge_url(path_or_url: str) -> str:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            return path_or_url
        if not bridge_base_url:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ESCO_BRIDGE_BASE_URL is not configured on GPU service.",
            )
        return f"{bridge_base_url}{path_or_url}"

    async def fetch_bridge_json(client: httpx.AsyncClient, path: str, params: dict) -> dict:
        response = await client.get(
            bridge_url(path),
            headers=bridge_headers(),
            params=params,
        )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "ESCO bridge candidate endpoint failed.",
                    "bridge_status": response.status_code,
                    "bridge_response": response.text[:1000],
                },
            )
        return response.json()

    async def fetch_candidate_image(
        client: httpx.AsyncClient,
        image_url: str,
        semaphore: asyncio.Semaphore,
    ) -> Image.Image:
        async with semaphore:
            response = await client.get(
                bridge_url(image_url),
                # Avoid AVIF: the deployed PIL/GPU stack has failed to decode it.
                headers=bridge_headers("image/jpeg,image/png,image/webp,*/*;q=0.5"),
            )
        if response.status_code >= 400:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={
                    "message": "Failed to fetch ESCO candidate image.",
                    "image_url": image_url,
                    "bridge_status": response.status_code,
                },
            )
        return decode_image(response.content)

    def build_candidate_params(body: dict, z: int | str) -> dict:
        params = {
            "timelapse_number": body["timelapse_number"],
            "dish_position_id": body["dish_position_id"],
            "z": z,
            "max_size": body.get("max_size", 384),
            "diff_threshold": body.get("diff_threshold", 3.0),
            "force_every": body.get("force_every", 40),
        }
        if body.get("max_concurrency") is not None:
            params["max_concurrency"] = body.get("max_concurrency")
        return params

    def build_classified(
        *,
        z: str,
        candidates: list[dict],
        images: list[Image.Image],
        pred_idx: list[int],
        confidences: list[float],
    ) -> list[dict]:
        classified = []
        for i, (candidate, image, class_idx, confidence) in enumerate(
            zip(candidates, images, pred_idx, confidences)
        ):
            class_name = class_names[int(class_idx)]
            stage, grade = _stage_and_grade(class_name)
            if stage is None:
                continue
            classified.append({
                "index": int(candidate.get("index", i)),
                "time": _candidate_time(candidate),
                "z": z,
                "stage": stage,
                "grade": grade,
                "class_idx": int(class_idx),
                "class_name": class_name,
                "confidence": float(confidence),
                "image_url": candidate.get("image_url"),
                "diff_score": candidate.get("diff_score"),
                "reason": candidate.get("reason"),
                "_image": image,
            })
        return classified

    def build_best_frames(classified: list[dict], include_gradcam: bool, *, across_planes: bool = False) -> dict:
        best = _pick_best_frames_across_planes(classified) if across_planes else _pick_best_frames_across_planes(classified)
        best_frames = {}
        gradcam = None
        try:
            if include_gradcam and best:
                gradcam = gradcam_cls(classifier, classifier.features[-1])

            for stage, item in best.items():
                image = item["_image"]
                morphology = None
                if stage == "Blastocyst":
                    morph_list = morphology_batch([image])
                    morphology = morph_list[0] if morph_list else None

                gradcam_b64 = None
                if include_gradcam and gradcam is not None:
                    tensor = classifier_transform(image).unsqueeze(0).to(torch_device)
                    cam = gradcam.generate_cam(tensor, item["class_idx"])
                    overlay = overlay_grad_cam(image, cam)
                    gradcam_b64 = pil_to_jpeg_b64(overlay)

                payload = {
                    "stage": stage,
                    "day": STAGE_DAY[stage],
                    "index": item["index"],
                    "time": item["time"],
                    "z": item["z"],
                    "class_idx": item["class_idx"],
                    "class_name": item["class_name"],
                    "grade": item["grade"],
                    "confidence": item["confidence"],
                    "image_url": item["image_url"],
                    "diff_score": item["diff_score"],
                    "reason": item["reason"],
                }
                if morphology:
                    payload["gardner"] = morphology["short_code"]
                    payload["morphology"] = morphology
                if gradcam_b64:
                    payload["gradcam_image"] = gradcam_b64
                best_frames[STAGE_DAY[stage]] = payload
        finally:
            if gradcam is not None:
                gradcam.close()
        return best_frames

    async def analyze_plane(body: dict, z: int | str, *, include_images: bool = False) -> dict:
        patient_id = str(body["patient_id"])
        embryo_id = str(body["embryo_id"])
        z_str = str(z)
        candidate_path = f"/patients/{patient_id}/embryos/{embryo_id}/focal-plane-candidates"

        async with httpx.AsyncClient(timeout=httpx.Timeout(fetch_timeout)) as client:
            candidate_payload = await fetch_bridge_json(
                client,
                candidate_path,
                build_candidate_params(body, z),
            )
            candidates = candidate_payload.get("candidates") or []
            if not candidates:
                return {
                    "success": False,
                    "message": "No ESCO candidate frames returned.",
                    "z": z_str,
                    "candidate_payload": candidate_payload,
                }

            semaphore = asyncio.Semaphore(fetch_concurrency)
            images = await asyncio.gather(*[
                fetch_candidate_image(client, _candidate_image_url(item), semaphore)
                for item in candidates
            ])

        _tensors, pred_idx, confidences = classify_batch(images)
        classified = build_classified(
            z=z_str,
            candidates=candidates,
            images=images,
            pred_idx=pred_idx,
            confidences=confidences,
        )
        if not classified:
            return {
                "success": False,
                "message": "Candidates were fetched, but none classified as 8 Cells, Morula, or Blastocyst.",
                "z": z_str,
                "total_frames": candidate_payload.get("total_frames"),
                "processed_frames": candidate_payload.get("processed_frames"),
                "candidate_count": len(candidates),
            }

        morphokinetics = _calculate_morphokinetics(classified)
        best_frames = build_best_frames(classified, bool(body.get("include_gradcam", False)))
        response = {
            "success": True,
            "z": z_str,
            "total_frames": candidate_payload.get("total_frames"),
            "processed_frames": candidate_payload.get("processed_frames"),
            "failed_frames": candidate_payload.get("failed_frames"),
            "partial": candidate_payload.get("partial", False),
            "candidate_count": len(candidates),
            "classified_count": len(classified),
            "morphokinetics": morphokinetics,
            "best_frames": best_frames,
            "classified_frames": [_without_image(item) for item in classified],
        }
        if include_images:
            response["_classified_with_images"] = classified
        return response

    @app.post("/esco/focal-plane-morphokinetics", dependencies=[Depends(require_api_key)])
    async def esco_focal_plane_morphokinetics(request: Request):
        body = await request.json()
        required_fields = (
            "patient_id",
            "embryo_id",
            "timelapse_number",
            "dish_position_id",
            "z",
        )
        missing = [field for field in required_fields if body.get(field) in (None, "")]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required field(s): {', '.join(missing)}",
            )

        z = body["z"]
        result = await analyze_plane(body, z)
        result.update({
            "patient_id": str(body["patient_id"]),
            "embryo_id": str(body["embryo_id"]),
            "timelapse_number": str(body["timelapse_number"]),
            "dish_position_id": str(body["dish_position_id"]),
        })
        return result

    @app.post("/esco/all-focal-planes-morphokinetics", dependencies=[Depends(require_api_key)])
    async def esco_all_focal_planes_morphokinetics(request: Request):
        body = await request.json()
        required_fields = (
            "patient_id",
            "embryo_id",
            "timelapse_number",
            "dish_position_id",
        )
        missing = [field for field in required_fields if body.get(field) in (None, "")]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required field(s): {', '.join(missing)}",
            )

        focal_planes = body.get("focal_planes") or DEFAULT_FOCAL_PLANES
        focal_planes = [str(z) for z in focal_planes]
        max_parallel_planes = int(body.get("max_parallel_planes") or default_parallel_planes)
        max_parallel_planes = max(1, min(max_parallel_planes, len(focal_planes)))
        plane_semaphore = asyncio.Semaphore(max_parallel_planes)

        async def run_plane(z: str):
            async with plane_semaphore:
                try:
                    return z, await analyze_plane(body, z, include_images=True)
                except HTTPException as exc:
                    return z, {
                        "success": False,
                        "z": z,
                        "error": exc.detail,
                        "status_code": exc.status_code,
                    }
                except Exception as exc:
                    return z, {
                        "success": False,
                        "z": z,
                        "error": str(exc),
                    }

        plane_pairs = await asyncio.gather(*[run_plane(z) for z in focal_planes])
        results_by_plane = {z: result for z, result in plane_pairs}

        all_classified_with_images = []
        for result in results_by_plane.values():
            all_classified_with_images.extend(result.pop("_classified_with_images", []))

        cross_plane_morphokinetics = _calculate_morphokinetics(all_classified_with_images)
        cross_plane_best_frames = build_best_frames(
            all_classified_with_images,
            bool(body.get("include_gradcam", False)),
            across_planes=True,
        ) if all_classified_with_images else {}
        recommended_plane = _choose_recommended_plane(results_by_plane)

        return {
            "success": any(result.get("success") for result in results_by_plane.values()),
            "patient_id": str(body["patient_id"]),
            "embryo_id": str(body["embryo_id"]),
            "timelapse_number": str(body["timelapse_number"]),
            "dish_position_id": str(body["dish_position_id"]),
            "focal_planes": focal_planes,
            "max_parallel_planes": max_parallel_planes,
            "progress": {
                "planes_total": len(focal_planes),
                "planes_completed": len(results_by_plane),
                "planes_failed": sum(1 for r in results_by_plane.values() if not r.get("success")),
            },
            "totals": {
                "total_frames": sum(int(r.get("total_frames") or 0) for r in results_by_plane.values()),
                "processed_frames": sum(int(r.get("processed_frames") or 0) for r in results_by_plane.values()),
                "failed_frames": sum(int(r.get("failed_frames") or 0) for r in results_by_plane.values()),
                "candidate_count": sum(int(r.get("candidate_count") or 0) for r in results_by_plane.values()),
                "classified_count": sum(int(r.get("classified_count") or 0) for r in results_by_plane.values()),
            },
            "cross_plane_morphokinetics": cross_plane_morphokinetics,
            "cross_plane_best_frames": cross_plane_best_frames,
            "recommended_plane": recommended_plane,
            "recommended_result": results_by_plane.get(recommended_plane) if recommended_plane else None,
            "results_by_plane": results_by_plane,
        }
