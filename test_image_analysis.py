"""Check deployment contracts without loading production checkpoints."""
import asyncio
import sys
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
import pytest


def test_new_embryo_contract_preserves_preprocessing_and_scores(monkeypatch):
    import torch
    from inference_runtime import embryo
    segmentation = SimpleNamespace(PB_WEIGHTS="pb", MORULA_WEIGHTS="m", CELL_WEIGHTS="c",
        FRAG_WEIGHTS="f", EMBRYO_WEIGHTS="e", _get_model=lambda _: None,
        get_explainability_data=lambda _: {
            "masks": {"zona_pellucida": "mask", "cells": "cells", "morula": "morula",
                      "compaction": "compaction", "fragmentation": "fragmentation", "embryo": "private"},
            "color_map": {"cells": [0, 255, 0], "morula": [255, 0, 0]},
            "metrics": {"cell_count": 8, "compaction_percentage": 60, "fragmentation_percentage": 12}})
    monkeypatch.setitem(sys.modules, "inference_runtime.segmentation", segmentation)
    seen = []
    def transform(image):
        seen.append(image.size)
        return torch.zeros(3, 4, 4)
    def morphology(images):
        seen.append(images[0].size)
        return [{"expansion": "1 (Morula Stage)", "icm": "Not Available", "te": "A (Good)",
                 "expansion_score": 4, "icm_score": 0, "te_score": 10,
                 "morphology_score": None, "short_code": "1NA"}]
    service = SimpleNamespace(app=FastAPI(), _torch_device=torch.device("cpu"),
        _classifier_transform=transform, _classifier=lambda _: torch.zeros(1, 10),
        class_names=["grade"] * 10, _gardners=object(), _morphology_batch=morphology,
        require_api_key=lambda: None)
    embryo.extend(service)
    out = BytesIO()
    Image.new("RGB", (30, 10)).save(out, format="PNG")
    files = {"file": ("image.png", out.getvalue(), "image/png")}
    with TestClient(service.app) as client:
        graded = client.post("/v1/grade", files=files)
        assert graded.status_code == 200
        assert graded.json()["confidence"] == pytest.approx(0.1)  # no legacy 0.7 floor
        assert seen[-1] == (10, 10)
        standard = client.post("/v1/morphology", files=files).json()["morphology"]
        assert standard["morphology_score"] is None
        assert seen[-1] == (10, 10)
        tracking = client.post("/v1/morphology", files=files, data={"profile": "tracking"}).json()["morphology"]
        assert seen[-1] == (30, 10)
        assert tracking["exp"] == "1 (Morula Stage)"
        assert tracking["morphology_score"] == 4.67
        assert "expansion" not in tracking
        assert client.post("/v1/explainability", files=files).json()["explainability"]["masks"]["zona_pellucida"] == "mask"
        day3 = client.post("/v1/image-analysis", files=files, data={"stage": "day3"})
        assert day3.status_code == 200
        assert set(day3.json()["masks"]) == {"cells", "zona_pellucida"}
        assert day3.json()["morphology"] == {"cell_count": 8}
        assert day3.json()["predicted_grade"] == "grade"
        day4 = client.post("/v1/image-analysis", files=files, data={"stage": "day4"})
        assert day4.status_code == 200
        assert set(day4.json()["masks"]) == {"cells", "zona_pellucida", "morula", "compaction", "fragmentation"}
        assert day4.json()["morphology"]["compaction_percentage"] == 60
        assert client.post("/v1/image-analysis", files=files, data={"stage": "day0"}).status_code == 422
        assert client.post("/v1/image-analysis", files=files, data={"stage": "day1"}).json()["masks"] == {}
        assert client.post("/v1/image-analysis", files=files, data={"stage": "day5"}).json()["morphology"]["short_code"] == "1NA"
