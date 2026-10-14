from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from inference_runtime import segmentation


def _prediction(mask):
    boxes = MagicMock()
    boxes.__len__.return_value = 1
    boxes.cls = torch.tensor([0])
    return SimpleNamespace(boxes=boxes, masks=SimpleNamespace(data=torch.tensor(mask[None])))


def test_label_matching_preserves_structure_identity():
    names = {0: "Embryo", 1: "polar-body", 2: "morula", 3: "Cell"}
    assert segmentation._match_class_ids(names, "Morula") == [2]
    assert segmentation._match_class_ids(names, "cell") == [3]
    assert segmentation._match_class_ids(names, "polar_body") == [1]
    assert segmentation._match_class_ids(names, "zona-pellucida") == []
    assert segmentation._match_class_ids({0: "MorulaOther"}, "Morula") == []
    assert segmentation.CLASS_NAME_PB == "polar-body"


def test_detected_lowercase_morula_does_not_crash(monkeypatch):
    mask = np.ones((4, 4), dtype=np.float32)
    model = SimpleNamespace(names={0: "morula"}, predict=lambda **kwargs: [_prediction(mask)])
    monkeypatch.setattr(segmentation, "_get_model", lambda _: model)
    result, _, _ = segmentation.infer_morula_mask_stretch(
        "configured.pt", "unused.png", target_class_name="Morula",
        image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    assert result.shape == (8, 8)
    assert np.all(result == 255)


def test_detected_mixed_case_cell_uses_model_class_id(monkeypatch):
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, 0] = 1
    model = SimpleNamespace(names={0: "Cell"}, predict=lambda **kwargs: [_prediction(mask)])
    monkeypatch.setattr(segmentation, "_get_model", lambda _: model)
    result, debug = segmentation.infer_cell_mask_filtered_by_morula(
        "configured.pt", "unused.png", np.ones((8, 8), dtype=np.uint8)*255,
        cell_class_name="cell", image=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    assert result.any()
    assert debug["kept_instances"] == 1
