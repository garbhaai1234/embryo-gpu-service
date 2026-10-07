"""
Embryo Analysis GPU Microservice

Runs as a standalone FastAPI service on the GPU VM. Owns the heavy models
used by the multi-embryo page:
  - Mask R-CNN sperm detector (maskrcnn_sperm_latest_v2.pth)
  - ConvNeXt classifier (best_convnextv2_base.pt)
  - Gardner morphology multi-task model (best_gardners.pt)

The main Garbha AI app (CPU VM) proxies multi-embryo detect/predict calls
to this service. S3 I/O stays on the CPU side.

Endpoints:
  GET  /health           — liveness + model load status
  POST /detect           — Mask R-CNN detect on one image, returns bboxes + crops
  POST /predict          — single-image classifier + Grad-CAM + morphology
  POST /predict_batch    — batched classifier + Grad-CAM + morphology
"""

import os
from typing import Any, Dict, List, Optional
import io
import base64
import itertools
import logging
import asyncio
import configparser
import httpx

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.security import APIKeyHeader
from torchvision.models import convnext_base
from esco_focal_plane_morphokinetics import register_esco_focal_plane_morphokinetics

import time
import torch
import torchvision
from torchvision.ops import nms
from torchvision.transforms.functional import to_tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Dedicated logger for the best.pt multi-embryo detector so its lines can be
# isolated in the console with:  tail -f uvicorn.log | grep embryo.yolo
yolo_logger = logging.getLogger("embryo.yolo")

# ── Config ──────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_config = configparser.ConfigParser()
_config.read(os.path.join(_BASE_DIR, "config.ini"))

ESCO_BRIDGE_BASE_URL = os.environ.get(
    "ESCO_BRIDGE_BASE_URL",
    _config.get("esco", "bridge_base_url", fallback=""),
).rstrip("/")

ESCO_BRIDGE_API_KEY = os.environ.get(
    "ESCO_BRIDGE_API_KEY",
    _config.get("esco", "bridge_api_key", fallback=""),
)

ESCO_FETCH_CONCURRENCY = int(os.environ.get("ESCO_FETCH_CONCURRENCY", "24"))
ESCO_FETCH_TIMEOUT = float(os.environ.get("ESCO_FETCH_TIMEOUT", "120"))


def _cfg(section: str, key: str, fallback: str = "") -> str:
    """Read config: env var > config.ini > fallback. Mirrors sperm service."""
    env_key = f"EMBRYO_{section.upper()}_{key.upper()}"
    return os.environ.get(env_key, _config.get(section, key, fallback=fallback))


MASKRCNN_MODEL_PATH = _cfg(
    "models",
    "maskrcnn_path",
    _cfg("models", "yolo_path", os.path.join(_BASE_DIR, "maskrcnn_sperm_latest_v2.pth")),
)
CLASSIFIER_MODEL_PATH = _cfg(
    "models", "classifier_path", os.path.join(_BASE_DIR, "best_convnextv2_base.pt")
)
GARDNERS_MODEL_PATH = _cfg(
    "models", "gardners_path", os.path.join(_BASE_DIR, "best_gardners.pt")
)
DEVICE = os.environ.get(
    "EMBRYO_DEVICE", _config.get("models", "device", fallback="cuda")
)
PREDICT_BATCH_SIZE = int(_cfg("models", "batch_size", "32"))

# ── Mask R-CNN detection config ──────────────────────────────────
DETECT_USE_SLICING = _cfg("detect", "use_slicing", "false").lower() == "true"
DETECT_SLICE_ROWS = int(_cfg("detect", "slice_rows", "2"))
DETECT_SLICE_COLS = int(_cfg("detect", "slice_cols", "2"))
DETECT_SLICE_OVERLAP_RATIO = float(_cfg("detect", "slice_overlap_ratio", "0.20"))

DETECT_SCORE_THRESH = float(_cfg("detect", "score_thresh", "0.55"))
DETECT_MASK_THRESH = float(_cfg("detect", "mask_thresh", "0.50"))
DETECT_NMS_IOU_THRESH = float(_cfg("detect", "nms_iou_thresh", "0.45"))

DETECT_MIN_MASK_AREA = int(_cfg("detect", "min_mask_area", "15"))
DETECT_MAX_MASK_AREA = int(_cfg("detect", "max_mask_area", "100000"))

DETECT_TILE_BATCH_SIZE = int(_cfg("detect", "tile_batch_size", "4"))
DETECT_USE_AMP = _cfg("detect", "use_amp", "true").lower() == "true"

# ── best.pt YOLO multi-embryo detector (console logging) ─────────
# Runs alongside /detect (the multi-embryo path) purely to log what best.pt
# sees. It does not change the response payload.
YOLO_LOG_ENABLED = _cfg("detect", "yolo_log_enabled", "true").lower() == "true"
YOLO_LOG_PATH = _cfg("detect", "yolo_log_path", "best.pt")
if not os.path.isabs(YOLO_LOG_PATH):
    YOLO_LOG_PATH = os.path.join(_BASE_DIR, YOLO_LOG_PATH)
YOLO_LOG_CONF = float(_cfg("detect", "yolo_log_conf", "0.25"))
YOLO_LOG_IOU = float(_cfg("detect", "yolo_log_iou", "0.45"))
YOLO_LOG_IMGSZ = int(_cfg("detect", "yolo_log_imgsz", "640"))
# Only emit per-box detail lines when the image is genuinely multi-embryo
# (>= this many boxes). Set to 1 to log single-embryo images in full too.
YOLO_LOG_MULTI_MIN = int(_cfg("detect", "yolo_log_multi_min", "2"))

API_KEY = os.environ.get("EMBRYO_API_KEY", _config.get("app", "api_key", fallback=""))
if not API_KEY:
    logger.warning(
        "EMBRYO_API_KEY not set. Configure via env var or config.ini [app] api_key."
    )

# ── Auth ────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Depends(_api_key_header)):
    if not API_KEY:
        return  # auth disabled when key not configured (dev only)
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Class labels ────────────────────────────────────────────────────────
class_names = [
    "8 Cells Grade A",
    "8 Cells Grade B",
    "8 Cells Grade C",
    "Blastocyst Grade A",
    "Blastocyst Grade B",
    "Blastocyst Grade C",
    "Error Image",
    "Morula Grade A",
    "Morula Grade B",
    "Morula Grade C",
]

exp_labels = [
    "1 (Morula Stage)",
    "2 (Early Blastocyst)",
    "3 (Blastocyst)",
    "4 (Expanded)",
    "5 (Fully Hatched)",
]
icm_labels = ["A (Good)", "B (Fair)", "C (Poor)", "Not Available"]
te_labels = ["A (Good)", "B (Fair)", "C (Poor)", "Not Available"]
expansion_scores = {
    "5 (Fully Hatched)": 10,
    "4 (Expanded)": 9,
    "3 (Blastocyst)": 8,
    "2 (Early Blastocyst)": 6,
    "1 (Morula Stage)": 4,
}
icm_scores = {"A (Good)": 10, "B (Fair)": 7, "C (Poor)": 4, "Not Available": 0}
te_scores = {"A (Good)": 10, "B (Fair)": 7, "C (Poor)": 4, "Not Available": 0}


# ── Model loading ───────────────────────────────────────────────────────
_torch_device = torch.device(DEVICE if torch.cuda.is_available() or DEVICE == "cpu" else "cpu")
logger.info(f"Embryo GPU service: torch device = {_torch_device}")


def _load_maskrcnn_sperm(path: str):
    """
    Load sperm Mask R-CNN v1 model.

    This uses torchvision.models.detection.maskrcnn_resnet50_fpn because your
    benchmark confirmed:
      Candidate v1: missing=0, unexpected=0, score=0
      Candidate v2: missing=72, unexpected=16, score=88
    """
    num_classes = 2  # background + sperm

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
    )

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    model.to(_torch_device)
    model.eval()
    return model

def _load_classifier(path: str, num_classes: int = 10):
    model = convnext_base(weights=None)
    model.classifier = nn.Sequential(
        nn.Flatten(1), nn.Linear(model.classifier[2].in_features, num_classes)
    )
    state_dict = torch.load(path, map_location=_torch_device)
    model.load_state_dict(state_dict)
    model.to(_torch_device)
    model.eval()
    return model


class MultiTaskModel(nn.Module):
    """Gardner morphology multi-task model (expansion / ICM / TE heads)."""

    def __init__(self, num_exp: int, num_icm: int, num_te: int):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "convnextv2_base", pretrained=False, num_classes=0
        )
        in_features = self.backbone.num_features
        self.dropout = nn.Dropout(0.5)
        self.exp_head = nn.Linear(in_features, num_exp)
        self.icm_head = nn.Linear(in_features, num_icm)
        self.te_head = nn.Linear(in_features, num_te)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return {
            "expansion": self.exp_head(x),
            "icm": self.icm_head(x),
            "te": self.te_head(x),
        }


def _load_gardners(path: str):
    model = MultiTaskModel(len(exp_labels), len(icm_labels), len(te_labels))
    state_dict = torch.load(path, map_location=_torch_device)
    model.load_state_dict(state_dict)
    model.to(_torch_device)
    model.eval()
    return model


# Load eagerly so the first request isn't penalised by cold-start.
logger.info("Loading Mask R-CNN sperm detector...")
_detector = _load_maskrcnn_sperm(MASKRCNN_MODEL_PATH)

logger.info("Loading ConvNeXt classifier...")
_classifier = _load_classifier(CLASSIFIER_MODEL_PATH)

logger.info("Loading Gardner morphology model...")
try:
    _gardners = _load_gardners(GARDNERS_MODEL_PATH)
except Exception as e:
    logger.warning(f"Gardner model failed to load: {e}")
    _gardners = None

logger.info("Loading best.pt YOLO multi-embryo detector (console logging)...")
_yolo_multi = None
_yolo_class_names: Dict[int, str] = {}
if YOLO_LOG_ENABLED:
    try:
        from ultralytics import YOLO  # imported lazily so it stays optional

        _yolo_multi = YOLO(YOLO_LOG_PATH)
        _yolo_multi.to(_torch_device)
        _yolo_class_names = dict(getattr(_yolo_multi, "names", {}) or {})
        yolo_logger.info(
            "best.pt loaded | path=%s | device=%s | classes=%s | conf=%.2f iou=%.2f imgsz=%d",
            YOLO_LOG_PATH,
            _torch_device,
            list(_yolo_class_names.values()) or "unknown",
            YOLO_LOG_CONF,
            YOLO_LOG_IOU,
            YOLO_LOG_IMGSZ,
        )
    except Exception as e:
        _yolo_multi = None
        yolo_logger.warning("best.pt failed to load (%s) — multi-embryo logs disabled", e)
else:
    yolo_logger.info("best.pt multi-embryo logging disabled via config [detect] yolo_log_enabled")

logger.info("All models loaded.")


# ── Preprocessing ───────────────────────────────────────────────────────
_classifier_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
_morphology_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _decode_image(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_jpeg_b64(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _square_padded_bbox(
    bbox: List[int],
    frame_shape: tuple,
    pad_ratio: float = 0.35,
) -> tuple:
    """Return a square display crop around a bbox, clipped to image bounds."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(frame_w - 1, x1))
    y1 = max(0, min(frame_h - 1, y1))
    x2 = max(0, min(frame_w, x2))
    y2 = max(0, min(frame_h, y2))

    if x2 <= x1 or y2 <= y1:
        return x1, y1, x2, y2

    box_w = x2 - x1
    box_h = y2 - y1
    side = int(round(max(box_w, box_h) * (1 + pad_ratio)))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop_x1 = max(0, cx - side // 2)
    crop_y1 = max(0, cy - side // 2)
    crop_x2 = min(frame_w, crop_x1 + side)
    crop_y2 = min(frame_h, crop_y1 + side)

    crop_x1 = max(0, crop_x2 - side)
    crop_y1 = max(0, crop_y2 - side)

    return crop_x1, crop_y1, crop_x2, crop_y2


_yolo_request_counter = itertools.count(1)


def _log_best_pt_multi_embryo(frame_bgr: np.ndarray, filename: str) -> Optional[Dict[str, Any]]:
    """Run best.pt on a full frame and log every embryo it detects.

    Console-only diagnostics for the multi-embryo path. Never raises — a
    failure here must not break /detect.
    """
    if _yolo_multi is None:
        return None

    req_id = next(_yolo_request_counter)
    frame_h, frame_w = frame_bgr.shape[:2]

    try:
        t0 = time.perf_counter()
        result = _yolo_multi.predict(
            source=frame_bgr,
            conf=YOLO_LOG_CONF,
            iou=YOLO_LOG_IOU,
            imgsz=YOLO_LOG_IMGSZ,
            device=_torch_device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = result.boxes
        count = 0 if boxes is None else len(boxes)
        is_multi = count >= YOLO_LOG_MULTI_MIN
        kind = "MULTI-EMBRYO" if is_multi else ("SINGLE-EMBRYO" if count == 1 else "NO-EMBRYO")

        yolo_logger.info(
            "[best.pt #%d] %s | file=%s | image=%dx%d | embryos=%d | %.1f ms | conf>=%.2f",
            req_id, kind, filename or "unnamed", frame_w, frame_h, count, elapsed_ms, YOLO_LOG_CONF,
        )

        detections: List[Dict[str, Any]] = []
        if count:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            # Reading order: top row left-to-right, then next row.
            order = sorted(
                range(count),
                key=lambda i: (int(xyxy[i][1] // max(1, frame_h * 0.25)), xyxy[i][0]),
            )

            for rank, i in enumerate(order, start=1):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy[i])
                w, h = x2 - x1, y2 - y1
                label = _yolo_class_names.get(int(clss[i]), str(int(clss[i])))
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(confs[i]),
                    "label": label,
                })
                if is_multi:
                    yolo_logger.info(
                        "[best.pt #%d]   embryo %d/%d | class=%s | conf=%.3f | bbox=(%d,%d,%d,%d) | %dx%d px | area=%d",
                        req_id, rank, count, label, float(confs[i]),
                        x1, y1, x2, y2, w, h, w * h,
                    )

            if is_multi:
                yolo_logger.info(
                    "[best.pt #%d] summary | embryos=%d | conf min/mean/max=%.3f/%.3f/%.3f",
                    req_id, count,
                    float(confs.min()), float(confs.mean()), float(confs.max()),
                )

        return {
            "count": count,
            "is_multi_embryo": is_multi,
            "inference_ms": round(elapsed_ms, 2),
            "detections": detections,
        }

    except Exception as e:
        yolo_logger.warning("[best.pt #%d] inference failed: %s", req_id, e)
        return None


def _detect_embryo_circles_on_bgr(frame_bgr: np.ndarray) -> Dict[str, Any]:
    """Detect whole round embryos using image geometry before model fallback."""
    frame_h, frame_w = frame_bgr.shape[:2]
    min_dim = min(frame_h, frame_w)
    empty = {
        "boxes": [],
        "crop_boxes": [],
        "scores": [],
        "mask_items": [],
        "raw_detection_count": 0,
        "final_detection_count": 0,
        "duplicates_removed": 0,
        "slice_count": 1,
        "rows": 1,
        "cols": 1,
    }
    if min_dim < 80:
        return empty

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)

    min_radius = max(20, int(round(min_dim * 0.14)))
    max_radius = max(min_radius + 5, int(round(min_dim * 0.29)))
    min_dist = max(float(min_radius) * 1.6, float(min_dim) * 0.32)

    full_circles = []
    for param2 in (42, 40, 38, 36):
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dist,
            param1=55,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue

        rounded = np.round(circles[0]).astype(int).tolist()
        full_circles = [
            (cx, cy, r)
            for cx, cy, r in rounded
            if r > 0
            and cx - r >= 0
            and cy - r >= 0
            and cx + r <= frame_w
            and cy + r <= frame_h
        ]
        if full_circles:
            break

    if not full_circles:
        return empty

    row_break = frame_h * 0.62
    full_circles.sort(key=lambda circle: (0 if circle[1] < row_break else 1, circle[0]))

    boxes: List[List[int]] = []
    crop_boxes: List[List[int]] = []
    scores: List[float] = []
    mask_items: List[dict] = []

    for cx, cy, r in full_circles:
        x1 = max(0, int(cx - r))
        y1 = max(0, int(cy - r))
        x2 = min(frame_w, int(cx + r))
        y2 = min(frame_h, int(cy + r))

        pad = max(4, int(round(r * 0.08)))
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(frame_w, x2 + pad)
        crop_y2 = min(frame_h, y2 + pad)

        boxes.append([x1, y1, x2, y2])
        crop_boxes.append([crop_x1, crop_y1, crop_x2, crop_y2])
        scores.append(1.0)
        mask_items.append({"mask_area": int(round(np.pi * r * r))})

    return {
        "boxes": boxes,
        "crop_boxes": crop_boxes,
        "scores": scores,
        "mask_items": mask_items,
        "raw_detection_count": len(full_circles),
        "final_detection_count": len(full_circles),
        "duplicates_removed": 0,
        "slice_count": 1,
        "rows": 1,
        "cols": 1,
    }


# ── Mask R-CNN sliced inference helpers ─────────────────────────────────

def _generate_slices(frame_h: int, frame_w: int, rows: int, cols: int, overlap_ratio: float):
    """Generate overlapping slice boxes in xyxy format."""
    if rows <= 1 and cols <= 1:
        return [(0, 0, frame_w, frame_h)]

    slices = []

    base_h = frame_h / rows
    base_w = frame_w / cols

    pad_h = int(base_h * overlap_ratio / 2)
    pad_w = int(base_w * overlap_ratio / 2)

    for r in range(rows):
        for c in range(cols):
            x1_base = int(round(c * base_w))
            y1_base = int(round(r * base_h))
            x2_base = int(round((c + 1) * base_w))
            y2_base = int(round((r + 1) * base_h))

            x1 = max(0, x1_base - pad_w)
            y1 = max(0, y1_base - pad_h)
            x2 = min(frame_w, x2_base + pad_w)
            y2 = min(frame_h, y2_base + pad_h)

            slices.append((x1, y1, x2, y2))

    return slices


@torch.inference_mode()
def _run_maskrcnn_sliced_on_bgr(
    frame_bgr: np.ndarray,
    score_thresh: float,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run optimized Mask R-CNN sliced inference on one BGR image/frame.

    Production default is fixed 2x2 slicing because your benchmark selected:
      rows=2, cols=2, slice_count=4.
    """
    frame_h, frame_w = frame_bgr.shape[:2]

    if not DETECT_USE_SLICING:
        rows = 1
        cols = 1
    else:
        rows = rows or DETECT_SLICE_ROWS
        cols = cols or DETECT_SLICE_COLS

    slice_boxes = _generate_slices(
        frame_h=frame_h,
        frame_w=frame_w,
        rows=rows,
        cols=cols,
        overlap_ratio=DETECT_SLICE_OVERLAP_RATIO,
    )

    tile_tensors = []
    tile_locations = []

    for sx1, sy1, sx2, sy2 in slice_boxes:
        tile_bgr = frame_bgr[sy1:sy2, sx1:sx2]

        if tile_bgr.size == 0:
            continue

        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(tile_rgb).to(_torch_device)

        tile_tensors.append(tensor)
        tile_locations.append((sx1, sy1, sx2, sy2))

    all_boxes: List[List[int]] = []
    all_scores: List[float] = []
    all_local_masks: List[np.ndarray] = []
    all_tile_locations: List[tuple] = []

    raw_detection_count = 0

    for start_idx in range(0, len(tile_tensors), DETECT_TILE_BATCH_SIZE):
        batch_tensors = tile_tensors[start_idx:start_idx + DETECT_TILE_BATCH_SIZE]
        batch_locations = tile_locations[start_idx:start_idx + DETECT_TILE_BATCH_SIZE]

        with torch.cuda.amp.autocast(
            enabled=(_torch_device.type == "cuda" and DETECT_USE_AMP)
        ):
            batch_outputs = _detector(batch_tensors)

        for output, tile_loc in zip(batch_outputs, batch_locations):
            sx1, sy1, sx2, sy2 = tile_loc

            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()

            raw_detection_count += len(scores)

            for i in range(len(scores)):
                score = float(scores[i])

                if score < score_thresh:
                    continue

                box = boxes[i].copy()
                mask_prob = masks[i, 0]

                mask_local = (mask_prob > DETECT_MASK_THRESH).astype(np.uint8)
                area = int(mask_local.sum())

                if area < DETECT_MIN_MASK_AREA or area > DETECT_MAX_MASK_AREA:
                    continue

                # Convert tile bbox to full-frame bbox.
                box[0] += sx1
                box[2] += sx1
                box[1] += sy1
                box[3] += sy1

                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, min(frame_w - 1, x1))
                x2 = max(0, min(frame_w - 1, x2))
                y1 = max(0, min(frame_h - 1, y1))
                y2 = max(0, min(frame_h - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score)
                all_local_masks.append(mask_local)
                all_tile_locations.append(tile_loc)

    if len(all_boxes) == 0:
        return {
            "boxes": [],
            "scores": [],
            "mask_items": [],
            "raw_detection_count": raw_detection_count,
            "final_detection_count": 0,
            "duplicates_removed": 0,
            "slice_count": len(slice_boxes),
            "rows": rows,
            "cols": cols,
        }

    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)

    keep_indices = nms(
        boxes_tensor,
        scores_tensor,
        DETECT_NMS_IOU_THRESH,
    ).cpu().numpy().tolist()

    final_boxes: List[List[int]] = []
    final_scores: List[float] = []
    final_mask_items: List[Dict[str, Any]] = []

    for i in keep_indices:
        final_boxes.append(all_boxes[i])
        final_scores.append(all_scores[i])
        final_mask_items.append({
            "tile_location": all_tile_locations[i],
            "mask_local": all_local_masks[i],
            "mask_area": int(all_local_masks[i].sum()),
        })

    duplicates_removed = len(all_boxes) - len(final_boxes)

    return {
        "boxes": final_boxes,
        "scores": final_scores,
        "mask_items": final_mask_items,
        "raw_detection_count": raw_detection_count,
        "final_detection_count": len(final_boxes),
        "duplicates_removed": duplicates_removed,
        "slice_count": len(slice_boxes),
        "rows": rows,
        "cols": cols,
    }


# ── Grad-CAM ────────────────────────────────────────────────────────────
class GradCAM:
    """Standard Grad-CAM hooked to a target layer of the classifier.

    Hooks are registered once at construction and reused across calls. We
    keep a single instance per request (or per batch) and remove hooks in
    a finally block so the global classifier doesn't accumulate state.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hooks = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        model_output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_class] = 1
        model_output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=[2, 3])
        cam = torch.zeros_like(self.activations[0, 0])
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        cam = torch.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.cpu().numpy()


def _overlay_grad_cam(original: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    img_resized = np.array(original.resize((224, 224)))
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, alpha, heatmap, 1 - alpha, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))


# ── Inference helpers ───────────────────────────────────────────────────
def _classify_batch(images: List[Image.Image]):
    """Run the classifier on a list of PIL images, batched. Returns parallel
    lists of (class_index, confidence_with_floor)."""
    tensors = [_classifier_transform(img).unsqueeze(0) for img in images]
    stacked = torch.cat(tensors, dim=0)
    pred_idx: List[int] = []
    confidences: List[float] = []
    with torch.no_grad():
        for start in range(0, stacked.shape[0], PREDICT_BATCH_SIZE):
            chunk = stacked[start:start + PREDICT_BATCH_SIZE].to(_torch_device)
            outputs = _classifier(chunk)
            probs = torch.softmax(outputs, dim=1)
            chunk_conf, chunk_pred = torch.max(probs, dim=1)
            for c, p in zip(chunk_conf.tolist(), chunk_pred.tolist()):
                # Match the legacy 0.70 confidence floor used by the UI.
                confidences.append(max(c, 0.70))
                pred_idx.append(p)
    return tensors, pred_idx, confidences


def _morphology_batch(images: List[Image.Image]) -> List[dict]:
    """Run the Gardner model on a list of (blastocyst) images, batched."""
    if _gardners is None or not images:
        return [None] * len(images)
    tensors = torch.cat(
        [_morphology_transform(img).unsqueeze(0) for img in images], dim=0
    )
    out: List[dict] = []
    with torch.no_grad():
        for start in range(0, tensors.shape[0], PREDICT_BATCH_SIZE):
            chunk = tensors[start:start + PREDICT_BATCH_SIZE].to(_torch_device)
            preds = _gardners(chunk)
            exp_idx = torch.argmax(preds["expansion"], dim=1).tolist()
            icm_idx = torch.argmax(preds["icm"], dim=1).tolist()
            te_idx = torch.argmax(preds["te"], dim=1).tolist()
            for k in range(chunk.shape[0]):
                exp_label = exp_labels[exp_idx[k]]
                icm_label = icm_labels[icm_idx[k]]
                te_label = te_labels[te_idx[k]]
                exp_score = expansion_scores[exp_label]
                icm_score = icm_scores[icm_label]
                te_score = te_scores[te_label]
                morphology_score = (
                    None
                    if (icm_score == 0 or te_score == 0)
                    else round((exp_score + icm_score + te_score) / 3, 2)
                )
                out.append({
                    "expansion": exp_label,
                    "icm": icm_label,
                    "te": te_label,
                    "expansion_score": exp_score,
                    "icm_score": icm_score,
                    "te_score": te_score,
                    "morphology_score": morphology_score,
                    "short_code": f"{exp_label[0]}{icm_label[0]}{te_label[0]}",
                })
    return out


# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(title="Embryo Analysis GPU Service")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(_torch_device),
        "models": {
            "detector_type": (
                "embryo_hough_circle+maskrcnn_resnet50_fpn_sliced_fallback"
                if DETECT_USE_SLICING
                else "embryo_hough_circle+maskrcnn_resnet50_fpn_fallback"
            ),
            "maskrcnn": os.path.basename(MASKRCNN_MODEL_PATH),
            "classifier": os.path.basename(CLASSIFIER_MODEL_PATH),
            "gardners_loaded": _gardners is not None,
        },
        "best_pt_multi_embryo_logging": {
            "enabled": YOLO_LOG_ENABLED,
            "loaded": _yolo_multi is not None,
            "path": os.path.basename(YOLO_LOG_PATH),
            "classes": list(_yolo_class_names.values()),
            "conf": YOLO_LOG_CONF,
            "iou": YOLO_LOG_IOU,
            "imgsz": YOLO_LOG_IMGSZ,
            "multi_min": YOLO_LOG_MULTI_MIN,
        },
        "sliced_detection": {
            "enabled": DETECT_USE_SLICING,
            "rows": DETECT_SLICE_ROWS if DETECT_USE_SLICING else 1,
            "cols": DETECT_SLICE_COLS if DETECT_USE_SLICING else 1,
            "slice_count": (
                DETECT_SLICE_ROWS * DETECT_SLICE_COLS
                if DETECT_USE_SLICING
                else 1
            ),
            "score_thresh": DETECT_SCORE_THRESH,
            "mask_thresh": DETECT_MASK_THRESH,
            "nms_iou_thresh": DETECT_NMS_IOU_THRESH,
        },
    }


@app.post("/detect", dependencies=[Depends(require_api_key)])
async def detect(file: UploadFile = File(...), conf: float = Form(DETECT_SCORE_THRESH)):
    """Run embryo-level detection on a single full image.

    Returns detected bboxes plus per-bbox JPEG crops (base64). Embryo-ID
    assignment is the caller's responsibility, so this endpoint remains
    stateless across requests.
    """
    raw = await file.read()
    img = _decode_image(raw)
    np_rgb = np.array(img)
    img_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

    # best.pt console diagnostics for the multi-embryo image (log only).
    yolo_log = _log_best_pt_multi_embryo(img_bgr, file.filename or "")

    start_time = time.perf_counter()

    result = _detect_embryo_circles_on_bgr(img_bgr)
    if not result["boxes"]:
        result = _run_maskrcnn_sliced_on_bgr(
            frame_bgr=img_bgr,
            score_thresh=float(conf),
        )

    inference_time_sec = time.perf_counter() - start_time

    detections = []

    for idx, (box, score) in enumerate(zip(result["boxes"], result["scores"]), start=1):
        x1, y1, x2, y2 = map(int, box)
        crop_boxes = result.get("crop_boxes", [])
        if idx - 1 < len(crop_boxes):
            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, crop_boxes[idx - 1])
        else:
            crop_x1, crop_y1, crop_x2, crop_y2 = _square_padded_bbox(
                [x1, y1, x2, y2],
                img_bgr.shape,
            )

        crop_bgr = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if crop_bgr.size == 0:
            continue

        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        mask_area = None
        mask_items = result.get("mask_items", [])
        if idx - 1 < len(mask_items):
            mask_area = mask_items[idx - 1].get("mask_area")

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "crop_bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
            "score": float(score),
            "mask_area": mask_area,
            "crop_jpeg_b64": _pil_to_jpeg_b64(crop_pil),
        })

    yolo_logger.info(
        "[best.pt] /detect returned %d embryo crop(s) from the active pipeline "
        "(best.pt saw %s)",
        len(detections),
        yolo_log["count"] if yolo_log else "n/a",
    )

    return {
        "detections": detections,
        "count": len(detections),
        "best_pt_log": yolo_log,
        "slicing": {
            "enabled": DETECT_USE_SLICING,
            "rows": result["rows"],
            "cols": result["cols"],
            "slice_count": result["slice_count"],
            "raw_detection_count": result["raw_detection_count"],
            "final_detection_count": result["final_detection_count"],
            "duplicates_removed": result["duplicates_removed"],
            "score_thresh": float(conf),
            "mask_thresh": DETECT_MASK_THRESH,
            "nms_iou_thresh": DETECT_NMS_IOU_THRESH,
            "inference_time_sec": inference_time_sec,
            "fps_for_request": (
                1.0 / inference_time_sec
                if inference_time_sec > 0
                else None
            ),
        },
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    """Single-image predict: ConvNeXt + Grad-CAM (+ Gardner if blastocyst)."""
    raw = await file.read()
    img = _decode_image(raw)

    tensors, pred_idx, confidences = _classify_batch([img])
    ai_grade = class_names[pred_idx[0]]
    confidence = confidences[0]

    cam_overlay_b64 = None
    gradcam = GradCAM(_classifier, _classifier.features[-1])
    try:
        cam = gradcam.generate_cam(tensors[0].to(_torch_device), pred_idx[0])
        overlay = _overlay_grad_cam(img, cam)
        cam_overlay_b64 = _pil_to_jpeg_b64(overlay)
    finally:
        gradcam.close()

    morphology = None
    if "blastocyst" in ai_grade.lower():
        morph_list = _morphology_batch([img])
        morphology = morph_list[0] if morph_list else None

    return {
        "success": True,
        "ai_grade": ai_grade,
        "confidence": confidence,
        "gradcam_image": cam_overlay_b64,
        "morphology": morphology,
    }


@app.post("/predict_batch", dependencies=[Depends(require_api_key)])
async def predict_batch(
    files: List[UploadFile] = File(...),
    embryo_ids: List[str] = Form(...),
):
    """Batched predict for up to ~100 embryos in one call. `files` and
    `embryo_ids` are paired by index."""
    if len(files) != len(embryo_ids):
        raise HTTPException(
            status_code=400,
            detail=f"files ({len(files)}) and embryo_ids ({len(embryo_ids)}) length mismatch",
        )
    if not files:
        return {"success": True, "results": {}}

    images: List[Image.Image] = []
    for f in files:
        raw = await f.read()
        images.append(_decode_image(raw))

    tensors, pred_idx, confidences = _classify_batch(images)
    ai_grades = [class_names[i] for i in pred_idx]

    # Gardner only on the blastocyst subset.
    blast_positions = [i for i, g in enumerate(ai_grades) if "blastocyst" in g.lower()]
    morph_results: dict[int, dict] = {}
    if blast_positions:
        blast_imgs = [images[i] for i in blast_positions]
        morph_list = _morphology_batch(blast_imgs)
        for pos, m in zip(blast_positions, morph_list):
            if m is not None:
                morph_results[pos] = m

    # Grad-CAM serially under one shared instance (per-image backward).
    results: dict[str, dict] = {}
    gradcam = GradCAM(_classifier, _classifier.features[-1])
    try:
        for idx, emb_id in enumerate(embryo_ids):
            cam = gradcam.generate_cam(tensors[idx].to(_torch_device), pred_idx[idx])
            overlay = _overlay_grad_cam(images[idx], cam)
            results[emb_id] = {
                "success": True,
                "ai_grade": ai_grades[idx],
                "confidence": confidences[idx],
                "gradcam_image": _pil_to_jpeg_b64(overlay),
                "morphology": morph_results.get(idx),
            }
    finally:
        gradcam.close()

    return {"success": True, "results": results}


# ── Timelapse helpers ───────────────────────────────────────────────────
# The timelapse router on the CPU VM extracts scene-change frames from a
# video and ships them here for batch classification. Grad-CAM only runs
# on the chosen best frame per stage (a separate /gradcam_frame call), so
# we don't waste backward passes on candidates that get discarded.

def _apply_gradcam_full_res(bgr_frame: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Compose a Grad-CAM overlay at the input frame's native resolution
    (vs. the 224x224 used for embryo crops)."""
    cam_resized = cv2.resize(cam, (bgr_frame.shape[1], bgr_frame.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr_frame, 0.6, heatmap, 0.4, 0)


@app.post("/classify_frames", dependencies=[Depends(require_api_key)])
async def classify_frames(files: List[UploadFile] = File(...)):
    """Batched classifier for timelapse video frames. Returns per-frame
    class_idx, top-1 confidence, and the full softmax distribution so the
    caller can do its own per-stage probability aggregation. Skips Grad-CAM
    to keep the response fast — call /gradcam_frame for the chosen best
    frames only."""
    if not files:
        return {"success": True, "results": []}

    images: List[Image.Image] = []
    for f in files:
        raw = await f.read()
        images.append(_decode_image(raw))

    tensors = [_classifier_transform(img).unsqueeze(0) for img in images]
    stacked = torch.cat(tensors, dim=0)

    out_class_idx: List[int] = []
    out_conf: List[float] = []
    out_probs: List[List[float]] = []
    with torch.no_grad():
        for start in range(0, stacked.shape[0], PREDICT_BATCH_SIZE):
            chunk = stacked[start:start + PREDICT_BATCH_SIZE].to(_torch_device)
            logits = _classifier(chunk)
            probs = torch.softmax(logits, dim=1)
            chunk_conf, chunk_idx = torch.max(probs, dim=1)
            for i in range(chunk.shape[0]):
                out_class_idx.append(int(chunk_idx[i].item()))
                out_conf.append(float(chunk_conf[i].item()))
                out_probs.append([float(x) for x in probs[i].tolist()])

    results = [
        {
            "class_idx": ci,
            "class_name": class_names[ci],
            "confidence": cf,
            "probs": pr,
        }
        for ci, cf, pr in zip(out_class_idx, out_conf, out_probs)
    ]
    return {"success": True, "results": results}


@app.post("/gradcam_frame", dependencies=[Depends(require_api_key)])
async def gradcam_frame(
    file: UploadFile = File(...),
    class_idx: int = Form(...),
):
    """Grad-CAM overlay for a single video frame at the frame's native
    resolution. Used by the timelapse router on the chosen best frame per
    stage so the stored S3 image preserves full detail."""
    raw = await file.read()
    img = _decode_image(raw)
    bgr_full = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    tensor = _classifier_transform(img).unsqueeze(0).to(_torch_device)
    gradcam = GradCAM(_classifier, _classifier.features[-1])
    try:
        cam = gradcam.generate_cam(tensor, class_idx)
    finally:
        gradcam.close()

    overlay_bgr = _apply_gradcam_full_res(bgr_full, cam)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    overlay_pil = Image.fromarray(overlay_rgb)
    return {
        "success": True,
        "gradcam_image": _pil_to_jpeg_b64(overlay_pil),
    }


register_esco_focal_plane_morphokinetics(
    app=app,
    require_api_key=require_api_key,
    class_names=class_names,
    decode_image=_decode_image,
    classify_batch=_classify_batch,
    morphology_batch=_morphology_batch,
    classifier_transform=_classifier_transform,
    torch_device=_torch_device,
    classifier=_classifier,
    gradcam_cls=GradCAM,
    overlay_grad_cam=_overlay_grad_cam,
    pil_to_jpeg_b64=_pil_to_jpeg_b64,
    esco_bridge_base_url=_cfg("esco_bridge", "base_url", "http://35.238.224.167:8787"),
    esco_bridge_api_key=_cfg("esco_bridge", "api_key", ""),
)


if __name__ == "__main__":
    import uvicorn

    port = int(_cfg("app", "port", "5003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
