"""Shared segmentation implementation for GPU and CPU service deployments."""
import os
import re
import logging
import base64
from functools import lru_cache
import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)
DEVICE = os.environ.get("EMBRYO_DEVICE", "cpu")
@lru_cache(maxsize=None)
def _get_model(weights_path: str) -> YOLO:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"YOLO weights file not found: {weights_path}")
    m = YOLO(weights_path)
    try:
        m.fuse()
    except Exception:
        pass
    return m


def warmup_models():
    """Preload every YOLO checkpoint used by the pipeline. Call once from the
    FastAPI startup hook so the first user request doesn't eat the cold-load
    cost (~0.5-2s per model, 5 models)."""
    for path in (PB_WEIGHTS, MORULA_WEIGHTS, CELL_WEIGHTS, FRAG_WEIGHTS, EMBRYO_WEIGHTS):
        try:
            _get_model(path)
        except Exception:
            logger.exception("Failed to warm model %s", path)

PB_WEIGHTS = os.environ.get("PB_WEIGHTS", "")
MORULA_WEIGHTS = os.environ.get("MORULA_WEIGHTS", "")
CELL_WEIGHTS = os.environ.get("CELL_WEIGHTS", "")
FRAG_WEIGHTS = os.environ.get("FRAG_WEIGHTS", "")
EMBRYO_WEIGHTS = os.environ.get("EMBRYO_WEIGHTS", "")

CLASS_NAME_PB = "zona-pellucida"
CLASS_NAME_MORULA = "Morula"
CLASS_NAME_CELL = "cell"
CLASS_NAME_FRAG = "Fragments"
CLASS_NAME_EMBRYO = "Embryo"

IMGSZ = 640
CONF = 0.25
ESCO_LIVE_IMAGE_PATH = "/integrations/esco/live/image"


def polylist_to_union_mask(h, w, polys_xy):
    m = np.zeros((h, w), dtype=np.uint8)
    for poly in polys_xy:
        if poly is None or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255)
    return m


def _normalize_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _match_class_ids(names: dict, class_name: str) -> list[int]:
    target = _normalize_class_name(class_name)
    exact = [k for k, v in names.items() if _normalize_class_name(v) == target]
    if exact:
        return exact
    # Fallback: substring match for mild naming differences
    return [k for k, v in names.items() if target in _normalize_class_name(v) or _normalize_class_name(v) in target]


def preprocess_image(image):
    """Enhanced preprocessing (CLAHE + non-local-means denoise).

    fastNlMeansDenoisingColored is the slowest step (~200-500ms on CPU).
    Kept as the final-resort fallback; the cheap CLAHE-only variant below
    is tried first.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    return denoised


def preprocess_image_fast(image):
    """CLAHE-only enhancement — same contrast lift as preprocess_image
    without the costly non-local-means denoise."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def adjust_gamma(image, gamma=1.0):
    """Gamma correction."""
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def _has_detections(results) -> bool:
    return (
        results is not None
        and results.masks is not None
        and results.boxes is not None
        and len(results.boxes) > 0
    )


def multi_scale_inference(model, image_path, *, image=None):
    """Run YOLO inference once at a low confidence threshold, with a small
    enhancement-based fallback chain only when the primary pass returns
    nothing.

    Pass ``image`` (BGR ndarray) to skip the imread when the caller has
    already decoded the file — saves one disk read + JPEG decode per call.
    """
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        return None, None

    # Primary: one pass at the lowest threshold captures everything the
    # old 5-step sweep used to find. Pass the ndarray directly so YOLO
    # doesn't re-decode the file.
    results = model.predict(
        source=image, conf=0.05, iou=0.3, device=DEVICE, verbose=False
    )[0]
    if _has_detections(results):
        return results, image

    # Fallback 1: cheap contrast lift (CLAHE only, no denoise).
    enhanced_fast = preprocess_image_fast(image)
    results = model.predict(
        source=enhanced_fast, conf=0.05, iou=0.3, device=DEVICE, verbose=False
    )[0]
    if _has_detections(results):
        return results, image

    # Fallback 2: full preprocessing including the slow non-local-means
    # denoise. Reserved for the last-resort path so it only costs the
    # 200-500ms when nothing else worked.
    enhanced = preprocess_image(image)
    results = model.predict(
        source=enhanced, conf=0.05, iou=0.3, device=DEVICE, verbose=False
    )[0]
    if _has_detections(results):
        return results, image

    return None, image


def calculate_fragmentation(masks, classes):
    """Calculate fragmentation area and percentage (cls 0 embryo, cls 1 fragments)."""
    embryo_mask = None
    fragment_masks = []

    for mask, cls in zip(masks, classes):
        if cls == 0:
            embryo_mask = mask if embryo_mask is None else np.logical_or(embryo_mask, mask)
        elif cls == 1:
            fragment_masks.append(mask)

    if embryo_mask is None:
        return None

    embryo_area = int(np.sum(embryo_mask))
    if embryo_area <= 0:
        return None

    if fragment_masks:
        combined_fragments = fragment_masks[0]
        for frag in fragment_masks[1:]:
            combined_fragments = np.logical_or(combined_fragments, frag)
        fragment_area = int(np.sum(combined_fragments))
    else:
        combined_fragments = np.zeros_like(embryo_mask, dtype=bool)
        fragment_area = 0

    fragmentation_pct = (fragment_area / embryo_area) * 100.0
    return {
        "embryo_mask": embryo_mask.astype(np.uint8) * 255,
        "fragment_mask": combined_fragments.astype(np.uint8) * 255,
        "embryo_area_px": embryo_area,
        "fragment_area_px": fragment_area,
        "fragmentation_pct": fragmentation_pct,
    }


def infer_fragmentation_from_model(weights_path: str, image_path: str, *, image=None):
    """
    Multi-scale fragmentation inference. Returns metrics dict or None.
    """
    model = _get_model(weights_path)
    results, image = multi_scale_inference(model, image_path, image=image)
    if results is None or image is None or results.masks is None or results.boxes is None:
        return None

    h, w = image.shape[:2]
    raw_masks = results.masks.data.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    masks = []
    for m in raw_masks:
        bm = (m > 0.5).astype(np.uint8)
        if bm.shape[0] != h or bm.shape[1] != w:
            bm = cv2.resize(bm, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append(bm.astype(bool))

    return calculate_fragmentation(masks, cls_ids.tolist())


def largest_component(mask_u8):
    m = (mask_u8 > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if n <= 2:
        return (m * 255).astype(np.uint8)
    areas = [(lab == i).sum() for i in range(1, n)]
    keep = 1 + int(np.argmax(areas))
    return ((lab == keep).astype(np.uint8) * 255)


def dilate(mask_u8, k):
    if k <= 0:
        return mask_u8
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    return (cv2.dilate((mask_u8 > 0).astype(np.uint8), ker, 1) * 255).astype(np.uint8)


def erode(mask_u8, k):
    if k <= 0:
        return mask_u8
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    return (cv2.erode((mask_u8 > 0).astype(np.uint8), ker, 1) * 255).astype(np.uint8)


def ring_from_mask(mask_u8, k=25):
    """Boundary ring = mask - erode(mask,k)."""
    m = (mask_u8 > 0).astype(np.uint8)
    inner = cv2.erode(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8), 1)
    ring = (m > 0) & (inner == 0)
    return (ring.astype(np.uint8) * 255)


def _ensure_u8(mask):
    if mask is None:
        return None
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() == 1:
        mask = mask * 255
    return mask


def _morula_edge(mask_u8, thickness=2):
    m = (mask_u8 > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=1)
    ero = cv2.erode(m, k, iterations=1)
    edge = (dil - ero) > 0
    edge = edge.astype(np.uint8) * 255
    if thickness > 1:
        edge = cv2.dilate(edge, np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8), iterations=1)
    return edge


def compute_and_render_morula_components(
    image_bgr,
    morula_mask,
    cell_mask,
    frag_mask,
    *,
    use_cells_in_morula=True,
    clean_largest_morula=True,
    alpha_cells=0.35,
    alpha_frags=0.35,
    alpha_compaction=0.25,
    edge_thickness=2
):
    """
    Returns:
      overlay_bgr, metrics_dict, masks_dict
    """
    morula = _ensure_u8(morula_mask)
    cells = _ensure_u8(cell_mask)
    frags = _ensure_u8(frag_mask)

    if clean_largest_morula:
        morula = largest_component(morula)

    morula_bin = morula > 0

    if use_cells_in_morula:
        cells_bin = (cells > 0) & morula_bin
    else:
        cells_bin = (cells > 0)

    frags_bin = (frags > 0) & morula_bin
    comp_bin = morula_bin & ~cells_bin & ~frags_bin

    morula_area = int(morula_bin.sum())
    cells_area = int(cells_bin.sum())
    frags_area = int(frags_bin.sum())
    comp_area = int(comp_bin.sum())

    if morula_area == 0:
        metrics = {
            "morula_area_px": 0,
            "cells_px": 0,
            "frags_px": 0,
            "comp_px": 0,
            "cells_pct": 0.0,
            "frags_pct": 0.0,
            "comp_pct": 0.0,
        }
        overlay = image_bgr.copy()
        return overlay, metrics, {
            "morula": morula,
            "cells_in_morula": None,
            "frags_in_morula": None,
            "compaction": None,
        }

    cells_pct = 100.0 * cells_area / morula_area
    frags_pct = 100.0 * frags_area / morula_area
    comp_pct = 100.0 * comp_area / morula_area

    metrics = {
        "morula_area_px": morula_area,
        "cells_px": cells_area,
        "frags_px": frags_area,
        "comp_px": comp_area,
        "cells_pct": cells_pct,
        "frags_pct": frags_pct,
        "comp_pct": comp_pct,
    }

    overlay = image_bgr.copy().astype(np.float32)

    col_cells = np.array([0, 255, 0], dtype=np.float32)
    col_frags = np.array([0, 165, 255], dtype=np.float32)
    col_comp = np.array([255, 0, 255], dtype=np.float32)
    col_edge = (255, 255, 0)

    overlay[comp_bin] = (1 - alpha_compaction) * overlay[comp_bin] + alpha_compaction * col_comp
    overlay[frags_bin] = (1 - alpha_frags) * overlay[frags_bin] + alpha_frags * col_frags
    overlay[cells_bin] = (1 - alpha_cells) * overlay[cells_bin] + alpha_cells * col_cells
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    edge = _morula_edge(morula, thickness=edge_thickness)
    overlay[edge > 0] = col_edge

    masks_out = {
        "morula": (morula_bin.astype(np.uint8) * 255),
        "cells_in_morula": (cells_bin.astype(np.uint8) * 255),
        "frags_in_morula": (frags_bin.astype(np.uint8) * 255),
        "compaction": (comp_bin.astype(np.uint8) * 255),
        "morula_edge": edge,
    }

    return overlay, metrics, masks_out


def infer_cell_mask_filtered_by_morula(
    cell_weights: str,
    image_path: str,
    morula_mask_u8: np.ndarray,
    *,
    imgsz: int = 640,
    conf: float = 0.25,
    max_cell_frac: float = 0.25,
    cell_class_name: str | None = None,
    image=None,
):
    """
    Returns:
      cell_mask_filtered (uint8 0/255 union of kept instances),
      debug dict
    Logic:
      keep instance if area(instance intersect morula) <= max_cell_frac * area(morula)
    """
    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    h0, w0 = image.shape[:2]

    morula_bin = (morula_mask_u8 > 0).astype(np.uint8)
    morula_area = int(morula_bin.sum())
    if morula_area == 0:
        return np.zeros((h0, w0), np.uint8), {"reason": "morula_empty"}

    max_allowed = max_cell_frac * morula_area

    model = _get_model(cell_weights)
    r = model.predict(
        source=image, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False
    )[0]

    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return np.zeros((h0, w0), np.uint8), {"reason": "no_cell_detections"}

    target_id = None
    if cell_class_name is not None:
        name_to_id = {v: k for k, v in model.names.items()}
        if cell_class_name not in name_to_id:
            raise ValueError(f"'{cell_class_name}' not in {list(model.names.values())}")
        target_id = name_to_id[cell_class_name]

    masks = r.masks.data.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    # Do the morula intersection test in model space (640²-ish) rather than
    # upscaling every instance back to the original (often 1024²+). The
    # threshold scales with morula area so the keep/drop decision is
    # equivalent, and we trade N per-instance resizes for one morula resize.
    mh, mw = masks.shape[1:]
    morula_model = cv2.resize(morula_bin, (mw, mh), interpolation=cv2.INTER_NEAREST)
    morula_model_area = int(morula_model.sum())
    max_allowed_model = max_cell_frac * morula_model_area

    kept, dropped = 0, 0
    union_modelspace = np.zeros((mh, mw), dtype=np.uint8)

    for m_i, cid in zip(masks, cls_ids):
        if target_id is not None and cid != target_id:
            continue

        inst_model = (m_i > 0.5).astype(np.uint8)
        inst_area_in_morula_model = int(((inst_model > 0) & (morula_model > 0)).sum())

        if inst_area_in_morula_model <= max_allowed_model:
            kept += 1
            union_modelspace = np.maximum(union_modelspace, inst_model)
        else:
            dropped += 1

    union_orig = cv2.resize(union_modelspace, (w0, h0), interpolation=cv2.INTER_NEAREST)
    cell_mask_filtered = (union_orig > 0).astype(np.uint8) * 255

    debug = {
        "morula_area_px": morula_area,
        "max_allowed_cell_area_px": float(max_allowed),
        "kept_instances": kept,
        "dropped_instances": dropped,
        "filtered_cell_area_px": int((cell_mask_filtered > 0).sum()),
    }
    return cell_mask_filtered, debug


def infer_union_mask(weights_path, image_path, class_name=None, imgsz=640, conf=0.25, *, image=None):
    model = _get_model(weights_path)
    source = image if image is not None else image_path
    r = model.predict(
        source=source, imgsz=imgsz, conf=conf, device=DEVICE, verbose=False
    )[0]
    h, w = r.orig_shape

    if r.masks is None or r.masks.xy is None or len(r.masks.xy) == 0:
        return np.zeros((h, w), np.uint8)

    if class_name is None:
        return polylist_to_union_mask(h, w, r.masks.xy)

    if r.boxes is None:
        return polylist_to_union_mask(h, w, r.masks.xy)

    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    names = model.names
    target_ids = _match_class_ids(names, class_name)
    if not target_ids:
        logger.warning("Class '%s' not found in model names: %s", class_name, list(names.values()))
        return np.zeros((h, w), np.uint8)

    keep_polys = []
    for poly, cid in zip(r.masks.xy, cls_ids):
        if int(cid) in target_ids:
            keep_polys.append(poly)

    if not keep_polys:
        logger.warning("No polygons matched for class '%s' (ids=%s)", class_name, target_ids)
        return np.zeros((h, w), np.uint8)

    return polylist_to_union_mask(h, w, keep_polys)


def infer_morula_mask_stretch(
    weights,
    image_path,
    target_class_name="Morula",
    rf_size=448,
    conf=0.001,
    *,
    image=None,
):
    model = _get_model(weights)

    if image is None:
        image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    h0, w0 = image.shape[:2]

    img_rs = cv2.resize(image, (rf_size, rf_size), interpolation=cv2.INTER_LINEAR)
    r = model.predict(
        source=img_rs, imgsz=rf_size, conf=conf, device=DEVICE, verbose=False
    )[0]

    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        return np.zeros((h0, w0), np.uint8), r, model.names

    name_to_id = {v: k for k, v in model.names.items()}
    if target_class_name not in name_to_id:
        raise ValueError(f"'{target_class_name}' not in {list(model.names.values())}")
    tid = name_to_id[target_class_name]

    masks = r.masks.data.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    union = np.zeros(masks.shape[1:], dtype=np.uint8)
    for m, cid in zip(masks, cls_ids):
        if cid == tid:
            union = np.maximum(union, (m > 0.5).astype(np.uint8))

    union = cv2.resize(union, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return (union * 255).astype(np.uint8), r, model.names


def morula_refine_watershed(gray_u8, inside_roi, morula_raw, pb_mask):
    """
    Irregular morula extraction using gradient watershed:
      FG marker: eroded morula_raw
      BG marker: PB neighborhood + boundary ring + outside ROI
    """
    h, w = gray_u8.shape

    work = cv2.bitwise_and(gray_u8, gray_u8, mask=inside_roi)
    gx = cv2.Scharr(work, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(work, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    grad_u8 = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_u8 = cv2.GaussianBlur(grad_u8, (5, 5), 0)

    fg = erode(morula_raw, 2)

    bg = np.zeros((h, w), np.uint8)
    bg |= dilate(pb_mask, 20)
    bg |= ring_from_mask(inside_roi, k=25)
    bg |= ((inside_roi == 0).astype(np.uint8) * 255)

    markers = np.zeros((h, w), np.int32)
    markers[bg > 0] = 1
    markers[fg > 0] = 2

    terrain = cv2.cvtColor(255 - grad_u8, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(terrain, markers)

    morula = (ws == 2).astype(np.uint8) * 255
    morula = largest_component(morula)
    return morula


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _as_data_url(b64_png: str | None) -> str | None:
    if not b64_png:
        return None
    return f"data:image/png;base64,{b64_png}"

def _encode_color_mask_base64(mask_u8, rgb, alpha=180):
    if mask_u8 is None:
        return None
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    m = (mask_u8 > 0).astype(np.uint8)
    h, w = m.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
    bgra[..., 0] = b * m
    bgra[..., 1] = g * m
    bgra[..., 2] = r * m
    bgra[..., 3] = (m * alpha).astype(np.uint8)
    ok, buf = cv2.imencode('.png', bgra)
    if not ok:
        raise ValueError('Failed to encode color mask PNG')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def _encode_mask_base64(mask_u8: np.ndarray) -> str:
    if mask_u8 is None:
        return None
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    ok, buf = cv2.imencode(".png", mask_u8)
    if not ok:
        raise ValueError("Failed to encode mask PNG")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _count_components(mask_u8: np.ndarray) -> int:
    if mask_u8 is None:
        return 0
    bin_m = (mask_u8 > 0).astype(np.uint8)
    n, _ = cv2.connectedComponents(bin_m)
    return max(0, n - 1)


def run_pipeline(image_path: str):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    required = {
        "PB_WEIGHTS": PB_WEIGHTS,
        "MORULA_WEIGHTS": MORULA_WEIGHTS,
        "CELL_WEIGHTS": CELL_WEIGHTS,
        "FRAG_WEIGHTS": FRAG_WEIGHTS,
        "EMBRYO_WEIGHTS": EMBRYO_WEIGHTS,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing model configuration: {', '.join(missing)}")

    # Decode the file once and reuse the ndarray for every inference call —
    # YOLO accepts numpy arrays as ``source=``, so this collapses what used
    # to be ~9 imread/decode passes per request down to one.
    pb_mask = infer_union_mask(PB_WEIGHTS, image_path, class_name=CLASS_NAME_PB, imgsz=IMGSZ, conf=CONF, image=bgr)
    embryo_mask = infer_union_mask(EMBRYO_WEIGHTS, image_path, class_name=CLASS_NAME_EMBRYO, imgsz=IMGSZ, conf=CONF, image=bgr)
    zona_pellucida = ((embryo_mask > 0) & ~(pb_mask > 0)).astype(np.uint8) * 255
    morula_raw, _r_morula, _ = infer_morula_mask_stretch(
        MORULA_WEIGHTS, image_path, target_class_name=CLASS_NAME_MORULA, conf=0.05, image=bgr
    )
    morula_raw = (morula_raw > 0).astype(np.uint8) * 255
    morula_raw = cv2.morphologyEx(morula_raw, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), 1)

    cell_mask, cell_dbg = infer_cell_mask_filtered_by_morula(
        CELL_WEIGHTS,
        image_path,
        morula_mask_u8=morula_raw,
        imgsz=IMGSZ,
        conf=0.25,
        max_cell_frac=0.45,
        cell_class_name=CLASS_NAME_CELL,
        image=bgr,
    )

    frag_metrics = infer_fragmentation_from_model(FRAG_WEIGHTS, image_path, image=bgr)
    if frag_metrics and frag_metrics.get("fragment_mask") is not None:
        frag_mask = frag_metrics["fragment_mask"]
    else:
        frag_mask = infer_union_mask(FRAG_WEIGHTS, image_path, class_name=CLASS_NAME_FRAG, imgsz=IMGSZ, conf=0.05, image=bgr)

    inside_roi = largest_component(dilate((cell_mask | morula_raw).astype(np.uint8), 15))
    morula = morula_refine_watershed(gray, inside_roi, morula_raw, pb_mask)

    cells_in_morula = ((cell_mask > 0) & (morula > 0)).astype(np.uint8) * 255
    frag_in_morula = ((frag_mask > 0) & (morula > 0)).astype(np.uint8) * 255
    compaction = ((morula > 0) & ~(cells_in_morula > 0) & ~(frag_in_morula > 0)).astype(np.uint8) * 255

    overlay, metrics, masks = compute_and_render_morula_components(
        image_bgr=bgr,
        morula_mask=morula_raw,
        cell_mask=cell_mask,
        frag_mask=frag_mask,
        alpha_cells=0.40,
        alpha_frags=0.35,
        alpha_compaction=0.22,
        edge_thickness=2
    )

    return {
        "pb": pb_mask,
        "zona_pellucida": zona_pellucida,
        "inside_roi": inside_roi,
        "morula_raw": morula_raw,
        "morula": morula,
        "cells_in_morula": cells_in_morula,
        "frag_in_morula": frag_in_morula,
        "compaction": compaction,
        "overlay": overlay,
        "metrics": metrics,
        "masks": masks,
        "cell_debug": cell_dbg,
        "fragmentation_metrics": frag_metrics,
    }


def get_explainability_data(image_path: str) -> dict:
    """
    Generate AI explainability data for the given image.
    """
    if not image_path:
        raise ValueError("image_path is required")

    pipeline = run_pipeline(image_path)

    color_map = {
        "cells": [0, 255, 0],
        "fragmentation": [255, 165, 0],
        "morula": [255, 0, 0],
        "polar_body": [0, 0, 255],
        "zona": [255, 255, 0],
        "zona_pellucida": [255, 255, 0],
        "compaction": [255, 0, 255],
    }

    frags_pct = float(pipeline["metrics"].get("frags_pct", 0.0))
    comp_pct = float(pipeline["metrics"].get("comp_pct", 0.0))
    kept_cells = (pipeline.get("cell_debug") or {}).get("kept_instances")
    frag_model_metrics = pipeline.get("fragmentation_metrics") or {}
    model_frag_pct = frag_model_metrics.get("fragmentation_pct")
    model_frag_area = frag_model_metrics.get("fragment_area_px")
    model_embryo_area = frag_model_metrics.get("embryo_area_px")

    final_frag_pct = round(float(model_frag_pct), 2) if model_frag_pct is not None else round(frags_pct, 2)
    metrics = {
        "cell_count": int(kept_cells) if isinstance(kept_cells, int) else _count_components(pipeline["cells_in_morula"]),
        "fragmentation_area_px": int(model_frag_area) if model_frag_area is not None else int(pipeline["metrics"].get("frags_px", 0)),
        "embryo_area_px": int(model_embryo_area) if model_embryo_area is not None else None,
        "fragmentation_percentage": final_frag_pct,
        "compaction_percentage": round(comp_pct, 2),
        "zona_thickness": None,
    }

    return {
        "masks": {
            "cells": _as_data_url(_encode_color_mask_base64(pipeline["cells_in_morula"], color_map["cells"])),
            "fragmentation": _as_data_url(_encode_color_mask_base64(pipeline["frag_in_morula"], color_map["fragmentation"])),
            "morula": _as_data_url(_encode_color_mask_base64(pipeline["morula"], color_map["morula"])),
            "polar_body": _as_data_url(_encode_color_mask_base64(pipeline["pb"], color_map["polar_body"])),
            "zona_pellucida": _as_data_url(_encode_color_mask_base64(pipeline["zona_pellucida"], color_map["zona_pellucida"])),
            "zona": _as_data_url(_encode_color_mask_base64(pipeline["zona_pellucida"], color_map["zona"])),
            "compaction": _as_data_url(_encode_color_mask_base64(pipeline["compaction"], color_map["compaction"])),
        },
        "color_map": color_map,
        "metrics": metrics,
    }
