# Complete Embryo AI Project Documentation (Read-Only Audit)

Audit date: 2026-10-06\
Scope: read-only inventory/search of `/home/user_garbha` plus source/configuration/log inspection where available. The home tree contains very large unrelated environments, generated web assets, and sperm-analysis resources; generated dependency contents were excluded from content-oriented inspection. No model was executed and no package was installed. Classification tags used below: **CONFIRMED FROM CODE**, **CONFIRMED FROM CONFIG**, **CONFIRMED FROM LOG**, **CONFIRMED FROM DATASET**, **INFERENCE**, **NOT VERIFIED**. No checkpoint was deserialized or executed.

## 1. Files and directories inspected

Inspected source/documentation/configuration: `/home/user_garbha/Garbha_prodapp/embryo_gpu/{README.md,main.py,esco_focal_plane_morphokinetics.py,config.ini,uvicorn.log}`; inventory of `/home/user_garbha`; selected path-only searches across the home tree; `/home/user_garbha/embryoscore` pipeline and app path inventory; `/home/user_garbha/Dataset` metadata inventory; model/checkpoint path inventory; and older `/home/user_garbha/garbhaai_dev/JSON_Backend_1` resource paths. The previous checkout audit found `.git` status unavailable (`fatal: bad object HEAD`). Not every notebook cell, patient-data file, generated dependency, or binary artifact was opened. This is a broad resource audit, not a claim that every byte under home was manually reviewed.

## 2. Models discovered

- **Mask R-CNN ResNet-50 FPN** from Torchvision, configured as a two-class detector (background plus foreground). The loader uses a checkpoint state dictionary. It is a fallback detector after Hough circle detection returns no boxes.
- **ConvNeXt Base** from Torchvision with a custom ten-class classification head. This is the active stage/grade image classifier.
- **`MultiTaskModel`**, a `timm` ConvNeXtV2 Base backbone with separate expansion, ICM, and TE linear heads. This is the Gardner morphology model.
- **Ultralytics YOLO** using `best.pt`, a diagnostics/logging path only. Its output does not drive `/detect` response detections.
- Additional checkpoint artifacts: `best_original.pt`, `best_MAP_45%.pt`, and copies under `3 pt files/`. Their architecture, provenance, and role are not established by source references/configuration found in this checkout.

## 3. Active models identified

Configured/runtime code paths select Mask R-CNN from `config.ini` (`models.maskrcnn_path`), ConvNeXt from `models.classifier_path` (checked-in default `best_convnextv2_base.pt`), Gardner from `models.gardners_path` (default `best_gardners.pt`), and diagnostic YOLO `best.pt`. Mask R-CNN and ConvNeXt load eagerly. Gardner loading is caught on failure and disables morphology. The historical `uvicorn.log` records repeated Gardner load failures because `timm` was unavailable. That log is historical evidence and does not establish the current live process state. Current live weights/device and successful model loading were not independently verified.

## 4. Embryo capabilities identified

Image-level stage/grade classification, circle-based embryo localization with a Mask R-CNN fallback, Grad-CAM visualization, optional blastocyst morphology grading, and ESCO candidate-frame classification and stage-time summaries across focal planes. The code does not implement zona pellucida scoring, fragmentation scoring, temporal embryo tracking, or persistent detection identity. Classifier labels include an `Error Image` class.

## 5. Exact class names and mappings

Stage/grade classifier class list, in index order (as defined by `class_names`):

0. `8 Cells Grade A`
1. `8 Cells Grade B`
2. `8 Cells Grade C`
3. `Blastocyst Grade A`
4. `Blastocyst Grade B`
5. `Blastocyst Grade C`
6. `Error Image`
7. `Morula Grade A`
8. `Morula Grade B`
9. `Morula Grade C`

Gardner expansion classes: `1 (Morula Stage)`, `2 (Early Blastocyst)`, `3 (Blastocyst)`, `4 (Expanded)`, `5 (Fully Hatched)`. Mapped scores: 4, 6, 8, 9, 10 respectively.

ICM and TE classes: `A (Good)`, `B (Fair)`, `C (Poor)`, `Not Available`. Both map to scores 10, 7, 4, 0 respectively. Each head selects `argmax` of its logits. `Not Available` in either ICM or TE makes combined `morphology_score` null; otherwise the score is the arithmetic mean of expansion, ICM, and TE scores, rounded to two decimal places. `short_code` concatenates the first characters of the selected labels.

For ESCO, stage parsing recognizes exactly `8 Cells`, `Morula`, and `Blastocyst`; grade is taken from the final token. Grade preference is A then B then C.

## 6. ICM implementation

`MultiTaskModel.icm_head` is a linear output head sized to four ICM labels. The shared ConvNeXtV2 Base features feed it. Inference takes `argmax`, maps through `icm_labels`, and uses the score mapping above. This is a learned classifier head; source inspection does not show a separately segmented or localized ICM region, region-specific crop, confidence threshold, or calibration. The head is only operational when Gardner loads successfully.

## 7. TE/TM implementation

The source names the head `te_head` and the output key/labels `te`; it represents trophectoderm (TE), not a separately named “TM” model. It is a linear head on shared ConvNeXtV2 Base features with four classes and the same A/B/C/Not Available mapping and scores as ICM. No separate TM class, head, segmentation, localization, or independent confidence handling was found.

## 8. Blastocyst / Gardner grading implementation

The ten-class ConvNeXt predicts stage and coarse grade. If its predicted class contains “Blastocyst,” the API additionally runs the Gardner multi-task model. Gardner predicts expansion, ICM, and TE jointly from a resized/center-cropped image; labels are mapped to scores and a simple mean is reported unless ICM or TE is `Not Available`. This is not a conventional single categorical full Gardner-grade classifier in the inspected code. No Gardner output is attached to non-blastocyst predictions.

## 9. Fragmentation implementation

No fragmentation output exists in the currently configured `embryo_gpu/main.py` production path. **CONFIRMED FROM INVENTORY:** older `/home/user_garbha/garbhaai_dev/JSON_Backend_1` contains `best_fragment.pt` and `Embryo_detection/detect/train2/best_fragment.pt`, and the training output tree contains `FragmentationYoloV8Weights.pt`. This establishes an experimental/legacy artifact trail, not a production endpoint or demonstrated fragmentation prediction. Training dataset, class schema, architecture details beyond the artifact name, metrics and runtime use are **NOT VERIFIED** in this continuation.

## 10. Zona implementation

No zona output exists in the current `embryo_gpu` service. **CONFIRMED FROM INVENTORY:** `/home/user_garbha/roboflow_test/embryo_zona_polygon_preview.jpg` is present, indicating a zona-labelled polygon preview/annotation experiment may exist; exact source dataset, annotation format/count, training, metrics, and deployed model are **NOT VERIFIED**. A matching production implementation was not found in the inspected production source.

## 11. Morphokinetics implementation

`esco_focal_plane_morphokinetics.py` registers `POST /esco/focal-plane-morphokinetics` and `POST /esco/all-focal-planes-morphokinetics`. They fetch ESCO candidate frames through a configured bridge, classify them with the stage/grade classifier, choose best frames by stage/grade/confidence and time ordering, and calculate `t8`, `tM`, `tB`, `d_8_M`, `d_M_B`, and total elapsed time where timestamps permit. Stage-to-day mapping is 8 Cells→Day 3, Morula→Day 4, Blastocyst→Day 5. The multi-plane route also selects a recommended focal plane and returns per-plane results. It analyzes supplied candidate images/timestamps; it does not decode a video or independently observe cell divisions. Thus event times depend on candidate generation and timestamp fidelity.

## 12. API endpoints identified

- `GET /health` — unprotected status/device/model and detector configuration summary.
- `POST /detect` — API-key protected single-image localization and crops; Hough circles first, Mask R-CNN fallback.
- `POST /predict` — protected single-image class, confidence, Grad-CAM and optional Gardner morphology.
- `POST /predict_batch` — protected parallel image/embryo-ID classification with optional morphology.
- `POST /classify_frames` — protected batched frame classes, confidence, and full softmax vectors.
- `POST /gradcam_frame` — protected frame-resolution Grad-CAM overlay.
- `POST /esco/focal-plane-morphokinetics` — protected ESCO single-plane timing analysis.
- `POST /esco/all-focal-planes-morphokinetics` — protected cross-plane timing/recommendation analysis.

ESCO routes require bridge configuration/credentials. Exact deployment URLs and current credentials were not inspected or verified.

## 13. Training information found

No training script for the specific deployed ConvNeXt/Gardner checkpoint was confirmed by inspected production source. Home inventory does show `Untitled*.ipynb`, `Embryo_day2,4.ipynb`, `testing_embryo_models.ipynb`, `efficientnet_optuna.ipynb`, `efficientnet_optuna_trial_5to10.ipynb`, `MaskRCNN 50 Head_Class.ipynb`, `Transfer Learning Mask R CNN101 to Mask R CNN 50.ipynb`, `/home/user_garbha/Documents/BlastocystModelling.ipynb`, `BlastocystModellingFinalized.ipynb`, and `/home/user_garbha/Documents/MultimodelTraining.ipynb`. These names establish notebook resources, not their actual contents or causal relationship to production weights. The `embryoscore/python_pipelines/D_ModelBuilding_pipeline/` contains `train_head.py`, `train_head_bkp.py`, and `transfer_learning.py`, but it is an embryo-score feature pipeline and a link to the image models is **NOT VERIFIED**. `best_MAP_45%.pt` filename is not metric proof.

## 14. Metrics found

No quality metrics for the deployed image classifier/Gardner head were verified. Separate detection experiments elsewhere contain evaluation artifacts, including `/home/user_garbha/garbha_embryo_transfer_learning_runs/best_original_embryo_transfer_50epochs_20260806_120428/{results.csv,results.png,BoxPR_curve.png,BoxF1_curve.png,BoxP_curve.png,BoxR_curve.png,confusion_matrix.png,confusion_matrix_normalized.png,args.yaml}` and `/home/user_garbha/Adv_Model_Training/maskrcnn_resnet50_fpn_v2_optuna/inference_outputs/count_good_maskrcnn_best_trial3_epoch24/COUNT_GOOD_maskrcnn_inference_summary.json`. These artifacts are not proven to evaluate the production classifier or detector; exact numeric metrics were not transcribed here. A sperm-only `maskrcnn_sperm_only_map_metrics.csv` exists at home root and is excluded from embryo model claims. Runtime thresholds and scores are not model quality metrics. Production detection defaults: score 0.55, mask 0.50, mask area 15–100000 pixels, NMS IoU 0.45. Classifier `/predict` and batch confidence is top-1 softmax with a 0.70 floor; `/classify_frames` returns raw top-1 confidence and full softmax. The floor is not accuracy or calibration.

## 15. Unknown or unverified items

- Whether the current deployment has successfully loaded the configured weights; historical logs show Gardner failures, while current process state was not checked.
- The exact Mask R-CNN checkpoint location resolves to an external absolute path in the checked-in config; its availability/content is not established here.
- Architecture, labels, training provenance, and intended selection status of `best_original.pt`, `best_MAP_45%.pt`, and duplicate artifacts under `3 pt files/`.
- Dataset identity/size, patient/site representation, train/validation/test split, leakage controls, annotation protocol, and class balance.
- All predictive quality, clinical validation, uncertainty/calibration, subgroup performance, and external validation metrics.
- Actual runtime package versions, hardware, CUDA availability, service health, and endpoint behavior for this checkout.
- Correctness of Hough/Mask R-CNN detection and Gardner predictions on unseen images; no tests or evaluation artifacts were inspected.
- Whether ESCO candidate timestamps are wall-clock hours or another time unit for every deployment; source uses numeric `time` ordering and differences.
- Fragmentation and zona capability outside this inspected repository scope.

## 16. Audit document path

`/home/user_garbha/Garbha_prodapp/embryo_gpu/EMBRYO_AI_SYSTEM_AUDIT.md`

## 17. Project file modification statement

No existing project file was modified. This audit document is the only file created by this continuation. Package installation and model execution were not performed.
