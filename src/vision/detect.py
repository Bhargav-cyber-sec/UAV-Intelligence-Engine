"""
YOLO detection + tracking (Week 2 — reliability rewrite).

Improvements over the initial version:
  1. Track-continuity ghost reconciliation: a dropped track re-appearing within
     ghost_window_frames of the same class and within max_motion_radius_px is
     merged into a single logical_track_id rather than scored as two events.
  2. Noise-floor filtering: tracks that survive fewer than min_track_frames
     frames are written to detections.json but flagged noise=True and are
     excluded from tracks.json (and therefore from Week 3 scoring).
  3. Richer Track dataclass: adds max_confidence, duration_s, bbox_trajectory
     (first/last bbox), area_px_mean so Week 3 event scoring has everything it
     needs without re-reading detections.json.
  4. SAHI mode fixed: tile detections within the same frame are merged via NMS,
     then the *merged* detections are fed to a lightweight IOU/distance tracker
     that assigns persistent logical_track_ids across frames (instead of every
     tile det getting its own unique id).
  5. Track buffer tuned for aerial footage: persist=True with a larger internal
     ByteTrack track_buffer (configurable via ontology.yaml detection.track_buffer)
     so fast-moving aircraft don't lose identity during brief occlusions.

Outputs (Mission Evidence Package):
  - detections.json  : per-frame detections (includes noise flag, logical_track_id)
  - tracks.json      : one entry per logical track, noise tracks excluded
  - detection_summary.json : aggregate stats (class counts, coverage, noise rate)
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from src.config import Config, load_config


class DetectionError(RuntimeError):
    """Raised when YOLO weights or input frames are missing."""


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    frame_index: int
    frame_file: str
    track_id: int | None          # raw ByteTrack id (may be None for SAHI merged dets)
    logical_track_id: int | None  # reconciled id after ghost-merging
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]        # [x1, y1, x2, y2] in sampled-frame pixel space
    noise: bool = False           # True if track survived fewer than min_track_frames


@dataclass
class Track:
    logical_track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    frame_count: int              # number of frames where detected
    duration_s: float             # (last_frame - first_frame) / detection_fps
    mean_confidence: float
    max_confidence: float
    first_bbox: list[float]       # bbox at first_frame
    last_bbox: list[float]        # bbox at last_frame
    area_px_mean: float           # mean bbox area in sampled-frame pixels


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(cfg: Config, model_path_override: str | Path | None = None):
    from ultralytics import YOLO  # lazy import — only when detection is actually run

    weights_path = Path(model_path_override or cfg.paths["models"]["yolo_weights"])
    if not weights_path.exists():
        raise DetectionError(
            f"YOLO weights not found at {weights_path}. "
            "Place fine-tuned UAV weights (e.g. yolo11_uav_custom.pt) in models/yolo/ "
            "or download yolo11n.pt from Ultralytics."
        )
    return YOLO(str(weights_path))


# ─────────────────────────────────────────────────────────────────────────────
# SAHI sliding-window inference (returns merged per-frame detections with no
# track id — track assignment is done separately by _SimpleIouTracker)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sliced_inference(
    model,
    frame_path: Path,
    conf: float,
    slice_h: int = 640,
    slice_w: int = 640,
    overlap: float = 0.2,
    iou_nms: float = 0.45,
    classes: list[int] | None = None,
) -> list[dict]:
    """Sliding-window SAHI inference on a single frame.

    Returns a list of {class_id, class_name, confidence, bbox_xyxy} dicts
    after cross-tile NMS.  No track ids assigned here.
    """
    import cv2

    img = cv2.imread(str(frame_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    step_y = max(1, int(slice_h * (1.0 - overlap)))
    step_x = max(1, int(slice_w * (1.0 - overlap)))

    def _offsets(dim: int, tile: int, step: int) -> list[int]:
        offs = list(range(0, max(1, dim - tile + 1), step))
        if offs and offs[-1] + tile < dim:
            offs.append(dim - tile)
        return offs or [0]

    raw_boxes: list[list[float]] = []
    scores: list[float] = []
    class_ids: list[int] = []

    for y0 in _offsets(h, slice_h, step_y):
        for x0 in _offsets(w, slice_w, step_x):
            crop = img[y0:y0 + slice_h, x0:x0 + slice_w]
            results = model.predict(crop, conf=conf, classes=classes, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()
                raw_boxes.append([box[0] + x0, box[1] + y0, box[2] + x0, box[3] + y0])
                scores.append(float(boxes.conf[i]))
                class_ids.append(int(boxes.cls[i]))

    if not raw_boxes:
        return []

    # cross-tile NMS
    xywh = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in raw_boxes]
    indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=conf, nms_threshold=iou_nms)
    if len(indices) == 0:
        return []

    flat = indices.flatten() if hasattr(indices, "flatten") else list(indices)
    return [
        {
            "class_id": class_ids[i],
            "class_name": model.names[class_ids[i]],
            "confidence": scores[i],
            "bbox_xyxy": raw_boxes[i],
        }
        for i in flat
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight IOU tracker for SAHI mode (assigns persistent track ids
# across frames using centroid distance + IOU matching — no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _centroid(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class _SimpleIouTracker:
    """Frame-by-frame IOU/centroid tracker used when SAHI mode is active.

    Assigns persistent track ids to merged per-frame SAHI detections.
    Much lighter than ByteTrack — no Kalman filter — but sufficient for
    associating SAHI detections across adjacent sampled frames.
    """

    def __init__(self, iou_thresh: float = 0.25, max_age: int = 5, max_dist_px: float = 120.0):
        self.iou_thresh = iou_thresh
        self.max_age = max_age          # frames a track lives without a match
        self.max_dist_px = max_dist_px
        self._next_id = 1
        self._active: dict[int, dict] = {}  # track_id -> {box, class_id, age}

    def update(self, dets: list[dict]) -> list[tuple[int, dict]]:
        """Match dets to active tracks.  Returns [(track_id, det), ...]."""
        if not dets:
            for tid in list(self._active):
                self._active[tid]["age"] += 1
                if self._active[tid]["age"] > self.max_age:
                    del self._active[tid]
            return []

        active_ids = list(self._active)
        assigned: set[int] = set()     # track ids matched this frame
        results: list[tuple[int, dict]] = []

        for det in dets:
            best_tid: int | None = None
            best_score = -1.0

            for tid in active_ids:
                if tid in assigned:
                    continue
                track_box = self._active[tid]["box"]
                iou_val = _iou(det["bbox_xyxy"], track_box)
                cx_d, cy_d = _centroid(det["bbox_xyxy"])
                cx_t, cy_t = _centroid(track_box)
                dist = math.hypot(cx_d - cx_t, cy_d - cy_t)
                if (iou_val >= self.iou_thresh or dist <= self.max_dist_px) and \
                   det["class_id"] == self._active[tid]["class_id"]:
                    score = iou_val - dist / 1000.0  # prefer IOU; penalise distance
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                assigned.add(best_tid)
                self._active[best_tid]["box"] = det["bbox_xyxy"]
                self._active[best_tid]["age"] = 0
                results.append((best_tid, det))
            else:
                new_id = self._next_id
                self._next_id += 1
                self._active[new_id] = {
                    "box": det["bbox_xyxy"],
                    "class_id": det["class_id"],
                    "age": 0,
                }
                results.append((new_id, det))

        # age out unmatched tracks
        for tid in active_ids:
            if tid not in assigned:
                self._active[tid]["age"] += 1
                if self._active[tid]["age"] > self.max_age:
                    del self._active[tid]

        return results


# ─────────────────────────────────────────────────────────────────────────────
# Ghost reconciliation (post-processing, applied to both modes)
# ─────────────────────────────────────────────────────────────────────────────

def _reconcile_ghosts(
    track_agg: dict[int, dict],
    ghost_window_frames: int,
    max_radius_px: float,
) -> dict[int, int]:
    """Return a mapping raw_track_id -> logical_track_id.

    A raw track is merged into an earlier logical track if:
      - same class
      - first_frame of new track is within ghost_window_frames of last_frame of old track
      - centroid of first bbox of new track is within max_radius_px of centroid of last bbox
        of the old track
    """
    # sort by first_frame so we always try to merge into earlier tracks
    sorted_tids = sorted(track_agg, key=lambda t: track_agg[t]["first_frame"])
    logical: dict[int, int] = {tid: tid for tid in sorted_tids}  # identity by default

    for i, tid in enumerate(sorted_tids):
        # find the logical root of this track
        root = logical[tid]
        while logical[root] != root:
            root = logical[root]
        logical[tid] = root

        for earlier in sorted_tids[:i]:
            e_root = logical[earlier]
            while logical[e_root] != e_root:
                e_root = logical[e_root]

            agg_e = track_agg[e_root]
            agg_t = track_agg[tid]

            if agg_e["class_name"] != agg_t["class_name"]:
                continue
            gap = agg_t["first_frame"] - agg_e["last_frame"]
            if not (0 < gap <= ghost_window_frames):
                continue

            last_box = agg_e.get("last_bbox")
            first_box = agg_t.get("first_bbox")
            if last_box and first_box:
                cx_e, cy_e = _centroid(last_box)
                cx_t, cy_t = _centroid(first_box)
                dist = math.hypot(cx_t - cx_e, cy_t - cy_e)
                if dist > max_radius_px:
                    continue

            # merge: point tid -> e_root
            logical[tid] = e_root
            root = e_root
            break

    # path-compress
    for tid in list(logical):
        r = tid
        while logical[r] != r:
            r = logical[r]
        logical[tid] = r

    return logical


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_track(
    frames_dir: str | Path,
    output_dir: str | Path,
    cfg: Config | None = None,
    model_path_override: str | Path | None = None,
) -> tuple[list[Detection], list[Track]]:
    """Run YOLO + tracking over the frame sequence in *frames_dir*.

    Supports:
      - Standard Ultralytics ByteTrack mode (default, best for continuous footage).
      - SAHI tile-slicing mode for high-res footage with very small targets
        (set detection.sahi.enabled: true in ontology.yaml).

    Ghost reconciliation and noise filtering are applied in both modes.
    """
    cfg = cfg or load_config()
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise DetectionError(
            f"No frame_*.jpg found in {frames_dir} — run src.media.sample first."
        )

    model = _load_model(cfg, model_path_override=model_path_override)

    det_cfg = cfg.ontology.get("detection", {})
    conf_thresh         = det_cfg.get("confidence_threshold", 0.25)
    tracker_cfg         = det_cfg.get("tracker", "bytetrack.yaml")
    classes_of_interest = det_cfg.get("classes_of_interest")  # None = all classes
    track_buffer        = det_cfg.get("track_buffer", 30)      # frames to keep lost track alive
    min_track_frames    = det_cfg.get("min_track_frames", 3)   # noise floor

    sahi_cfg  = det_cfg.get("sahi", {})
    use_sahi  = sahi_cfg.get("enabled", False)

    tc_cfg            = cfg.ontology.get("track_continuity", {})
    detection_fps     = cfg.ontology.get("sampling", {}).get("detection_fps", 3)
    ghost_window_s    = tc_cfg.get("ghost_window_s", 2.0)
    ghost_window_frames = max(1, int(ghost_window_s * detection_fps))
    max_radius_px     = tc_cfg.get("max_motion_radius_px", 120)

    raw_detections: list[dict] = []   # raw before logical_id assignment
    track_agg: dict[int, dict] = {}

    sahi_tracker: _SimpleIouTracker | None = None
    if use_sahi:
        sahi_tracker = _SimpleIouTracker(
            iou_thresh=0.25,
            max_age=track_buffer,
            max_dist_px=max_radius_px,
        )

    # ── per-frame inference ──────────────────────────────────────────────────
    for frame_index, frame_path in enumerate(frame_paths):

        if use_sahi:
            merged = _run_sliced_inference(
                model=model,
                frame_path=frame_path,
                conf=conf_thresh,
                slice_h=sahi_cfg.get("slice_height", 640),
                slice_w=sahi_cfg.get("slice_width", 640),
                overlap=sahi_cfg.get("overlap_ratio", 0.2),
                iou_nms=sahi_cfg.get("nms_threshold", 0.45),
                classes=classes_of_interest,
            )
            assigned = sahi_tracker.update(merged)  # type: ignore[union-attr]

            for tid, det in assigned:
                raw_detections.append({
                    "frame_index": frame_index,
                    "frame_file": frame_path.name,
                    "track_id": tid,
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox_xyxy": det["bbox_xyxy"],
                })
                agg = track_agg.setdefault(tid, {
                    "class_name": det["class_name"],
                    "class_id": det["class_id"],
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "first_bbox": det["bbox_xyxy"],
                    "last_bbox": det["bbox_xyxy"],
                    "confidences": [],
                    "areas": [],
                })
                agg["last_frame"] = frame_index
                agg["last_bbox"] = det["bbox_xyxy"]
                agg["confidences"].append(det["confidence"])
                b = det["bbox_xyxy"]
                agg["areas"].append((b[2] - b[0]) * (b[3] - b[1]))

        else:
            results = model.track(
                source=str(frame_path),
                persist=True,
                tracker=tracker_cfg,
                conf=conf_thresh,
                classes=classes_of_interest,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is None or boxes.id is None:
                continue

            for i in range(len(boxes)):
                class_id   = int(boxes.cls[i])
                class_name = model.names[class_id]
                confidence = float(boxes.conf[i])
                xyxy       = [float(v) for v in boxes.xyxy[i].tolist()]
                tid        = int(boxes.id[i])

                raw_detections.append({
                    "frame_index": frame_index,
                    "frame_file": frame_path.name,
                    "track_id": tid,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": xyxy,
                })

                agg = track_agg.setdefault(tid, {
                    "class_name": class_name,
                    "class_id": class_id,
                    "first_frame": frame_index,
                    "last_frame": frame_index,
                    "first_bbox": xyxy,
                    "last_bbox": xyxy,
                    "confidences": [],
                    "areas": [],
                })
                agg["last_frame"] = frame_index
                agg["last_bbox"] = xyxy
                agg["confidences"].append(confidence)
                agg["areas"].append((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))

    # ── ghost reconciliation ──────────────────────────────────────────────────
    logical_map = _reconcile_ghosts(track_agg, ghost_window_frames, max_radius_px)

    # build merged agg for logical tracks
    logical_agg: dict[int, dict] = {}
    for raw_tid, agg in track_agg.items():
        lid = logical_map[raw_tid]
        if lid not in logical_agg:
            logical_agg[lid] = {
                "class_name": agg["class_name"],
                "first_frame": agg["first_frame"],
                "last_frame": agg["last_frame"],
                "first_bbox": agg["first_bbox"],
                "last_bbox": agg["last_bbox"],
                "confidences": list(agg["confidences"]),
                "areas": list(agg["areas"]),
            }
        else:
            la = logical_agg[lid]
            la["last_frame"] = max(la["last_frame"], agg["last_frame"])
            la["first_frame"] = min(la["first_frame"], agg["first_frame"])
            if agg["first_frame"] <= la["first_frame"]:
                la["first_bbox"] = agg["first_bbox"]
            if agg["last_frame"] >= la["last_frame"]:
                la["last_bbox"] = agg["last_bbox"]
            la["confidences"].extend(agg["confidences"])
            la["areas"].extend(agg["areas"])

    # ── build Detection + Track lists ────────────────────────────────────────
    # which logical tracks survive noise floor?
    clean_logical_ids: set[int] = {
        lid
        for lid, la in logical_agg.items()
        if len(la["confidences"]) >= min_track_frames
    }

    all_detections: list[Detection] = []
    for rd in raw_detections:
        raw_tid = rd["track_id"]
        lid = logical_map.get(raw_tid, raw_tid) if raw_tid is not None else None
        noise = lid not in clean_logical_ids if lid is not None else True
        all_detections.append(Detection(
            frame_index=rd["frame_index"],
            frame_file=rd["frame_file"],
            track_id=raw_tid,
            logical_track_id=lid,
            class_id=rd["class_id"],
            class_name=rd["class_name"],
            confidence=rd["confidence"],
            bbox_xyxy=rd["bbox_xyxy"],
            noise=noise,
        ))

    tracks: list[Track] = []
    for lid in sorted(clean_logical_ids):
        la = logical_agg[lid]
        confs = la["confidences"]
        areas = la["areas"]
        duration_s = (la["last_frame"] - la["first_frame"]) / max(detection_fps, 1)
        tracks.append(Track(
            logical_track_id=lid,
            class_name=la["class_name"],
            first_frame=la["first_frame"],
            last_frame=la["last_frame"],
            frame_count=len(confs),
            duration_s=round(duration_s, 3),
            mean_confidence=round(sum(confs) / len(confs), 4),
            max_confidence=round(max(confs), 4),
            first_bbox=la["first_bbox"],
            last_bbox=la["last_bbox"],
            area_px_mean=round(sum(areas) / len(areas), 1),
        ))

    # ── write outputs ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "detections.json", "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in all_detections], f, indent=2)

    with open(output_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in tracks], f, indent=2)

    # summary
    total = len(all_detections)
    noise_count = sum(1 for d in all_detections if d.noise)
    frames_covered = len({d.frame_index for d in all_detections if not d.noise})
    summary = {
        "total_detections": total,
        "clean_detections": total - noise_count,
        "noise_detections": noise_count,
        "noise_rate_pct": round(100 * noise_count / total, 1) if total else 0,
        "total_tracks": len(tracks),
        "frames_with_clean_detections": frames_covered,
        "total_frames_processed": len(frame_paths),
        "detection_fps": detection_fps,
        "ghost_window_frames": ghost_window_frames,
        "min_track_frames": min_track_frames,
        "class_counts": {
            cls: sum(1 for t in tracks if t.class_name == cls)
            for cls in sorted({t.class_name for t in tracks})
        },
    }
    with open(output_dir / "detection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return all_detections, tracks
