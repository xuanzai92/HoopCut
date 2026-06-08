import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
ShotAttribution = Dict[str, object]

TRACKER_IDENTITY_THRESHOLD = 0.52
TRACKER_HISTOGRAM_FLOOR = 0.34
SEARCH_MATCH_THRESHOLD = 0.58
SEARCH_HISTOGRAM_FLOOR = 0.42
REVALIDATION_SWITCH_DELTA = 0.08


def _clamp(value: float, minimum: int, maximum: int) -> int:
    return int(min(max(value, minimum), maximum))


def _normalize_bbox(bbox: BBox, frame_width: int, frame_height: int) -> Optional[BBox]:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None

    x = _clamp(x, 0, max(frame_width - 1, 0))
    y = _clamp(y, 0, max(frame_height - 1, 0))
    width = _clamp(width, 1, frame_width - x)
    height = _clamp(height, 1, frame_height - y)
    if width <= 0 or height <= 0:
        return None

    return (x, y, width, height)


def _bbox_area(bbox: Optional[BBox]) -> int:
    if bbox is None:
        return 0
    return max(bbox[2], 0) * max(bbox[3], 0)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    return (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)


def _bbox_iou(first: Optional[BBox], second: Optional[BBox]) -> float:
    if first is None or second is None:
        return 0.0

    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(inter_x2 - inter_x1, 0)
    inter_h = max(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    union = _bbox_area(first) + _bbox_area(second) - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def _bbox_scale_similarity(first: Optional[BBox], second: Optional[BBox]) -> float:
    if first is None or second is None:
        return 0.0

    first_area = max(_bbox_area(first), 1)
    second_area = max(_bbox_area(second), 1)
    ratio = min(first_area, second_area) / max(first_area, second_area)
    return float(max(0.0, min(1.0, ratio)))


def _crop_from_bbox(frame, bbox: BBox):
    x, y, width, height = bbox
    return frame[y:y + height, x:x + width]


def _point_in_bbox(point: Tuple[int, int], bbox: BBox) -> bool:
    px, py = point
    x, y, width, height = bbox
    return x <= px <= x + width and y <= py <= y + height


def _expand_bbox(bbox: BBox, horizontal_ratio: float = 0.35, vertical_ratio: float = 0.25) -> BBox:
    x, y, width, height = bbox
    extra_width = int(round(width * horizontal_ratio))
    extra_height = int(round(height * vertical_ratio))
    return (
        x - extra_width,
        y - extra_height,
        width + extra_width * 2,
        height + extra_height * 2,
    )


def _distance_to_bbox(point: Tuple[int, int], bbox: BBox) -> float:
    px, py = point
    x, y, width, height = bbox
    dx = max(x - px, 0, px - (x + width))
    dy = max(y - py, 0, py - (y + height))
    return math.sqrt(dx ** 2 + dy ** 2)


def _compute_histogram(image) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
    if hist is None:
        return None

    hist = hist.astype("float32")
    cv2.normalize(hist, hist)
    return hist


def _hist_similarity(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 0.0

    distance = cv2.compareHist(first, second, cv2.HISTCMP_BHATTACHARYYA)
    return float(max(0.0, 1.0 - distance))


def _build_template(image) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width < 12 or height < 12:
        return None

    scale = min(64.0 / max(width, 1), 128.0 / max(height, 1), 1.0)
    resized_width = max(16, int(round(width * scale)))
    resized_height = max(24, int(round(height * scale)))
    return cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)


def _collect_target_ball_samples(
    ball_positions,
    tracker: "TargetPlayerTracker",
    frame_start: int,
    frame_end: int,
    horizontal_ratio: float = 0.35,
    vertical_ratio: float = 0.25,
    max_gap: int = 10,
):
    samples = []

    for position in ball_positions:
        point, frame_index = position[0], position[1]
        if frame_index < frame_start or frame_index > frame_end:
            continue

        tracked_bbox = tracker.get_box_at_frame(frame_index, max_gap=max_gap)
        if tracked_bbox is None:
            continue

        expanded_bbox = _expand_bbox(
            tracked_bbox,
            horizontal_ratio=horizontal_ratio,
            vertical_ratio=vertical_ratio,
        )
        inside = _point_in_bbox(point, expanded_bbox)
        min_distance = _distance_to_bbox(point, tracked_bbox)
        bbox_scale = max(tracked_bbox[2], tracked_bbox[3], 1)
        distance_score = max(0.0, 1.0 - (min_distance / max(bbox_scale * 0.9, 1.0)))
        proximity_score = 1.0 if inside else round(distance_score, 3)

        samples.append({
            "frame": frame_index,
            "inside": inside,
            "score": proximity_score,
        })

    return samples


class TargetPlayerTracker:
    """Tracks the user-selected player while rejecting identity drift."""

    _hog_detector = None

    def __init__(
        self,
        selection_box: Dict[str, int],
        max_missing_frames: int = 12,
        start_frame: int = 0,
        start_time: float = 0.0,
        revalidate_interval: int = 10,
        reacquire_interval: int = 5,
    ):
        self.selection_box = selection_box
        self.max_missing_frames = max_missing_frames
        self.start_frame = max(int(start_frame), 0)
        self.start_time = max(float(start_time), 0.0)
        self.revalidate_interval = max(int(revalidate_interval), 4)
        self.reacquire_interval = max(int(reacquire_interval), 2)

        self.tracker = None
        self.tracker_type = self._get_tracker_name()

        self.current_bbox: Optional[BBox] = None
        self.last_bbox: Optional[BBox] = None
        self.initial_bbox: Optional[BBox] = None
        self.history: List[Dict] = []

        self.reference_hist: Optional[np.ndarray] = None
        self.adaptive_hist: Optional[np.ndarray] = None
        self.reference_template: Optional[np.ndarray] = None

        self.active_frames = 0
        self.total_frames = 0
        self.missing_frames = 0
        self.reacquired_count = 0
        self.guarded_switches = 0
        self.lost_frames = 0
        self.latest_status = "idle"
        self.last_reacquire_frame = -1

    @staticmethod
    def _tracker_factories():
        legacy = getattr(cv2, "legacy", None)
        return [
            ("CSRT", getattr(cv2, "TrackerCSRT_create", None)),
            ("KCF", getattr(cv2, "TrackerKCF_create", None)),
            ("MIL", getattr(cv2, "TrackerMIL_create", None)),
            ("CSRT", getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None),
            ("KCF", getattr(legacy, "TrackerKCF_create", None) if legacy is not None else None),
            ("MIL", getattr(legacy, "TrackerMIL_create", None) if legacy is not None else None),
        ]

    @classmethod
    def _get_tracker_name(cls) -> str:
        for name, factory in cls._tracker_factories():
            if callable(factory):
                return name
        raise RuntimeError("当前 OpenCV 版本不支持可用的目标跟踪器")

    @classmethod
    def _create_tracker(cls):
        for _, factory in cls._tracker_factories():
            if callable(factory):
                return factory()
        raise RuntimeError("当前 OpenCV 版本不支持可用的目标跟踪器")

    @classmethod
    def _get_hog_detector(cls):
        if cls._hog_detector is None:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            cls._hog_detector = hog
        return cls._hog_detector

    def _scale_selection_box(self, frame_shape) -> Optional[BBox]:
        frame_height, frame_width = frame_shape[:2]
        source_width = max(int(self.selection_box["frameWidth"]), 1)
        source_height = max(int(self.selection_box["frameHeight"]), 1)
        scale_x = frame_width / source_width
        scale_y = frame_height / source_height

        bbox = (
            int(round(self.selection_box["x"] * scale_x)),
            int(round(self.selection_box["y"] * scale_y)),
            int(round(self.selection_box["width"] * scale_x)),
            int(round(self.selection_box["height"] * scale_y)),
        )
        return _normalize_bbox(bbox, frame_width, frame_height)

    def _append_history(
        self,
        frame_index: int,
        bbox: Optional[BBox],
        visible: bool,
        status: str,
        confidence: float = 0.0,
        source: str = "tracker",
    ):
        self.total_frames += 1
        if visible and bbox is not None:
            self.active_frames += 1
        else:
            self.lost_frames += 1

        record = {
            "frame": frame_index,
            "bbox": bbox,
            "visible": visible,
            "status": status,
            "confidence": round(float(confidence), 3),
            "source": source,
        }
        self.history.append(record)
        if len(self.history) > 240:
            self.history.pop(0)

        self.latest_status = status
        return record

    def _motion_score(self, bbox: BBox, anchor_bbox: Optional[BBox]) -> float:
        if anchor_bbox is None:
            anchor_bbox = self.initial_bbox

        if anchor_bbox is None:
            return 0.5

        center_x, center_y = _bbox_center(bbox)
        anchor_center_x, anchor_center_y = _bbox_center(anchor_bbox)
        anchor_span = max(math.sqrt(_bbox_area(anchor_bbox)), 1.0)
        center_distance = math.sqrt((center_x - anchor_center_x) ** 2 + (center_y - anchor_center_y) ** 2)
        position_score = max(0.0, 1.0 - (center_distance / (anchor_span * 1.8)))

        iou_score = _bbox_iou(bbox, anchor_bbox)
        scale_score = _bbox_scale_similarity(bbox, anchor_bbox)
        return float(min(1.0, position_score * 0.45 + iou_score * 0.25 + scale_score * 0.30))

    def _extract_appearance(self, frame, bbox: BBox):
        crop = _crop_from_bbox(frame, bbox)
        return _compute_histogram(crop), _build_template(crop)

    def _combined_hist_similarity(self, candidate_hist: Optional[np.ndarray]) -> float:
        if candidate_hist is None:
            return 0.0

        reference_score = _hist_similarity(self.reference_hist, candidate_hist)
        adaptive_score = _hist_similarity(self.adaptive_hist, candidate_hist)
        return float(reference_score * 0.7 + adaptive_score * 0.3)

    def _update_adaptive_hist(self, candidate_hist: Optional[np.ndarray]):
        if candidate_hist is None:
            return

        if self.adaptive_hist is None:
            self.adaptive_hist = candidate_hist.copy()
            return

        self.adaptive_hist = cv2.addWeighted(self.adaptive_hist, 0.85, candidate_hist, 0.15, 0.0)
        cv2.normalize(self.adaptive_hist, self.adaptive_hist)

    def _score_tracker_candidate(self, frame, bbox: BBox) -> Tuple[float, float]:
        candidate_hist, _ = self._extract_appearance(frame, bbox)
        hist_score = self._combined_hist_similarity(candidate_hist)
        anchor_bbox = self.current_bbox or self.last_bbox or self.initial_bbox
        motion_score = self._motion_score(bbox, anchor_bbox)
        combined_score = hist_score * 0.7 + motion_score * 0.3
        return float(hist_score), float(combined_score)

    def _reinitialize_tracker(self, frame, bbox: BBox):
        self.tracker = self._create_tracker()
        self.tracker.init(frame, bbox)
        self.current_bbox = bbox
        self.last_bbox = bbox

    def _build_search_region(self, frame_shape, anchor_bbox: BBox) -> Optional[Tuple[int, int, int, int]]:
        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = anchor_bbox
        search_width = int(round(width * 2.8))
        search_height = int(round(height * 2.5))
        center_x, center_y = _bbox_center(anchor_bbox)

        search_x = int(round(center_x - search_width / 2))
        search_y = int(round(center_y - search_height / 2))
        search_bbox = _normalize_bbox((search_x, search_y, search_width, search_height), frame_width, frame_height)
        return search_bbox

    def _detect_people_candidates(self, frame, search_bbox: BBox) -> List[Dict]:
        search_crop = _crop_from_bbox(frame, search_bbox)
        if search_crop.size == 0:
            return []

        hog = self._get_hog_detector()
        rects, weights = hog.detectMultiScale(
            search_crop,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        candidates = []
        for rect, weight in zip(rects, weights):
            x, y, width, height = [int(value) for value in rect]
            normalized_bbox = _normalize_bbox(
                (search_bbox[0] + x, search_bbox[1] + y, width, height),
                frame.shape[1],
                frame.shape[0],
            )
            if normalized_bbox is None:
                continue

            candidate_hist, _ = self._extract_appearance(frame, normalized_bbox)
            hist_score = self._combined_hist_similarity(candidate_hist)
            motion_score = self._motion_score(normalized_bbox, self.last_bbox or self.current_bbox or self.initial_bbox)
            detector_confidence = min(max(float(weight), 0.0), 2.0) / 2.0
            score = hist_score * 0.50 + motion_score * 0.25 + detector_confidence * 0.25

            if hist_score < SEARCH_HISTOGRAM_FLOOR or score < SEARCH_MATCH_THRESHOLD:
                continue

            candidates.append({
                "bbox": normalized_bbox,
                "score": float(score),
                "histScore": float(hist_score),
                "source": "person-detector",
            })

        return candidates

    def _search_nearby_target(self, frame, anchor_bbox: Optional[BBox]) -> Optional[Dict]:
        if self.reference_template is None:
            template_enabled = False
        else:
            template_enabled = True

        if anchor_bbox is None:
            anchor_bbox = self.last_bbox or self.initial_bbox
        if anchor_bbox is None:
            return None

        search_bbox = self._build_search_region(frame.shape, anchor_bbox)
        if search_bbox is None:
            return None

        search_crop = _crop_from_bbox(frame, search_bbox)
        if search_crop.size == 0:
            return None

        candidates = self._detect_people_candidates(frame, search_bbox)

        if template_enabled:
            search_gray = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
            template = self.reference_template
            for scale in (0.85, 1.0, 1.15):
                target_width = max(16, int(round(template.shape[1] * scale)))
                target_height = max(24, int(round(template.shape[0] * scale)))

                if target_width >= search_gray.shape[1] or target_height >= search_gray.shape[0]:
                    continue

                resized_template = cv2.resize(
                    template,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                response = cv2.matchTemplate(search_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_value, _, max_location = cv2.minMaxLoc(response)

                candidate_bbox = _normalize_bbox(
                    (
                        search_bbox[0] + int(max_location[0]),
                        search_bbox[1] + int(max_location[1]),
                        target_width,
                        target_height,
                    ),
                    frame.shape[1],
                    frame.shape[0],
                )
                if candidate_bbox is None:
                    continue

                candidate_hist, _ = self._extract_appearance(frame, candidate_bbox)
                hist_score = self._combined_hist_similarity(candidate_hist)
                motion_score = self._motion_score(candidate_bbox, anchor_bbox)
                combined_score = float(max_value) * 0.45 + hist_score * 0.40 + motion_score * 0.15

                if hist_score < SEARCH_HISTOGRAM_FLOOR or combined_score < SEARCH_MATCH_THRESHOLD:
                    continue

                candidates.append({
                    "bbox": candidate_bbox,
                    "score": float(combined_score),
                    "histScore": float(hist_score),
                    "source": "template-search",
                })

        if not candidates:
            return None

        return max(candidates, key=lambda item: item["score"])

    def initialize(self, frame, frame_index: int = 0) -> Optional[Dict]:
        initial_bbox = self._scale_selection_box(frame.shape)
        if initial_bbox is None:
            return None

        initial_hist, initial_template = self._extract_appearance(frame, initial_bbox)
        self.reference_hist = initial_hist
        self.adaptive_hist = initial_hist.copy() if initial_hist is not None else None
        self.reference_template = initial_template
        self.initial_bbox = initial_bbox
        self.missing_frames = 0

        self._reinitialize_tracker(frame, initial_bbox)
        return self._append_history(
            frame_index,
            initial_bbox,
            True,
            status="initialized",
            confidence=1.0,
            source="selection",
        )

    def _should_revalidate(self, frame_index: int) -> bool:
        if self.current_bbox is None:
            return False
        if frame_index == self.start_frame:
            return True
        return (frame_index - self.start_frame) % self.revalidate_interval == 0

    def _should_reacquire(self, frame_index: int) -> bool:
        if self.last_reacquire_frame < 0:
            return True
        return frame_index - self.last_reacquire_frame >= self.reacquire_interval

    def update(self, frame, frame_index: int) -> Optional[Dict]:
        if self.initial_bbox is None:
            return self.initialize(frame, frame_index)

        tracker_candidate = None
        tracker_hist_score = 0.0
        tracker_score = 0.0

        if self.current_bbox is not None and self.tracker is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                normalized_bbox = _normalize_bbox(
                    (
                        int(round(bbox[0])),
                        int(round(bbox[1])),
                        int(round(bbox[2])),
                        int(round(bbox[3])),
                    ),
                    frame.shape[1],
                    frame.shape[0],
                )
                if normalized_bbox is not None:
                    tracker_hist_score, tracker_score = self._score_tracker_candidate(frame, normalized_bbox)
                    if (
                        tracker_hist_score >= TRACKER_HISTOGRAM_FLOOR
                        and tracker_score >= TRACKER_IDENTITY_THRESHOLD
                    ):
                        tracker_candidate = normalized_bbox
                    else:
                        self.guarded_switches += 1

        if tracker_candidate is not None:
            status = "tracking"
            self.current_bbox = tracker_candidate
            self.last_bbox = tracker_candidate
            self.missing_frames = 0
            if self._should_revalidate(frame_index):
                recalibrated_candidate = self._search_nearby_target(frame, tracker_candidate)
                if (
                    recalibrated_candidate is not None
                    and recalibrated_candidate["score"] >= tracker_score + REVALIDATION_SWITCH_DELTA
                ):
                    tracker_candidate = recalibrated_candidate["bbox"]
                    tracker_score = float(recalibrated_candidate["score"])
                    self._reinitialize_tracker(frame, tracker_candidate)
                    status = "revalidated"
                else:
                    self._reinitialize_tracker(frame, tracker_candidate)
                candidate_hist, _ = self._extract_appearance(frame, tracker_candidate)
                self._update_adaptive_hist(candidate_hist)
            return self._append_history(
                frame_index,
                tracker_candidate,
                True,
                status=status,
                confidence=tracker_score,
                source="tracker",
            )

        search_candidate = None
        if self._should_reacquire(frame_index):
            self.last_reacquire_frame = frame_index
            search_candidate = self._search_nearby_target(frame, self.last_bbox or self.current_bbox or self.initial_bbox)

        if search_candidate is not None:
            reacquired_bbox = search_candidate["bbox"]
            self._reinitialize_tracker(frame, reacquired_bbox)
            self.missing_frames = 0
            self.reacquired_count += 1
            candidate_hist, _ = self._extract_appearance(frame, reacquired_bbox)
            self._update_adaptive_hist(candidate_hist)
            return self._append_history(
                frame_index,
                reacquired_bbox,
                True,
                status="reacquired",
                confidence=search_candidate["score"],
                source=str(search_candidate["source"]),
            )

        self.current_bbox = None
        self.missing_frames += 1
        return self._append_history(
            frame_index,
            self.last_bbox,
            False,
            status="lost",
            confidence=max(tracker_score, tracker_hist_score),
            source="guard",
        )

    def get_box_at_frame(self, frame_index: int, max_gap: int = 12) -> Optional[BBox]:
        best_record = None
        best_gap = None

        for record in reversed(self.history):
            bbox = record.get("bbox")
            if bbox is None or not record.get("visible"):
                continue

            gap = abs(frame_index - record["frame"])
            if gap > max_gap:
                continue

            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_record = record

            if gap == 0:
                break

        if best_record is None:
            return None

        return best_record["bbox"]

    def get_summary(self) -> Dict:
        coverage = round(self.active_frames / self.total_frames, 3) if self.total_frames else 0.0
        return {
            "enabled": True,
            "trackerType": self.tracker_type,
            "activeFrames": self.active_frames,
            "totalFrames": self.total_frames,
            "coverage": coverage,
            "missingFrames": self.missing_frames,
            "lostFrames": self.lost_frames,
            "reacquiredCount": self.reacquired_count,
            "guardedSwitches": self.guarded_switches,
            "latestStatus": self.latest_status,
            "startFrame": self.start_frame,
            "startTime": round(self.start_time, 3),
        }


def classify_shot_owner(
    ball_positions,
    tracker: Optional[TargetPlayerTracker],
    shot_release_frame: int,
    lookback_frames: int = 12,
) -> Tuple[str, float, bool]:
    attribution = classify_shot_involvement(
        ball_positions,
        tracker,
        shot_release_frame,
        shooter_lookback_frames=lookback_frames,
    )
    return (
        str(attribution.get("owner", "unknown")),
        float(attribution.get("owner_confidence", 0.0)),
        bool(attribution.get("target_visible", False)),
    )


def classify_shot_involvement(
    ball_positions,
    tracker: Optional[TargetPlayerTracker],
    shot_release_frame: int,
    shooter_lookback_frames: int = 12,
    assist_lookback_frames: int = 72,
    assist_release_gap_frames: int = 8,
    assist_max_gap_frames: int = 45,
) -> ShotAttribution:
    if tracker is None:
        return {
            "owner": "unknown",
            "owner_confidence": 0.0,
            "target_visible": False,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
        }

    release_bbox = tracker.get_box_at_frame(shot_release_frame)
    assist_window_end = shot_release_frame - assist_release_gap_frames
    release_samples = _collect_target_ball_samples(
        ball_positions,
        tracker,
        shot_release_frame - shooter_lookback_frames,
        shot_release_frame,
    )
    assist_samples = []
    if assist_window_end >= shot_release_frame - assist_lookback_frames:
        assist_samples = _collect_target_ball_samples(
            ball_positions,
            tracker,
            shot_release_frame - assist_lookback_frames,
            assist_window_end,
            horizontal_ratio=0.45,
            vertical_ratio=0.35,
            max_gap=12,
        )

    target_visible = release_bbox is not None or bool(release_samples) or bool(assist_samples)
    if not target_visible:
        return {
            "owner": "unknown",
            "owner_confidence": 0.0,
            "target_visible": False,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
        }

    owner = "unknown"
    owner_confidence = 0.0
    release_score = 0.0

    if release_samples:
        inside_ratio = sum(1 for sample in release_samples if sample["inside"]) / len(release_samples)
        best_score = max(sample["score"] for sample in release_samples)
        release_score = float(release_samples[-1]["score"])
        owner_confidence = round(
            min(1.0, inside_ratio * 0.45 + best_score * 0.35 + release_score * 0.20),
            3,
        )
        if owner_confidence >= 0.45:
            owner = "target"

    if owner == "target":
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": True,
            "highlight_role": "score",
            "highlight_confidence": owner_confidence,
        }

    if assist_window_end < shot_release_frame - assist_lookback_frames:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
        }

    strong_control_samples = [sample for sample in assist_samples if sample["score"] >= 0.55]

    if len(strong_control_samples) < 2:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
        }

    last_target_control_frame = int(strong_control_samples[-1]["frame"])
    frame_gap = shot_release_frame - last_target_control_frame

    if frame_gap < assist_release_gap_frames or frame_gap > assist_max_gap_frames:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
        }

    best_control_score = max(float(sample["score"]) for sample in strong_control_samples)
    sustained_control_score = min(1.0, len(strong_control_samples) / 3.0)
    gap_score = max(
        0.0,
        1.0 - ((frame_gap - assist_release_gap_frames) / max(assist_max_gap_frames - assist_release_gap_frames, 1)),
    )
    separation_score = 1.0 if not release_samples else max(0.0, 1.0 - release_score)

    highlight_confidence = round(
        min(
            1.0,
            best_control_score * 0.50
            + sustained_control_score * 0.25
            + gap_score * 0.15
            + separation_score * 0.10,
        ),
        3,
    )

    highlight_role = "assist" if highlight_confidence >= 0.5 else "none"
    return {
        "owner": owner,
        "owner_confidence": owner_confidence,
        "target_visible": target_visible,
        "highlight_role": highlight_role,
        "highlight_confidence": highlight_confidence if highlight_role != "none" else 0.0,
    }


def draw_target_bbox(
    frame,
    bbox: Optional[BBox],
    visible: bool = True,
    status: str = "tracking",
    confidence: float = 0.0,
):
    if bbox is None:
        return

    x, y, width, height = bbox
    color = (74, 222, 128) if visible else (37, 99, 235)
    label = "Target" if visible else "Target Lost"
    if status == "reacquired":
        color = (0, 191, 255)
        label = "Target Reacquired"
    elif status == "revalidated":
        color = (80, 180, 255)
        label = "Target Revalidated"
    elif status == "lost":
        color = (0, 165, 255)

    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    label_text = label if confidence <= 0 else f"{label} {confidence:.2f}"
    cv2.putText(
        frame,
        label_text,
        (x, max(18, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )
