import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
ShotAttribution = Dict[str, object]

TRACKER_IDENTITY_THRESHOLD = 0.58
TRACKER_HISTOGRAM_FLOOR = 0.42
TRACKER_TEMPLATE_FLOOR = 0.16
SEARCH_MATCH_THRESHOLD = 0.62
SEARCH_HISTOGRAM_FLOOR = 0.46
SEARCH_TEMPLATE_FLOOR = 0.18
REVALIDATION_SWITCH_DELTA = 0.10
LOCAL_REVIEW_MATCH_THRESHOLD = 0.55
LOCAL_REVIEW_HISTOGRAM_FLOOR = 0.40
LOCAL_REVIEW_STRONG_SCORE = 0.58
PARTIAL_ASSIST_EVIDENCE_THRESHOLD = 0.42
ASSIST_CONFIRM_TOLERANCE = 0.03
RUNTIME_REFERENCE_TEMPLATE_FLOOR = 0.28
ADAPTIVE_REFERENCE_HIST_FLOOR = 0.52
ADAPTIVE_REFERENCE_TEMPLATE_FLOOR = 0.22
TRUSTED_TRACKING_MOTION_FLOOR = 0.82
BALL_ALIGNED_SCALE_FACTORS = (0.72, 0.85, 1.0, 1.18, 1.38)
TEMPLATE_SEARCH_SCALE_FACTORS = (0.78, 0.92, 1.0, 1.12, 1.28)


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


def _template_similarity(
    reference_template: Optional[np.ndarray],
    candidate_template: Optional[np.ndarray],
) -> float:
    if reference_template is None or candidate_template is None:
        return 0.0

    try:
        resized_candidate = cv2.resize(
            candidate_template,
            (reference_template.shape[1], reference_template.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        response = cv2.matchTemplate(resized_candidate, reference_template, cv2.TM_CCOEFF_NORMED)
        if response.size == 0:
            return 0.0
        return float(max(0.0, min(1.0, response[0][0])))
    except cv2.error:
        return 0.0


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


def _collect_ball_points_by_frame(ball_positions, frame_start: int, frame_end: int) -> List[Dict[str, object]]:
    samples_by_frame: Dict[int, Dict[str, object]] = {}

    for position in ball_positions:
        point, frame_index = position[0], int(position[1])
        if frame_index < frame_start or frame_index > frame_end:
            continue

        confidence = float(position[4]) if len(position) > 4 else 0.0
        existing = samples_by_frame.get(frame_index)
        if existing is None or confidence >= float(existing["confidence"]):
            samples_by_frame[frame_index] = {
                "frame": frame_index,
                "point": point,
                "confidence": confidence,
            }

    return [samples_by_frame[frame] for frame in sorted(samples_by_frame)]


def _frame_scores_from_samples(samples: List[Dict[str, object]]) -> Dict[int, float]:
    frame_scores: Dict[int, float] = {}
    for sample in samples:
        frame = int(sample["frame"])
        frame_scores[frame] = max(frame_scores.get(frame, 0.0), float(sample["score"]))
    return frame_scores


def _build_control_windows(
    samples: List[Dict[str, object]],
    strong_threshold: float,
    max_gap_frames: int,
    min_samples: int = 2,
    min_span_frames: int = 2,
) -> List[Dict[str, float]]:
    strong_frames = [
        (frame, score)
        for frame, score in sorted(_frame_scores_from_samples(samples).items())
        if score >= strong_threshold
    ]
    if not strong_frames:
        return []

    windows: List[Dict[str, float]] = []
    current_frames: List[int] = []
    current_scores: List[float] = []

    def flush_window():
        if not current_frames:
            return

        span = current_frames[-1] - current_frames[0]
        if len(current_frames) < min_samples or span < min_span_frames:
            return

        windows.append({
            "start_frame": int(current_frames[0]),
            "end_frame": int(current_frames[-1]),
            "sample_count": int(len(current_frames)),
            "span": int(span),
            "best_score": round(max(current_scores), 3),
            "mean_score": round(sum(current_scores) / len(current_scores), 3),
        })

    for frame, score in strong_frames:
        if current_frames and frame - current_frames[-1] > max_gap_frames:
            flush_window()
            current_frames = []
            current_scores = []

        current_frames.append(int(frame))
        current_scores.append(float(score))

    flush_window()
    return windows


def _control_window_confidence(window: Optional[Dict[str, float]]) -> float:
    if not window:
        return 0.0

    sustained_score = min(1.0, float(window["sample_count"]) / 3.0)
    span_score = min(1.0, float(window["span"]) / 8.0)
    return round(
        min(
            1.0,
            float(window["best_score"]) * 0.42
            + float(window["mean_score"]) * 0.28
            + sustained_score * 0.18
            + span_score * 0.12,
        ),
        3,
    )


def _handoff_confidence(
    samples: List[Dict[str, object]],
    control_window: Optional[Dict[str, float]],
    shot_release_frame: int,
    separation_threshold: float = 0.38,
    strong_threshold: float = 0.55,
    max_release_delay_frames: int = 18,
) -> float:
    if not control_window:
        return 0.0

    control_end_frame = int(control_window["end_frame"])
    post_control_samples = [
        sample
        for sample in sorted(samples, key=lambda item: int(item["frame"]))
        if control_end_frame < int(sample["frame"]) <= shot_release_frame
    ]
    if not post_control_samples:
        return 0.0

    separation_samples = [
        sample for sample in post_control_samples
        if (not bool(sample.get("inside"))) and float(sample.get("score", 0.0)) <= separation_threshold
    ]
    if not separation_samples:
        return 0.0

    first_separation_frame = int(separation_samples[0]["frame"])
    last_separation_frame = int(separation_samples[-1]["frame"])
    release_delay = max(first_separation_frame - control_end_frame, 0)

    strong_retouch = any(
        int(sample["frame"]) > first_separation_frame
        and (
            bool(sample.get("inside"))
            or float(sample.get("score", 0.0)) >= strong_threshold
        )
        for sample in post_control_samples
    )

    quickness_score = max(
        0.0,
        1.0 - (release_delay / max(max_release_delay_frames, 1)),
    )
    separation_density = min(1.0, len(separation_samples) / max(len(post_control_samples), 1))
    persistence_score = min(1.0, (last_separation_frame - first_separation_frame + 1) / 8.0)
    retouch_score = 0.0 if strong_retouch else 1.0

    return round(
        min(
            1.0,
            quickness_score * 0.35
            + separation_density * 0.25
            + persistence_score * 0.15
            + retouch_score * 0.25,
        ),
        3,
    )


def _post_handoff_continuity_confidence(
    samples: List[Dict[str, object]],
    start_frame: int,
    shot_release_frame: int,
    max_tail_gap_frames: int = 10,
    max_internal_gap_frames: int = 16,
) -> float:
    ordered_frames = sorted(
        int(sample["frame"])
        for sample in samples
        if start_frame < int(sample["frame"]) <= shot_release_frame
    )
    if not ordered_frames:
        return 0.0

    tail_gap = max(shot_release_frame - ordered_frames[-1], 0)
    head_gap = max(ordered_frames[0] - start_frame, 0)
    expected_span = max(shot_release_frame - start_frame, 1)
    observed_span = max(ordered_frames[-1] - ordered_frames[0], 0)

    sample_score = min(1.0, len(ordered_frames) / 3.0)
    span_score = min(1.0, observed_span / expected_span)
    tail_score = max(0.0, 1.0 - (tail_gap / max(max_tail_gap_frames, 1)))
    head_score = max(0.0, 1.0 - (head_gap / max(max_internal_gap_frames, 1)))

    if len(ordered_frames) > 1:
        gaps = [
            ordered_frames[index] - ordered_frames[index - 1]
            for index in range(1, len(ordered_frames))
        ]
        max_gap = max(gaps)
        gap_score = max(
            0.0,
            1.0 - (max(max_gap - 1, 0) / max(max_internal_gap_frames, 1)),
        )
    else:
        gap_score = 0.55 if tail_gap <= max_tail_gap_frames // 2 else 0.25

    return round(
        min(
            1.0,
            sample_score * 0.25
            + span_score * 0.20
            + tail_score * 0.35
            + gap_score * 0.15
            + head_score * 0.05,
        ),
        3,
    )


def _build_ball_presence_windows(
    samples: List[Dict[str, object]],
    start_frame: int,
    shot_release_frame: int,
    max_gap_frames: int = 8,
) -> List[Dict[str, int]]:
    ordered_frames = sorted(
        int(sample["frame"])
        for sample in samples
        if start_frame < int(sample["frame"]) <= shot_release_frame
    )
    if not ordered_frames:
        return []

    windows: List[Dict[str, int]] = []
    current_frames: List[int] = []

    def flush_window():
        if not current_frames:
            return
        windows.append({
            "start_frame": int(current_frames[0]),
            "end_frame": int(current_frames[-1]),
            "sample_count": int(len(current_frames)),
            "span": int(current_frames[-1] - current_frames[0]),
        })

    for frame in ordered_frames:
        if current_frames and frame - current_frames[-1] > max_gap_frames:
            flush_window()
            current_frames = []
        current_frames.append(int(frame))

    flush_window()
    return windows


def _terminal_release_window_confidence(
    samples: List[Dict[str, object]],
    start_frame: int,
    shot_release_frame: int,
    max_gap_frames: int = 8,
    max_tail_gap_frames: int = 6,
) -> float:
    windows = _build_ball_presence_windows(
        samples,
        start_frame=start_frame,
        shot_release_frame=shot_release_frame,
        max_gap_frames=max_gap_frames,
    )
    if not windows:
        return 0.0

    terminal_window = windows[-1]
    tail_gap = max(shot_release_frame - int(terminal_window["end_frame"]), 0)
    if tail_gap > max_tail_gap_frames:
        return 0.0

    sample_score = min(1.0, int(terminal_window["sample_count"]) / 3.0)
    span_score = min(1.0, int(terminal_window["span"]) / 8.0)
    tail_score = max(0.0, 1.0 - (tail_gap / max(max_tail_gap_frames, 1)))

    return round(
        min(
            1.0,
            sample_score * 0.45
            + span_score * 0.20
            + tail_score * 0.35,
        ),
        3,
    )


def _receiver_trajectory_confidence(
    samples: List[Dict[str, object]],
    start_frame: int,
    shot_release_frame: int,
    max_gap_frames: int = 8,
    max_tail_gap_frames: int = 6,
    regression_tolerance: float = 10.0,
    max_detour_ratio: float = 2.4,
) -> float:
    ordered_samples = [
        sample
        for sample in sorted(samples, key=lambda item: int(item["frame"]))
        if start_frame < int(sample["frame"]) <= shot_release_frame and sample.get("point") is not None
    ]
    if not ordered_samples:
        return 0.0

    windows = _build_ball_presence_windows(
        ordered_samples,
        start_frame=start_frame,
        shot_release_frame=shot_release_frame,
        max_gap_frames=max_gap_frames,
    )
    if not windows:
        return 0.0

    terminal_window = windows[-1]
    tail_gap = max(shot_release_frame - int(terminal_window["end_frame"]), 0)
    if tail_gap > max_tail_gap_frames:
        return 0.0

    terminal_start_frame = int(terminal_window["start_frame"])
    terminal_end_frame = int(terminal_window["end_frame"])
    terminal_samples = [
        sample
        for sample in ordered_samples
        if terminal_start_frame <= int(sample["frame"]) <= terminal_end_frame
    ]
    if not terminal_samples:
        return 0.0

    terminal_anchor = (
        sum(int(sample["point"][0]) for sample in terminal_samples) / len(terminal_samples),
        sum(int(sample["point"][1]) for sample in terminal_samples) / len(terminal_samples),
    )
    path_samples = [
        sample
        for sample in ordered_samples
        if int(sample["frame"]) <= terminal_end_frame
    ]
    if not path_samples:
        return 0.0

    tail_score = max(0.0, 1.0 - (tail_gap / max(max_tail_gap_frames, 1)))
    terminal_density_score = min(1.0, int(terminal_window["sample_count"]) / 3.0)
    terminal_span_score = min(1.0, int(terminal_window["span"]) / 6.0)

    if len(path_samples) == 1:
        return round(
            min(
                1.0,
                terminal_density_score * 0.55
                + tail_score * 0.30
                + terminal_span_score * 0.15,
            ),
            3,
        )

    path_points = [sample["point"] for sample in path_samples]
    anchor_distances = [
        math.dist((float(point[0]), float(point[1])), terminal_anchor)
        for point in path_points
    ]
    start_distance = max(anchor_distances[0], 1.0)
    terminal_distances = anchor_distances[-len(terminal_samples):]
    terminal_distance = sum(terminal_distances) / len(terminal_distances)
    net_progress_score = max(
        0.0,
        min(1.0, (start_distance - terminal_distance) / start_distance),
    )

    progress_steps = 0
    regression_penalty = 0.0
    for previous_distance, current_distance in zip(anchor_distances, anchor_distances[1:]):
        if current_distance <= previous_distance + regression_tolerance:
            progress_steps += 1
        else:
            regression_penalty += current_distance - previous_distance - regression_tolerance

    progress_ratio = progress_steps / max(len(anchor_distances) - 1, 1)
    regression_score = max(0.0, 1.0 - (regression_penalty / start_distance))

    cumulative_distance = sum(
        math.dist(
            (float(previous_point[0]), float(previous_point[1])),
            (float(current_point[0]), float(current_point[1])),
        )
        for previous_point, current_point in zip(path_points, path_points[1:])
    )
    direct_distance = math.dist(
        (float(path_points[0][0]), float(path_points[0][1])),
        terminal_anchor,
    )
    if direct_distance <= 12:
        detour_score = 0.72
    else:
        detour_ratio = cumulative_distance / max(direct_distance, 1.0)
        detour_score = max(
            0.0,
            1.0 - ((detour_ratio - 1.0) / max(max_detour_ratio - 1.0, 0.1)),
        )

    return round(
        min(
            1.0,
            net_progress_score * 0.24
            + regression_score * 0.24
            + detour_score * 0.20
            + progress_ratio * 0.12
            + terminal_density_score * 0.12
            + tail_score * 0.05
            + terminal_span_score * 0.03,
        ),
        3,
    )


def _should_confirm_assist_highlight(
    *,
    window_confidence: float,
    handoff_confidence: float,
    continuity_confidence: float,
    terminal_release_confidence: float,
    receiver_trajectory_confidence: float,
    highlight_confidence: float,
    confirm_threshold: float,
    handoff_threshold: float,
    continuity_threshold: float,
    terminal_threshold: float,
    receiver_threshold: float,
    tolerant_window_floor: float,
    tolerant_handoff_floor: float,
    tolerant_tail_average_floor: float,
    tolerant_tail_min_floor: float,
) -> bool:
    if (
        highlight_confidence >= confirm_threshold
        and handoff_confidence >= handoff_threshold
        and continuity_confidence >= continuity_threshold
        and terminal_release_confidence >= terminal_threshold
        and receiver_trajectory_confidence >= receiver_threshold
    ):
        return True

    tail_metrics = [
        continuity_confidence,
        terminal_release_confidence,
        receiver_trajectory_confidence,
    ]
    tail_thresholds = [
        continuity_threshold,
        terminal_threshold,
        receiver_threshold,
    ]
    strong_tail_support = sum(
        metric >= threshold
        for metric, threshold in zip(tail_metrics, tail_thresholds)
    )
    near_tail_support = sum(
        metric >= max(threshold - ASSIST_CONFIRM_TOLERANCE, 0.0)
        for metric, threshold in zip(tail_metrics, tail_thresholds)
    )
    tail_average = sum(tail_metrics) / max(len(tail_metrics), 1)
    tail_min = min(tail_metrics) if tail_metrics else 0.0

    high_anchor_confirm_floor = max(confirm_threshold - 0.02, 0.0)
    high_anchor_window_floor = min(
        1.0,
        max(tolerant_window_floor, confirm_threshold + 0.18),
    )
    high_anchor_handoff_floor = min(
        1.0,
        max(tolerant_handoff_floor, handoff_threshold + 0.12),
    )
    high_anchor_tail_average_floor = max(
        tolerant_tail_average_floor,
        (sum(tail_thresholds) / max(len(tail_thresholds), 1)) + 0.05,
    )
    high_anchor_tail_min_floor = min(
        tolerant_tail_min_floor,
        max(min(tail_thresholds) - 0.05, 0.0),
    )

    return (
        highlight_confidence >= min(1.0, confirm_threshold + 0.02)
        and window_confidence >= tolerant_window_floor
        and handoff_confidence >= tolerant_handoff_floor
        and strong_tail_support >= 2
        and near_tail_support == len(tail_metrics)
        and tail_average >= tolerant_tail_average_floor
        and tail_min >= tolerant_tail_min_floor
    ) or (
        highlight_confidence >= high_anchor_confirm_floor
        and window_confidence >= high_anchor_window_floor
        and handoff_confidence >= high_anchor_handoff_floor
        and strong_tail_support >= 2
        and receiver_trajectory_confidence >= receiver_threshold
        and tail_average >= high_anchor_tail_average_floor
        and tail_min >= high_anchor_tail_min_floor
    )


def _local_window_confidence(matches: List[Dict[str, object]]) -> float:
    if not matches:
        return 0.0

    ordered_matches = sorted(matches, key=lambda item: int(item["frame"]))
    best_score = max(float(item["score"]) for item in ordered_matches)
    mean_score = sum(float(item["score"]) for item in ordered_matches) / len(ordered_matches)
    span = max(int(ordered_matches[-1]["frame"]) - int(ordered_matches[0]["frame"]), 0)
    sustain_score = min(1.0, len(ordered_matches) / 3.0)
    span_score = min(1.0, span / 8.0)

    return round(
        min(
            1.0,
            best_score * 0.38
            + mean_score * 0.32
            + sustain_score * 0.18
            + span_score * 0.12,
        ),
        3,
    )


def _build_ball_aligned_candidates(ball_point: Tuple[int, int], expected_bbox: BBox, frame_shape) -> List[BBox]:
    frame_height, frame_width = frame_shape[:2]
    ball_x, ball_y = ball_point
    expected_width = max(int(expected_bbox[2]), 24)
    expected_height = max(int(expected_bbox[3]), 48)
    candidates: List[BBox] = []
    seen = set()

    for scale in BALL_ALIGNED_SCALE_FACTORS:
        width = max(24, int(round(expected_width * scale)))
        height = max(48, int(round(expected_height * scale)))
        for horizontal_ratio in (0.25, 0.4, 0.55, 0.7):
            for vertical_ratio in (0.2, 0.35, 0.5, 0.65):
                bbox = _normalize_bbox(
                    (
                        int(round(ball_x - width * horizontal_ratio)),
                        int(round(ball_y - height * vertical_ratio)),
                        width,
                        height,
                    ),
                    frame_width,
                    frame_height,
                )
                if bbox is None or bbox in seen:
                    continue
                seen.add(bbox)
                candidates.append(bbox)

    return candidates


def _score_local_target_candidate(
    frame,
    bbox: BBox,
    tracker: "TargetPlayerTracker",
    ball_point: Tuple[int, int],
    expected_bbox: BBox,
) -> Dict[str, object]:
    candidate_hist, candidate_template = tracker._extract_appearance(frame, bbox)
    hist_score = tracker._combined_hist_similarity(candidate_hist)
    template_score = tracker._best_template_similarity(candidate_template)
    expanded_bbox = _expand_bbox(bbox, horizontal_ratio=0.18, vertical_ratio=0.12)
    inside = _point_in_bbox(ball_point, expanded_bbox)
    bbox_scale = max(max(bbox[2], bbox[3]), 1)
    distance_score = max(0.0, 1.0 - (_distance_to_bbox(ball_point, bbox) / bbox_scale))
    proximity_score = 1.0 if inside else round(distance_score, 3)
    scale_score = _bbox_scale_similarity(bbox, expected_bbox)

    combined_score = round(
        min(
            1.0,
            hist_score * 0.55
            + template_score * 0.25
            + proximity_score * 0.12
            + scale_score * 0.08,
        ),
        3,
    )

    return {
        "bbox": bbox,
        "inside": inside,
        "score": combined_score,
        "histScore": round(float(hist_score), 3),
        "templateScore": round(float(template_score), 3),
        "proximityScore": round(float(proximity_score), 3),
    }


def _find_local_target_match(
    frame,
    frame_index: int,
    ball_point: Tuple[int, int],
    tracker: "TargetPlayerTracker",
    expected_bbox: BBox,
) -> Optional[Dict[str, object]]:
    candidate_bboxes = _build_ball_aligned_candidates(ball_point, expected_bbox, frame.shape)
    tracked_bbox = tracker.get_box_at_frame(frame_index, max_gap=6)
    if tracked_bbox is not None:
        candidate_bboxes.insert(0, tracked_bbox)

    best_match = None
    seen = set()

    for bbox in candidate_bboxes:
        if bbox in seen:
            continue
        seen.add(bbox)

        scored_candidate = _score_local_target_candidate(
            frame,
            bbox,
            tracker,
            ball_point,
            expected_bbox,
        )
        if not bool(scored_candidate["inside"]):
            continue
        if (
            float(scored_candidate["histScore"]) < LOCAL_REVIEW_HISTOGRAM_FLOOR
            or float(scored_candidate["score"]) < LOCAL_REVIEW_MATCH_THRESHOLD
        ):
            continue

        if best_match is None or float(scored_candidate["score"]) > float(best_match["score"]):
            best_match = scored_candidate

    return best_match


class TargetPlayerTracker:
    """Tracks the user-selected player while rejecting identity drift."""

    _hog_detector = None

    def __init__(
        self,
        selection_box: Dict[str, int],
        max_missing_frames: int = 12,
        start_frame: int = 0,
        start_time: float = 0.0,
        revalidate_interval: int = 6,
        reacquire_interval: int = 3,
    ):
        self.selection_box = selection_box
        self.max_missing_frames = max_missing_frames
        self.start_frame = max(int(start_frame), 0)
        self.start_time = max(float(start_time), 0.0)
        self.revalidate_interval = max(int(revalidate_interval), 3)
        self.reacquire_interval = max(int(reacquire_interval), 2)

        self.tracker = None
        self.tracker_type = self._get_tracker_name()

        self.current_bbox: Optional[BBox] = None
        self.last_bbox: Optional[BBox] = None
        self.initial_bbox: Optional[BBox] = None
        self.trusted_bbox: Optional[BBox] = None
        self.history: List[Dict] = []

        self.reference_hist: Optional[np.ndarray] = None
        self.adaptive_hist: Optional[np.ndarray] = None
        self.reference_template: Optional[np.ndarray] = None
        self.reference_hists: List[np.ndarray] = []
        self.reference_templates: List[np.ndarray] = []
        self.reference_sample_frames: List[int] = []

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

    def _effective_motion_score(
        self,
        bbox: BBox,
        local_anchor_bbox: Optional[BBox] = None,
    ) -> Tuple[float, float, float]:
        if local_anchor_bbox is None:
            local_anchor_bbox = self.current_bbox or self.last_bbox or self.initial_bbox

        trusted_anchor_bbox = self.trusted_bbox or self.initial_bbox or local_anchor_bbox
        local_motion_score = self._motion_score(bbox, local_anchor_bbox)
        trusted_motion_score = self._motion_score(bbox, trusted_anchor_bbox)

        if (
            local_anchor_bbox is None
            or trusted_anchor_bbox is None
            or trusted_anchor_bbox == local_anchor_bbox
        ):
            effective_motion_score = local_motion_score
        else:
            # Keep short-term continuity, but do not let it outrun the last
            # trusted identity anchor and slowly drift onto another player.
            effective_motion_score = min(local_motion_score, trusted_motion_score)

        return (
            float(local_motion_score),
            float(trusted_motion_score),
            float(effective_motion_score),
        )

    def _refresh_trusted_bbox(self, bbox: Optional[BBox]) -> None:
        if bbox is None:
            return
        self.trusted_bbox = bbox

    def _should_refresh_trusted_tracking_bbox(
        self,
        bbox: BBox,
        hist_score: float,
        template_score: float,
    ) -> bool:
        if (
            hist_score < ADAPTIVE_REFERENCE_HIST_FLOOR
            or template_score < ADAPTIVE_REFERENCE_TEMPLATE_FLOOR
        ):
            return False

        _, trusted_motion_score, _ = self._effective_motion_score(
            bbox,
            self.current_bbox or self.last_bbox or self.initial_bbox,
        )
        return trusted_motion_score >= TRUSTED_TRACKING_MOTION_FLOOR

    def _extract_appearance(self, frame, bbox: BBox):
        crop = _crop_from_bbox(frame, bbox)
        return _compute_histogram(crop), _build_template(crop)

    def _register_reference_appearance(
        self,
        candidate_hist: Optional[np.ndarray],
        candidate_template: Optional[np.ndarray],
        frame_index: Optional[int] = None,
    ):
        if candidate_hist is not None:
            if not self.reference_hists:
                self.reference_hists.append(candidate_hist.copy())
            else:
                best_existing_hist = max(
                    _hist_similarity(existing_hist, candidate_hist)
                    for existing_hist in self.reference_hists
                )
                if best_existing_hist < 0.96:
                    self.reference_hists.append(candidate_hist.copy())

            if self.reference_hist is None:
                self.reference_hist = candidate_hist.copy()

        if candidate_template is not None:
            if not self.reference_templates:
                self.reference_templates.append(candidate_template.copy())
            else:
                best_existing_template = max(
                    _template_similarity(existing_template, candidate_template)
                    for existing_template in self.reference_templates
                )
                if best_existing_template < 0.94:
                    self.reference_templates.append(candidate_template.copy())

            if self.reference_template is None:
                self.reference_template = candidate_template.copy()

        self._record_reference_frame(frame_index)

    def _best_template_similarity(self, candidate_template: Optional[np.ndarray]) -> float:
        if candidate_template is None:
            return 0.0

        templates = self.reference_templates or ([self.reference_template] if self.reference_template is not None else [])
        if not templates:
            return 0.0

        return max(_template_similarity(reference_template, candidate_template) for reference_template in templates)

    def _reference_hist_similarity(self, candidate_hist: Optional[np.ndarray]) -> float:
        if candidate_hist is None:
            return 0.0

        reference_candidates = self.reference_hists or ([self.reference_hist] if self.reference_hist is not None else [])
        return max(
            (_hist_similarity(reference_hist, candidate_hist) for reference_hist in reference_candidates),
            default=0.0,
        )

    def _adaptive_hist_similarity(self, candidate_hist: Optional[np.ndarray]) -> float:
        if candidate_hist is None:
            return 0.0
        return _hist_similarity(self.adaptive_hist, candidate_hist)

    def _record_reference_frame(self, frame_index: Optional[int]) -> None:
        if frame_index is None:
            return

        normalized_frame_index = int(frame_index)
        if normalized_frame_index not in self.reference_sample_frames:
            self.reference_sample_frames.append(normalized_frame_index)
            self.reference_sample_frames.sort()

    def _combined_hist_similarity(self, candidate_hist: Optional[np.ndarray]) -> float:
        if candidate_hist is None:
            return 0.0

        reference_score = self._reference_hist_similarity(candidate_hist)
        adaptive_score = self._adaptive_hist_similarity(candidate_hist)
        if reference_score <= 0.0:
            return float(adaptive_score)
        return float(reference_score * 0.85 + adaptive_score * 0.15)

    def _update_adaptive_hist(self, candidate_hist: Optional[np.ndarray]):
        if candidate_hist is None:
            return

        if self.adaptive_hist is None:
            self.adaptive_hist = candidate_hist.copy()
            return

        self.adaptive_hist = cv2.addWeighted(self.adaptive_hist, 0.85, candidate_hist, 0.15, 0.0)
        cv2.normalize(self.adaptive_hist, self.adaptive_hist)

    def _score_tracker_candidate(self, frame, bbox: BBox) -> Tuple[float, float, float]:
        candidate_hist, candidate_template = self._extract_appearance(frame, bbox)
        reference_hist_score = self._reference_hist_similarity(candidate_hist)
        adaptive_hist_score = self._adaptive_hist_similarity(candidate_hist)
        template_score = self._best_template_similarity(candidate_template)
        _, _, motion_score = self._effective_motion_score(bbox)
        combined_score = (
            reference_hist_score * 0.46
            + template_score * 0.26
            + adaptive_hist_score * 0.10
            + motion_score * 0.18
        )
        return (
            float(reference_hist_score),
            float(template_score),
            float(combined_score),
        )

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

    def _detect_people_candidates(
        self,
        frame,
        search_bbox: BBox,
        anchor_bbox: Optional[BBox],
    ) -> List[Dict]:
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

            candidate_hist, candidate_template = self._extract_appearance(frame, normalized_bbox)
            reference_hist_score = self._reference_hist_similarity(candidate_hist)
            adaptive_hist_score = self._adaptive_hist_similarity(candidate_hist)
            template_score = self._best_template_similarity(candidate_template)
            _, _, motion_score = self._effective_motion_score(normalized_bbox, anchor_bbox)
            detector_confidence = min(max(float(weight), 0.0), 2.0) / 2.0
            score = (
                reference_hist_score * 0.42
                + template_score * 0.24
                + adaptive_hist_score * 0.10
                + motion_score * 0.14
                + detector_confidence * 0.10
            )

            if (
                reference_hist_score < SEARCH_HISTOGRAM_FLOOR
                or template_score < SEARCH_TEMPLATE_FLOOR
                or score < SEARCH_MATCH_THRESHOLD
            ):
                continue

            candidates.append({
                "bbox": normalized_bbox,
                "score": float(score),
                "histScore": float(reference_hist_score),
                "templateScore": float(template_score),
                "source": "person-detector",
            })

        return candidates

    def _search_nearby_target(self, frame, anchor_bbox: Optional[BBox]) -> Optional[Dict]:
        templates = self.reference_templates or ([self.reference_template] if self.reference_template is not None else [])
        template_enabled = bool(templates)

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

        candidates = self._detect_people_candidates(frame, search_bbox, anchor_bbox)

        if template_enabled:
            search_gray = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
            for template in templates:
                for scale in TEMPLATE_SEARCH_SCALE_FACTORS:
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

                    candidate_hist, candidate_template = self._extract_appearance(frame, candidate_bbox)
                    reference_hist_score = self._reference_hist_similarity(candidate_hist)
                    adaptive_hist_score = self._adaptive_hist_similarity(candidate_hist)
                    template_score = self._best_template_similarity(candidate_template)
                    _, _, motion_score = self._effective_motion_score(candidate_bbox, anchor_bbox)
                    combined_score = (
                        float(max_value) * 0.28
                        + reference_hist_score * 0.30
                        + template_score * 0.24
                        + adaptive_hist_score * 0.06
                        + motion_score * 0.12
                    )

                    if (
                        reference_hist_score < SEARCH_HISTOGRAM_FLOOR
                        or template_score < SEARCH_TEMPLATE_FLOOR
                        or combined_score < SEARCH_MATCH_THRESHOLD
                    ):
                        continue

                    candidates.append({
                        "bbox": candidate_bbox,
                        "score": float(combined_score),
                        "histScore": float(reference_hist_score),
                        "templateScore": float(template_score),
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
        self.reference_hist = initial_hist.copy() if initial_hist is not None else None
        self.adaptive_hist = initial_hist.copy() if initial_hist is not None else None
        self.reference_template = initial_template.copy() if initial_template is not None else None
        self.reference_hists = []
        self.reference_templates = []
        self.reference_sample_frames = []
        self._register_reference_appearance(initial_hist, initial_template, frame_index=frame_index)
        self.initial_bbox = initial_bbox
        self.trusted_bbox = initial_bbox
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

    def add_reference_sample(
        self,
        frame,
        frame_index: int,
        anchor_bbox: Optional[BBox] = None,
    ) -> Optional[Dict]:
        if self.initial_bbox is None or self.reference_hist is None:
            return None

        if anchor_bbox is None:
            anchor_bbox = self.trusted_bbox or self.last_bbox or self.current_bbox or self.initial_bbox

        candidate = self._search_nearby_target(frame, anchor_bbox)
        if candidate is None:
            return None

        candidate_bbox = candidate["bbox"]
        normalized_bbox = _normalize_bbox(candidate_bbox, frame.shape[1], frame.shape[0]) if candidate_bbox else None
        if normalized_bbox is None:
            return None

        candidate_hist, candidate_template = self._extract_appearance(frame, normalized_bbox)
        hist_score = self._combined_hist_similarity(candidate_hist)
        template_score = self._best_template_similarity(candidate_template)
        combined_score = round(hist_score * 0.7 + template_score * 0.3, 3)

        if hist_score < TRACKER_HISTOGRAM_FLOOR or combined_score < TRACKER_IDENTITY_THRESHOLD:
            return None

        self._register_reference_appearance(candidate_hist, candidate_template, frame_index=frame_index)
        self._update_adaptive_hist(candidate_hist)
        self._refresh_trusted_bbox(normalized_bbox)

        return {
            "frame": int(frame_index),
            "bbox": normalized_bbox,
            "histScore": round(float(hist_score), 3),
            "templateScore": round(float(template_score), 3),
            "score": combined_score,
        }

    def register_tracking_sample(
        self,
        frame,
        frame_index: int,
        bbox: Optional[BBox] = None,
    ) -> Optional[Dict]:
        if self.reference_hist is None:
            return None

        if frame_index in self.reference_sample_frames:
            return None

        if bbox is None:
            bbox = self.current_bbox or self.last_bbox or self.initial_bbox

        if bbox is None:
            return None

        normalized_bbox = _normalize_bbox(bbox, frame.shape[1], frame.shape[0])
        if normalized_bbox is None:
            return None

        candidate_hist, candidate_template = self._extract_appearance(frame, normalized_bbox)
        reference_hist_score = self._reference_hist_similarity(candidate_hist)
        adaptive_hist_score = self._adaptive_hist_similarity(candidate_hist)
        template_score = self._best_template_similarity(candidate_template)
        _, _, motion_score = self._effective_motion_score(normalized_bbox)
        combined_score = round(
            reference_hist_score * 0.50
            + template_score * 0.26
            + adaptive_hist_score * 0.10
            + motion_score * 0.14,
            3,
        )

        if (
            reference_hist_score < TRACKER_HISTOGRAM_FLOOR
            or template_score < RUNTIME_REFERENCE_TEMPLATE_FLOOR
            or combined_score < TRACKER_IDENTITY_THRESHOLD
        ):
            return None

        self._record_reference_frame(frame_index)
        self._update_adaptive_hist(candidate_hist)
        self._refresh_trusted_bbox(normalized_bbox)

        return {
            "frame": int(frame_index),
            "bbox": normalized_bbox,
            "histScore": round(float(reference_hist_score), 3),
            "templateScore": round(float(template_score), 3),
            "motionScore": round(float(motion_score), 3),
            "score": combined_score,
        }

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
        tracker_template_score = 0.0
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
                    tracker_hist_score, tracker_template_score, tracker_score = self._score_tracker_candidate(frame, normalized_bbox)
                    if (
                        tracker_hist_score >= TRACKER_HISTOGRAM_FLOOR
                        and tracker_template_score >= TRACKER_TEMPLATE_FLOOR
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
            if self._should_refresh_trusted_tracking_bbox(
                tracker_candidate,
                tracker_hist_score,
                tracker_template_score,
            ):
                self._refresh_trusted_bbox(tracker_candidate)
            if self._should_revalidate(frame_index):
                recalibrated_candidate = self._search_nearby_target(frame, tracker_candidate)
                if (
                    recalibrated_candidate is not None
                    and recalibrated_candidate["score"] >= tracker_score + REVALIDATION_SWITCH_DELTA
                ):
                    tracker_candidate = recalibrated_candidate["bbox"]
                    tracker_score = float(recalibrated_candidate["score"])
                    self._reinitialize_tracker(frame, tracker_candidate)
                    self._refresh_trusted_bbox(tracker_candidate)
                    status = "revalidated"
                else:
                    self._reinitialize_tracker(frame, tracker_candidate)
                if (
                    tracker_hist_score >= ADAPTIVE_REFERENCE_HIST_FLOOR
                    and tracker_template_score >= ADAPTIVE_REFERENCE_TEMPLATE_FLOOR
                ):
                    candidate_hist, _ = self._extract_appearance(frame, tracker_candidate)
                    self._update_adaptive_hist(candidate_hist)
                    self._refresh_trusted_bbox(tracker_candidate)
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
            search_candidate = self._search_nearby_target(
                frame,
                self.trusted_bbox or self.last_bbox or self.current_bbox or self.initial_bbox,
            )

        if search_candidate is not None:
            reacquired_bbox = search_candidate["bbox"]
            self._reinitialize_tracker(frame, reacquired_bbox)
            self.missing_frames = 0
            self.reacquired_count += 1
            self._refresh_trusted_bbox(reacquired_bbox)
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
            confidence=max(tracker_score, tracker_hist_score, tracker_template_score),
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
            "referenceFrames": list(self.reference_sample_frames),
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


def review_shot_with_local_window(
    ball_positions,
    frame_buffer: Dict[int, np.ndarray],
    tracker: Optional[TargetPlayerTracker],
    shot_release_frame: int,
    shooter_lookback_frames: int = 18,
    assist_lookback_frames: int = 90,
    assist_release_gap_frames: int = 8,
    assist_max_gap_frames: int = 90,
) -> ShotAttribution:
    if tracker is None or not frame_buffer or tracker.reference_hist is None:
        return {
            "owner": "unknown",
            "owner_confidence": 0.0,
            "target_visible": False,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    expected_bbox = (
        tracker.get_box_at_frame(shot_release_frame, max_gap=6)
        or tracker.current_bbox
        or tracker.last_bbox
        or tracker.initial_bbox
    )
    if expected_bbox is None:
        return {
            "owner": "unknown",
            "owner_confidence": 0.0,
            "target_visible": False,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    review_start = shot_release_frame - assist_lookback_frames
    local_ball_frames = _collect_ball_points_by_frame(ball_positions, review_start, shot_release_frame)
    local_matches: List[Dict[str, object]] = []
    local_timeline_samples: List[Dict[str, object]] = []

    for sample in local_ball_frames:
        frame_index = int(sample["frame"])
        raw_frame = frame_buffer.get(frame_index)
        if raw_frame is None:
            continue

        frame = raw_frame
        if isinstance(raw_frame, np.ndarray) and raw_frame.ndim == 1:
            frame = cv2.imdecode(raw_frame, cv2.IMREAD_COLOR)
        if frame is None or getattr(frame, "size", 0) == 0:
            continue

        ball_point = sample["point"]
        local_match = _find_local_target_match(
            frame,
            frame_index,
            ball_point,
            tracker,
            expected_bbox,
        )
        if local_match is None:
            local_timeline_samples.append({
                "frame": frame_index,
                "score": 0.0,
                "inside": False,
            })
            continue

        timeline_sample = {
            "frame": frame_index,
            "score": float(local_match["score"]),
            "inside": bool(local_match["inside"]),
        }
        local_matches.append(timeline_sample)
        local_timeline_samples.append(timeline_sample)

    release_matches = [
        match
        for match in local_matches
        if shot_release_frame - shooter_lookback_frames <= int(match["frame"]) <= shot_release_frame
    ]
    release_windows = _build_control_windows(
        release_matches,
        strong_threshold=LOCAL_REVIEW_STRONG_SCORE,
        max_gap_frames=8,
    )
    latest_release_window = release_windows[-1] if release_windows else None
    release_confidence = (
        _control_window_confidence(latest_release_window)
        if latest_release_window
        else _local_window_confidence(release_matches)
    )
    target_visible = bool(local_matches)

    if (
        release_confidence >= 0.64
        and latest_release_window is not None
    ):
        return {
            "owner": "target",
            "owner_confidence": release_confidence,
            "target_visible": True,
            "highlight_role": "score",
            "highlight_confidence": release_confidence,
            "involvement_start_frame": int(latest_release_window["start_frame"]),
            "involvement_end_frame": int(latest_release_window["end_frame"]),
        }

    assist_window_end = shot_release_frame - assist_release_gap_frames
    assist_matches = [
        match
        for match in local_matches
        if review_start <= int(match["frame"]) <= assist_window_end
    ]
    assist_windows = _build_control_windows(
        assist_matches,
        strong_threshold=LOCAL_REVIEW_STRONG_SCORE,
        max_gap_frames=12,
    )
    latest_assist_window = assist_windows[-1] if assist_windows else None
    assist_confidence = (
        _control_window_confidence(latest_assist_window)
        if latest_assist_window
        else _local_window_confidence(assist_matches)
    )
    last_assist_frame = int(latest_assist_window["end_frame"]) if latest_assist_window else None

    if last_assist_frame is not None:
        frame_gap = shot_release_frame - last_assist_frame
    else:
        frame_gap = None

    if (
        assist_confidence >= 0.56
        and latest_assist_window is not None
        and frame_gap is not None
        and assist_release_gap_frames <= frame_gap <= assist_max_gap_frames
    ):
        handoff_confidence = _handoff_confidence(
            local_timeline_samples,
            latest_assist_window,
            shot_release_frame=shot_release_frame,
            separation_threshold=0.42,
            strong_threshold=LOCAL_REVIEW_STRONG_SCORE,
        )
        continuity_confidence = _post_handoff_continuity_confidence(
            local_ball_frames,
            start_frame=last_assist_frame,
            shot_release_frame=shot_release_frame,
            max_tail_gap_frames=10,
            max_internal_gap_frames=18,
        )
        terminal_release_confidence = _terminal_release_window_confidence(
            local_ball_frames,
            start_frame=last_assist_frame,
            shot_release_frame=shot_release_frame,
            max_gap_frames=8,
            max_tail_gap_frames=6,
        )
        receiver_trajectory_confidence = _receiver_trajectory_confidence(
            local_ball_frames,
            start_frame=last_assist_frame,
            shot_release_frame=shot_release_frame,
            max_gap_frames=8,
            max_tail_gap_frames=6,
        )
        gap_score = max(
            0.0,
            1.0 - ((frame_gap - assist_release_gap_frames) / max(assist_max_gap_frames - assist_release_gap_frames, 1)),
        )
        highlight_confidence = round(
            min(
                1.0,
                assist_confidence * 0.24
                + handoff_confidence * 0.22
                + continuity_confidence * 0.16
                + terminal_release_confidence * 0.16
                + receiver_trajectory_confidence * 0.16
                + gap_score * 0.06,
            ),
            3,
        )

        if _should_confirm_assist_highlight(
            window_confidence=assist_confidence,
            handoff_confidence=handoff_confidence,
            continuity_confidence=continuity_confidence,
            terminal_release_confidence=terminal_release_confidence,
            receiver_trajectory_confidence=receiver_trajectory_confidence,
            highlight_confidence=highlight_confidence,
            confirm_threshold=0.58,
            handoff_threshold=0.4,
            continuity_threshold=0.3,
            terminal_threshold=0.25,
            receiver_threshold=0.28,
            tolerant_window_floor=0.62,
            tolerant_handoff_floor=0.44,
            tolerant_tail_average_floor=0.36,
            tolerant_tail_min_floor=0.22,
        ):
            return {
                "owner": "unknown",
                "owner_confidence": release_confidence,
                "target_visible": True,
                "highlight_role": "assist",
                "highlight_confidence": highlight_confidence,
                "involvement_start_frame": int(latest_assist_window["start_frame"]),
                "involvement_end_frame": last_assist_frame,
            }

    return {
        "owner": "unknown",
        "owner_confidence": release_confidence,
        "target_visible": target_visible,
        "highlight_role": "none",
        "highlight_confidence": assist_confidence if assist_confidence >= 0.45 else 0.0,
        "involvement_start_frame": int(latest_assist_window["start_frame"]) if assist_confidence >= 0.45 and latest_assist_window else None,
        "involvement_end_frame": int(latest_assist_window["end_frame"]) if assist_confidence >= 0.45 and latest_assist_window else None,
    }


def classify_shot_involvement(
    ball_positions,
    tracker: Optional[TargetPlayerTracker],
    shot_release_frame: int,
    shooter_lookback_frames: int = 12,
    assist_lookback_frames: int = 120,
    assist_release_gap_frames: int = 8,
    assist_max_gap_frames: int = 90,
) -> ShotAttribution:
    if tracker is None:
        return {
            "owner": "unknown",
            "owner_confidence": 0.0,
            "target_visible": False,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
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
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    owner = "unknown"
    owner_confidence = 0.0
    release_score = 0.0

    release_windows = _build_control_windows(
        release_samples,
        strong_threshold=0.55,
        max_gap_frames=8,
    )
    latest_release_window = release_windows[-1] if release_windows else None

    if release_samples:
        inside_ratio = sum(1 for sample in release_samples if sample["inside"]) / len(release_samples)
        best_score = max(sample["score"] for sample in release_samples)
        release_score = (
            float(latest_release_window["best_score"])
            if latest_release_window
            else float(release_samples[-1]["score"])
        )
        owner_confidence = round(
            min(1.0, inside_ratio * 0.45 + best_score * 0.35 + release_score * 0.20),
            3,
        )
        if (
            owner_confidence >= 0.58
            and latest_release_window is not None
        ):
            owner = "target"

    if owner == "target":
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": True,
            "highlight_role": "score",
            "highlight_confidence": owner_confidence,
            "involvement_start_frame": int(latest_release_window["start_frame"]),
            "involvement_end_frame": int(latest_release_window["end_frame"]),
        }

    if assist_window_end < shot_release_frame - assist_lookback_frames:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    assist_windows = _build_control_windows(
        assist_samples,
        strong_threshold=0.6,
        max_gap_frames=12,
    )
    latest_assist_window = assist_windows[-1] if assist_windows else None

    if latest_assist_window is None:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    last_target_control_frame = int(latest_assist_window["end_frame"])
    frame_gap = shot_release_frame - last_target_control_frame

    if frame_gap < assist_release_gap_frames or frame_gap > assist_max_gap_frames:
        return {
            "owner": owner,
            "owner_confidence": owner_confidence,
            "target_visible": target_visible,
            "highlight_role": "none",
            "highlight_confidence": 0.0,
            "involvement_start_frame": None,
            "involvement_end_frame": None,
        }

    handoff_samples = _collect_target_ball_samples(
        ball_positions,
        tracker,
        last_target_control_frame + 1,
        shot_release_frame,
        horizontal_ratio=0.45,
        vertical_ratio=0.35,
        max_gap=12,
    )
    handoff_confidence = _handoff_confidence(
        handoff_samples,
        latest_assist_window,
        shot_release_frame=shot_release_frame,
        separation_threshold=0.38,
        strong_threshold=0.55,
    )
    continuity_samples = _collect_ball_points_by_frame(
        ball_positions,
        last_target_control_frame + 1,
        shot_release_frame,
    )
    continuity_confidence = _post_handoff_continuity_confidence(
        continuity_samples,
        start_frame=last_target_control_frame,
        shot_release_frame=shot_release_frame,
        max_tail_gap_frames=10,
        max_internal_gap_frames=18,
    )
    terminal_release_confidence = _terminal_release_window_confidence(
        continuity_samples,
        start_frame=last_target_control_frame,
        shot_release_frame=shot_release_frame,
        max_gap_frames=8,
        max_tail_gap_frames=6,
    )
    receiver_trajectory_confidence = _receiver_trajectory_confidence(
        continuity_samples,
        start_frame=last_target_control_frame,
        shot_release_frame=shot_release_frame,
        max_gap_frames=8,
        max_tail_gap_frames=6,
    )
    window_confidence = _control_window_confidence(latest_assist_window)
    gap_score = max(
        0.0,
        1.0 - ((frame_gap - assist_release_gap_frames) / max(assist_max_gap_frames - assist_release_gap_frames, 1)),
    )
    separation_score = 1.0 if not release_samples else max(0.0, 1.0 - release_score)

    highlight_confidence = round(
        min(
            1.0,
            window_confidence * 0.20
            + handoff_confidence * 0.20
            + continuity_confidence * 0.14
            + terminal_release_confidence * 0.17
            + receiver_trajectory_confidence * 0.18
            + gap_score * 0.05
            + separation_score * 0.06,
        ),
        3,
    )

    highlight_role = (
        "assist"
        if _should_confirm_assist_highlight(
            window_confidence=window_confidence,
            handoff_confidence=handoff_confidence,
            continuity_confidence=continuity_confidence,
            terminal_release_confidence=terminal_release_confidence,
            receiver_trajectory_confidence=receiver_trajectory_confidence,
            highlight_confidence=highlight_confidence,
            confirm_threshold=0.5,
            handoff_threshold=0.35,
            continuity_threshold=0.25,
            terminal_threshold=0.25,
            receiver_threshold=0.28,
            tolerant_window_floor=0.6,
            tolerant_handoff_floor=0.38,
            tolerant_tail_average_floor=0.33,
            tolerant_tail_min_floor=0.2,
        )
        else "none"
    )
    partial_assist_confidence = (
        highlight_confidence
        if latest_assist_window is not None and highlight_confidence >= PARTIAL_ASSIST_EVIDENCE_THRESHOLD
        else 0.0
    )
    preserve_partial_assist_window = partial_assist_confidence > 0.0

    return {
        "owner": owner,
        "owner_confidence": owner_confidence,
        "target_visible": target_visible,
        "highlight_role": highlight_role,
        "highlight_confidence": (
            highlight_confidence
            if highlight_role != "none"
            else partial_assist_confidence
        ),
        "involvement_start_frame": (
            int(latest_assist_window["start_frame"])
            if highlight_role != "none" or preserve_partial_assist_window
            else None
        ),
        "involvement_end_frame": (
            last_target_control_frame
            if highlight_role != "none" or preserve_partial_assist_window
            else None
        ),
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
