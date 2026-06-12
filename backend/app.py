from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import os
import json
import uuid
import threading
import time
import logging
import shutil
import io
import zipfile
import mimetypes
import base64
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import cv2
from shot_detector_video import BasketballShotDetector
from video_processor import VideoProcessor
from player_tracker import TargetPlayerTracker

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('basketball_highlight.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BACKEND_HOST = os.getenv('BACKEND_HOST', '127.0.0.1')
BACKEND_PORT = int(os.getenv('BACKEND_PORT', '5050'))
FRONTEND_PORT = os.getenv('FRONTEND_PORT', '5173')
BACKEND_DEBUG = os.getenv('BACKEND_DEBUG', '').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
DEBUG_KEEP_ARTIFACTS = os.getenv('DEBUG_KEEP_ARTIFACTS', '').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
TASK_RETENTION_SECONDS = int(os.getenv('TASK_RETENTION_SECONDS', str(7 * 24 * 60 * 60)))
AUTO_KEEP_REVIEW_ANNOTATIONS = os.getenv('AUTO_KEEP_REVIEW_ANNOTATIONS', '1').strip().lower() in {
    '1',
    'true',
    'yes',
    'on',
}
DETECTION_PROGRESS_ATTEMPT_RANGES = (
    (10, 60),
    (60, 65),
    (65, 68),
    (68, 69),
)
DEFAULT_CLIP_BEFORE_SECONDS = float(os.getenv('DEFAULT_CLIP_BEFORE_SECONDS', '6'))
DEFAULT_CLIP_AFTER_SECONDS = float(os.getenv('DEFAULT_CLIP_AFTER_SECONDS', '2'))
API_RUNTIME_VERSION = 2
SERVER_STARTED_AT = datetime.now(timezone.utc)
SMART_SELECTION_MAX_CANDIDATES = 8
SMART_SELECTION_MAX_SAMPLE_TIMES = 18
SMART_SELECTION_MAX_FRAME_WIDTH = 960
SMART_SELECTION_JPEG_QUALITY = 82
ANNOTATED_REVIEW_COVERAGE_THRESHOLD = 0.55
ANNOTATED_REVIEW_OUTCOMES = {
    'review_candidates',
    'confirmed_with_review_candidates',
    'target_attempt_fallback',
    'global_makes_without_target',
    'no_makes_detected',
    'no_attempts_detected',
}

app = Flask(__name__)
LOCAL_FRONTEND_ORIGINS = list({
    f'http://127.0.0.1:{FRONTEND_PORT}',
    f'http://localhost:{FRONTEND_PORT}',
})
CORS(app, origins=LOCAL_FRONTEND_ORIGINS)
socketio = SocketIO(app, cors_allowed_origins=LOCAL_FRONTEND_ORIGINS, async_mode='threading')

# 绝对路径基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# 配置
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
TEMP_FOLDER = os.path.join(BASE_DIR, 'temp')
CHUNKS_FOLDER = os.path.join(TEMP_FOLDER, 'chunks')
TASK_METADATA_FOLDER = os.path.join(TEMP_FOLDER, 'task_metadata')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}
MAX_FILE_SIZE = 2048 * 1024 * 1024  # 2GB (increased for large video files)

# 创建必要的目录
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEMP_FOLDER, CHUNKS_FOLDER, TASK_METADATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"确保目录存在: {folder}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER
app.config['CHUNKS_FOLDER'] = CHUNKS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 全局任务存储
processing_tasks = {}
TERMINAL_TASK_STATUSES = {'completed', 'failed'}
PERSISTED_TASK_FIELDS = {
    'status',
    'progress',
    'stage',
    'result',
    'error',
    'created_at',
    'updated_at',
    'file_id',
    'input_path',
    'before_seconds',
    'after_seconds',
    'processing_mode',
    'manual_moments',
    'target_player_box',
    'effective_target_player_box',
}

def allowed_file(filename):
    """检查文件扩展名是否被允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_path):
    """验证文件大小"""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"文件大小 {file_size / (1024*1024):.1f}MB 超过限制 {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    return file_size

def validate_target_player_box(raw_box):
    """验证目标人物选区，使用原始像素坐标和选中时间点保存。"""
    if raw_box is None:
        return None

    if not isinstance(raw_box, dict):
        raise ValueError('targetPlayerBox 必须是对象')

    required_fields = ['x', 'y', 'width', 'height', 'frameWidth', 'frameHeight', 'selectionTime']
    box = {}

    for field in required_fields:
        value = raw_box.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f'targetPlayerBox.{field} 必须是数字')
        box[field] = float(value)

    if box['width'] < 20 or box['height'] < 20:
        raise ValueError('targetPlayerBox 过小，请重新框选人物区域')

    if box['frameWidth'] <= 0 or box['frameHeight'] <= 0:
        raise ValueError('targetPlayerBox 画面尺寸无效')

    if box['selectionTime'] < 0:
        raise ValueError('targetPlayerBox.selectionTime 不能为负数')

    if box['x'] < 0 or box['y'] < 0:
        raise ValueError('targetPlayerBox 坐标不能为负数')

    if box['x'] + box['width'] > box['frameWidth'] or box['y'] + box['height'] > box['frameHeight']:
        raise ValueError('targetPlayerBox 超出当前画面范围')

    normalized_box = {
        'x': int(round(box['x'])),
        'y': int(round(box['y'])),
        'width': int(round(box['width'])),
        'height': int(round(box['height'])),
        'frameWidth': int(round(box['frameWidth'])),
        'frameHeight': int(round(box['frameHeight'])),
        'selectionTime': round(box['selectionTime'], 3),
    }

    selection_frame = raw_box.get('selectionFrame')
    if isinstance(selection_frame, (int, float)) and selection_frame >= 0:
        normalized_box['selectionFrame'] = int(round(selection_frame))

    return normalized_box


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    is_opened = getattr(cap, 'isOpened', None)
    if not callable(is_opened) or not is_opened():
        raise ValueError('无法打开视频文件')

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or total_frames <= 0:
            raise ValueError('无法读取视频时长')

        duration = total_frames / fps
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
        }
    finally:
        cap.release()


def validate_manual_moments(raw_moments, duration):
    if not isinstance(raw_moments, list) or not raw_moments:
        raise ValueError('manualMoments 不能为空')

    normalized_moments = []
    seen = set()
    max_duration = max(float(duration or 0.0), 0.0)

    for raw_moment in raw_moments:
        if not isinstance(raw_moment, (int, float)):
            raise ValueError('manualMoments 只能包含数字时间点')

        moment = round(float(raw_moment), 3)
        if moment < 0:
            raise ValueError('manualMoments 不能包含负数时间点')

        if max_duration > 0 and moment > max_duration:
            raise ValueError('manualMoments 超出视频总时长')

        moment_key = round(moment, 2)
        if moment_key in seen:
            continue

        seen.add(moment_key)
        normalized_moments.append(moment)

    if not normalized_moments:
        raise ValueError('manualMoments 不能为空')

    normalized_moments.sort()
    return normalized_moments


def find_uploaded_filenames(file_id):
    if not file_id:
        return []

    prefix = f'{file_id}_'
    try:
        return sorted(
            filename for filename in os.listdir(app.config['UPLOAD_FOLDER'])
            if filename.startswith(prefix)
        )
    except OSError:
        return []


def find_uploaded_file_path(file_id):
    uploaded_filenames = find_uploaded_filenames(file_id)
    if not uploaded_filenames:
        return None

    return os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filenames[0])


def resolve_task_input_path(task):
    input_path = task.get('input_path')
    if isinstance(input_path, str) and os.path.exists(input_path):
        return input_path

    fallback_path = find_uploaded_file_path(task.get('file_id'))
    if fallback_path and os.path.exists(fallback_path):
        task['input_path'] = fallback_path
        return fallback_path

    return None


def get_original_upload_filename(file_id, input_path):
    basename = os.path.basename(input_path)
    prefix = f'{file_id}_'
    if file_id and basename.startswith(prefix):
        return basename[len(prefix):]
    return basename


def guess_video_mime_type(filename):
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'video/mp4'


def build_task_source_payload(task_id, task):
    input_path = resolve_task_input_path(task)
    if not input_path or not os.path.exists(input_path):
        raise FileNotFoundError('源视频文件不存在')

    file_id = task.get('file_id')
    filename = get_original_upload_filename(file_id, input_path)
    result = task.get('result') if isinstance(task.get('result'), dict) else {}
    effective_target_player_box = task.get('target_player_box')
    if not isinstance(effective_target_player_box, dict):
        effective_target_player_box = (
            result.get('targetPlayerBox')
            if isinstance(result.get('targetPlayerBox'), dict)
            else None
        )
    return {
        'taskId': task_id,
        'fileId': file_id,
        'filename': filename,
        'fileSize': os.path.getsize(input_path),
        'mimeType': guess_video_mime_type(filename),
        'sourceStreamUrl': f'/api/tasks/{task_id}/source/stream',
        'processingMode': task.get('processing_mode', 'auto'),
        'manualMoments': task.get('manual_moments') or [],
        'targetPlayerBox': effective_target_player_box,
    }


def delete_uploaded_files(file_id):
    removed_any = False
    for filename in find_uploaded_filenames(file_id):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            os.remove(file_path)
            removed_any = True
            logger.info(f"清理过期上传文件: {file_path}")
        except OSError as error:
            logger.warning(f"删除上传文件失败: {file_path}, error={error}")
    return removed_any


def should_generate_annotated_video(target_player_box):
    return DEBUG_KEEP_ARTIFACTS or bool(target_player_box)


def _normalize_frame_bbox(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> Optional[Tuple[int, int, int, int]]:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None

    x = max(0, min(int(x), max(frame_width - 1, 0)))
    y = max(0, min(int(y), max(frame_height - 1, 0)))
    width = max(1, min(int(width), max(frame_width - x, 0)))
    height = max(1, min(int(height), max(frame_height - y, 0)))
    if width <= 0 or height <= 0:
        return None

    return (x, y, width, height)


def _bbox_iou(
    first: Optional[Tuple[int, int, int, int]],
    second: Optional[Tuple[int, int, int, int]],
) -> float:
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
    union = aw * ah + bw * bh - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def _build_smart_selection_candidate_times(duration: float) -> List[float]:
    if duration <= 0:
        return [0.0]

    safe_duration = max(duration - 0.25, 0.0)
    times = {0.0}

    early_seconds = [0.4, 0.9, 1.5, 2.2, 3.2, 4.4, 6.0, 8.0, 10.5, 13.0]
    for candidate in early_seconds:
        if candidate <= safe_duration:
            times.add(round(candidate, 2))

    coverage_ratios = [0.02, 0.04, 0.07, 0.11, 0.16, 0.22, 0.3, 0.4, 0.55, 0.72, 0.88]
    for ratio in coverage_ratios:
        times.add(round(safe_duration * ratio, 2))

    ordered_times = sorted(times)
    if len(ordered_times) <= SMART_SELECTION_MAX_SAMPLE_TIMES:
        return ordered_times

    # 保留完整的前段密采样，再补充更稀疏的全段覆盖。
    kept = ordered_times[:10]
    tail = ordered_times[10:]
    if not tail:
        return kept

    step = max(len(tail) / max(SMART_SELECTION_MAX_SAMPLE_TIMES - len(kept), 1), 1.0)
    index = 0.0
    while len(kept) < SMART_SELECTION_MAX_SAMPLE_TIMES and int(round(index)) < len(tail):
        kept.append(tail[int(round(index))])
        index += step

    deduped = []
    for candidate_time in kept:
        if not deduped or abs(deduped[-1] - candidate_time) >= 0.2:
            deduped.append(candidate_time)
    return deduped


def _resize_frame_for_smart_selection(frame):
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= SMART_SELECTION_MAX_FRAME_WIDTH:
        return frame, 1.0

    scale = SMART_SELECTION_MAX_FRAME_WIDTH / max(frame_width, 1)
    resized = cv2.resize(
        frame,
        (
            int(round(frame_width * scale)),
            int(round(frame_height * scale)),
        ),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized, scale


def _score_smart_selection_bbox(
    bbox: Tuple[int, int, int, int],
    detector_weight: float,
    selection_time: float,
    duration: float,
    frame_shape,
) -> float:
    frame_height, frame_width = frame_shape[:2]
    _, _, width, height = bbox
    area_ratio = (width * height) / max(frame_width * frame_height, 1)
    height_ratio = height / max(frame_height, 1)
    aspect_ratio = width / max(height, 1)

    center_x = bbox[0] + width / 2.0
    center_y = bbox[1] + height / 2.0
    edge_margin = min(
        center_x / max(frame_width, 1),
        (frame_width - center_x) / max(frame_width, 1),
        center_y / max(frame_height, 1),
        (frame_height - center_y) / max(frame_height, 1),
    )

    detector_score = min(max(float(detector_weight), 0.0), 2.0) / 2.0
    size_score = min(1.0, height_ratio / 0.58) * 0.7 + min(1.0, area_ratio / 0.11) * 0.3
    aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 0.42) / 0.38)
    edge_score = min(edge_margin / 0.18, 1.0)
    earlier_score = 1.0 - min(selection_time / max(duration, 1e-6), 1.0)

    return round(
        size_score * 0.42
        + detector_score * 0.22
        + aspect_score * 0.12
        + edge_score * 0.10
        + earlier_score * 0.14,
        4,
    )


def _encode_frame_as_data_url(frame) -> str:
    encoded, buffer = cv2.imencode(
        '.jpg',
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), SMART_SELECTION_JPEG_QUALITY],
    )
    if not encoded:
        raise ValueError('无法编码候选截图')

    payload = base64.b64encode(buffer.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{payload}'


def build_selection_frame_candidates(
    file_id: str,
    input_path: str,
    max_candidates: int = SMART_SELECTION_MAX_CANDIDATES,
) -> List[Dict]:
    if not file_id or not input_path or not os.path.exists(input_path):
        return []

    cap = cv2.VideoCapture(input_path)
    is_opened = getattr(cap, 'isOpened', None)
    if not callable(is_opened) or not is_opened():
        return []

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or total_frames <= 0 or frame_width <= 0 or frame_height <= 0:
            return []

        duration = total_frames / fps
        sample_times = _build_smart_selection_candidate_times(duration)
        hog = TargetPlayerTracker._get_hog_detector()
        collected: List[Dict] = []

        for sample_time in sample_times:
            sample_frame = min(max(int(round(sample_time * fps)), 0), max(total_frames - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue

            detection_frame, scale = _resize_frame_for_smart_selection(frame)
            rects, weights = hog.detectMultiScale(
                detection_frame,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.05,
            )
            if len(rects) == 0:
                continue

            best_candidate = None
            for rect, weight in zip(rects, weights):
                x, y, width, height = [int(value) for value in rect]
                original_bbox = _normalize_frame_bbox(
                    (
                        int(round(x / scale)),
                        int(round(y / scale)),
                        int(round(width / scale)),
                        int(round(height / scale)),
                    ),
                    frame_width,
                    frame_height,
                )
                if original_bbox is None:
                    continue

                if original_bbox[3] < int(frame_height * 0.18) or original_bbox[2] < int(frame_width * 0.06):
                    continue

                candidate_score = _score_smart_selection_bbox(
                    original_bbox,
                    float(weight),
                    sample_time,
                    duration,
                    frame.shape,
                )
                if best_candidate is None or candidate_score > best_candidate['recommendationScore']:
                    best_candidate = {
                        'bbox': original_bbox,
                        'recommendationScore': candidate_score,
                    }

            if best_candidate is None:
                continue

            is_duplicate = any(
                abs(existing['time'] - sample_time) < 0.45
                or (
                    abs(existing['time'] - sample_time) < 1.25
                    and _bbox_iou(existing['bbox'], best_candidate['bbox']) >= 0.72
                )
                for existing in collected
            )
            if is_duplicate:
                continue

            x, y, width, height = best_candidate['bbox']
            collected.append({
                'imageUrl': _encode_frame_as_data_url(frame),
                'width': frame_width,
                'height': frame_height,
                'time': round(sample_time, 3),
                'frame': sample_frame,
                'source': 'smart',
                'recommended': True,
                'recommendationScore': best_candidate['recommendationScore'],
                'suggestedBox': {
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'frameWidth': frame_width,
                    'frameHeight': frame_height,
                    'selectionTime': round(sample_time, 3),
                    'selectionFrame': sample_frame,
                },
                'bbox': best_candidate['bbox'],
            })

        collected.sort(
            key=lambda candidate: (
                -float(candidate.get('recommendationScore') or 0.0),
                float(candidate.get('time') or 0.0),
            ),
        )

        trimmed = collected[:max_candidates]
        for candidate in trimmed:
            candidate.pop('bbox', None)
        return trimmed
    finally:
        cap.release()


def should_keep_annotated_video(target_player_box, diagnostics, tracking_summary, possible_highlights):
    if DEBUG_KEEP_ARTIFACTS:
        return True, 'debug'

    if not target_player_box or not AUTO_KEEP_REVIEW_ANNOTATIONS:
        return False, None

    coverage = float(tracking_summary.get('coverage') or 0.0)
    outcome = str(diagnostics.get('outcome') or '')

    if coverage < ANNOTATED_REVIEW_COVERAGE_THRESHOLD:
        return True, 'tracking_low_coverage'

    if int(possible_highlights or 0) > 0:
        return True, 'highlight_review'

    if outcome in ANNOTATED_REVIEW_OUTCOMES:
        return True, 'risk_review'

    return False, None

def save_task_metadata(task_id, metadata):
    metadata_path = os.path.join(TASK_METADATA_FOLDER, f'{task_id}.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata_path


def get_task_state_path(task_id):
    return os.path.join(TASK_METADATA_FOLDER, f'{task_id}.json')


def persist_task_state(task_id):
    task = processing_tasks.get(task_id)
    if not task:
        return None

    metadata_path = task.get('metadata_path') or get_task_state_path(task_id)
    task['metadata_path'] = metadata_path
    payload = {'taskId': task_id}
    for field in PERSISTED_TASK_FIELDS:
        if field in task:
            if field == 'result':
                payload[field] = normalize_result_payload(task.get(field))
            else:
                payload[field] = task.get(field)

    temp_path = f'{metadata_path}.tmp'
    with open(temp_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(payload, metadata_file, ensure_ascii=False, indent=2)
    os.replace(temp_path, metadata_path)
    return metadata_path


def delete_task_state(task_id):
    metadata_path = get_task_state_path(task_id)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)


def _coerce_task_timestamp(value):
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None

    return None


def _format_task_timestamp(value):
    timestamp = _coerce_task_timestamp(value)
    if timestamp is None:
        return None

    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace('+00:00', 'Z')


def _resolve_detection_progress(current_frame, total_frames, attempt_index, previous_progress=10):
    if total_frames <= 0:
        return int(previous_progress)

    safe_attempt_index = max(int(attempt_index or 1), 1)
    range_index = min(safe_attempt_index - 1, len(DETECTION_PROGRESS_ATTEMPT_RANGES) - 1)
    range_start, range_end = DETECTION_PROGRESS_ATTEMPT_RANGES[range_index]
    normalized = min(max(float(current_frame) / float(total_frames), 0.0), 1.0)
    span = max(range_end - range_start, 0)
    progress = range_start + int(normalized * span)
    return min(max(progress, int(previous_progress)), DETECTION_PROGRESS_ATTEMPT_RANGES[-1][1])


def build_detection_progress_callback(task_id):
    state = {
        'attempt_index': 1,
        'last_frame': -1,
        'max_progress': DETECTION_PROGRESS_ATTEMPT_RANGES[0][0],
    }

    def progress_callback(current_frame, total_frames):
        if task_id not in processing_tasks:
            return

        if state['last_frame'] >= 0 and current_frame < state['last_frame']:
            state['attempt_index'] += 1

        state['last_frame'] = int(current_frame)
        progress = _resolve_detection_progress(
            current_frame=current_frame,
            total_frames=total_frames,
            attempt_index=state['attempt_index'],
            previous_progress=state['max_progress'],
        )
        state['max_progress'] = progress

        if state['attempt_index'] > 1:
            stage = (
                f'正在自动补跑分析... '
                f'(第 {state["attempt_index"]} 轮 {current_frame}/{total_frames})'
            )
        else:
            stage = f'正在分析视频... ({current_frame}/{total_frames})'

        update_task_progress(task_id, progress=progress, stage=stage)

    return progress_callback


LEGACY_RESULT_TEXT_REPLACEMENTS = (
    ('待确认候选', '系统补充候选'),
    ('待确认回合', '系统补充回合'),
    ('待确认片段', '系统补充片段'),
)
CONFIRMED_HIGHLIGHT_ROLES = {'score', 'assist'}


def _normalize_legacy_result_payload(value):
    if isinstance(value, str):
        normalized = value
        for legacy_text, replacement in LEGACY_RESULT_TEXT_REPLACEMENTS:
            normalized = normalized.replace(legacy_text, replacement)
        return normalized

    if isinstance(value, list):
        return [_normalize_legacy_result_payload(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _normalize_legacy_result_payload(item)
            for key, item in value.items()
        }

    return value


def _coerce_result_int(value):
    return int(value) if isinstance(value, (int, float)) else 0


def _merge_unique_dict_items(items, identity_builder):
    merged = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        identity = identity_builder(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(item)
    return merged


def _clip_identity(clip):
    return (
        str(clip.get('filename') or ''),
        _coerce_result_int(clip.get('index')),
        str(clip.get('highlightRole') or ''),
    )


def _timestamp_identity(timestamp):
    return (
        _coerce_result_int(timestamp.get('frame')),
        round(float(timestamp.get('timestamp') or 0.0), 3),
        str(timestamp.get('highlight_role') or ''),
    )


def _split_highlight_dicts(items, role_key):
    confirmed_items = []
    debug_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get(role_key) or '')
        if role == 'possible':
            debug_items.append(item)
        else:
            confirmed_items.append(item)
    return confirmed_items, debug_items


def _count_role(items, role_key, role):
    return sum(1 for item in items if str(item.get(role_key) or '') == role)


def _normalize_result_delivery_payload(value):
    if not isinstance(value, dict):
        return value

    normalized = dict(value)

    raw_clips = normalized.get('clips') if isinstance(normalized.get('clips'), list) else []
    existing_debug_clips = (
        normalized.get('debugClips')
        if isinstance(normalized.get('debugClips'), list)
        else []
    )
    confirmed_clips, split_debug_clips = _split_highlight_dicts(raw_clips, 'highlightRole')
    debug_clips = _merge_unique_dict_items(
        [*existing_debug_clips, *split_debug_clips],
        _clip_identity,
    )
    normalized['clips'] = confirmed_clips
    normalized['debugClips'] = debug_clips

    raw_timestamps = normalized.get('timestamps') if isinstance(normalized.get('timestamps'), list) else []
    existing_debug_timestamps = (
        normalized.get('debugTimestamps')
        if isinstance(normalized.get('debugTimestamps'), list)
        else []
    )
    confirmed_timestamps, split_debug_timestamps = _split_highlight_dicts(raw_timestamps, 'highlight_role')
    debug_timestamps = _merge_unique_dict_items(
        [*existing_debug_timestamps, *split_debug_timestamps],
        _timestamp_identity,
    )
    normalized['timestamps'] = confirmed_timestamps
    normalized['debugTimestamps'] = debug_timestamps

    confirmed_scores = max(
        _coerce_result_int(normalized.get('targetScores')),
        _count_role(confirmed_clips, 'highlightRole', 'score'),
        _count_role(confirmed_timestamps, 'highlight_role', 'score'),
    )
    confirmed_assists = max(
        _coerce_result_int(normalized.get('targetAssists')),
        _count_role(confirmed_clips, 'highlightRole', 'assist'),
        _count_role(confirmed_timestamps, 'highlight_role', 'assist'),
    )
    confirmed_highlights = max(
        _coerce_result_int(normalized.get('targetHighlights')),
        confirmed_scores + confirmed_assists,
        len(confirmed_clips),
        len(confirmed_timestamps),
    )
    possible_highlights = max(
        _coerce_result_int(normalized.get('possibleHighlights')),
        _coerce_result_int(normalized.get('reviewCandidateHighlights')),
        len(debug_clips),
        len(debug_timestamps),
    )

    selection_summary = (
        dict(normalized.get('selectionSummary'))
        if isinstance(normalized.get('selectionSummary'), dict)
        else {}
    )
    selection_summary['confirmed'] = max(
        _coerce_result_int(selection_summary.get('confirmed')),
        confirmed_highlights,
    )
    selection_summary['possible'] = max(
        _coerce_result_int(selection_summary.get('possible')),
        possible_highlights,
    )
    normalized['selectionSummary'] = selection_summary

    normalized['targetScores'] = confirmed_scores
    normalized['targetAssists'] = confirmed_assists
    normalized['targetHighlights'] = selection_summary['confirmed']
    normalized['relatedHighlights'] = selection_summary['confirmed']
    normalized['possibleHighlights'] = selection_summary['possible']
    normalized['reviewCandidateHighlights'] = max(
        _coerce_result_int(normalized.get('reviewCandidateHighlights')),
        selection_summary['possible'],
    )

    pipeline = dict(normalized.get('pipeline')) if isinstance(normalized.get('pipeline'), dict) else {}
    attribution = dict(pipeline.get('attribution')) if isinstance(pipeline.get('attribution'), dict) else {}
    attribution['confirmedHighlights'] = selection_summary['confirmed']
    attribution['possibleHighlights'] = selection_summary['possible']
    attribution['confirmedScores'] = confirmed_scores
    attribution['confirmedAssists'] = confirmed_assists
    pipeline['attribution'] = attribution

    export = dict(pipeline.get('export')) if isinstance(pipeline.get('export'), dict) else {}
    export['selectedClipCount'] = len(confirmed_clips)
    export['selectedHighlights'] = selection_summary['confirmed']
    export['scoreClips'] = confirmed_scores
    export['assistClips'] = confirmed_assists
    export['possibleClips'] = selection_summary['possible']
    pipeline['export'] = export
    normalized['pipeline'] = pipeline

    return normalized


def normalize_result_payload(value):
    return _normalize_result_delivery_payload(_normalize_legacy_result_payload(value))


def build_progress_response(task):
    response = {
        'progress': task.get('progress', 0),
        'stage': task.get('stage', ''),
        'status': task.get('status', 'failed'),
        'completed': task.get('status') in TERMINAL_TASK_STATUSES,
        'processingMode': task.get('processing_mode', 'auto'),
        'manualMoments': task.get('manual_moments') or [],
        'createdAt': _format_task_timestamp(task.get('created_at')),
        'updatedAt': _format_task_timestamp(task.get('updated_at')),
    }

    if task.get('status') == 'completed' and task.get('result'):
        normalized_result = normalize_result_payload(task.get('result'))
        normalized_result['processingMode'] = task.get('processing_mode', normalized_result.get('processingMode', 'auto'))
        normalized_result['manualMoments'] = task.get('manual_moments') or normalized_result.get('manualMoments') or []
        task_target_player_box = task.get('target_player_box')
        if isinstance(task_target_player_box, dict):
            existing_target_player_box = (
                normalized_result.get('targetPlayerBox')
                if isinstance(normalized_result.get('targetPlayerBox'), dict)
                else None
            )
            if existing_target_player_box and existing_target_player_box != task_target_player_box:
                normalized_result.setdefault('effectiveTargetPlayerBox', existing_target_player_box)
            normalized_result['targetPlayerBox'] = task_target_player_box

        response['result'] = normalized_result
    elif task.get('status') == 'failed' and task.get('error'):
        response['error'] = task.get('error')

    return response


def _build_task_from_payload(task_id, payload):
    created_at = _coerce_task_timestamp(payload.get('created_at'))
    if created_at is None:
        created_at = _coerce_task_timestamp(payload.get('createdAt'))

    updated_at = _coerce_task_timestamp(payload.get('updated_at'))
    if updated_at is None:
        updated_at = created_at

    task = {
        'status': payload.get('status', 'failed'),
        'progress': payload.get('progress', 0),
        'stage': payload.get('stage', '任务状态未知'),
        'result': normalize_result_payload(payload.get('result')),
        'error': payload.get('error'),
        'created_at': created_at if created_at is not None else time.time(),
        'updated_at': updated_at if updated_at is not None else time.time(),
        'file_id': payload.get('file_id', payload.get('fileId')),
        'input_path': payload.get('input_path'),
        'before_seconds': payload.get('before_seconds', payload.get('beforeSeconds')),
        'after_seconds': payload.get('after_seconds', payload.get('afterSeconds')),
        'processing_mode': payload.get('processing_mode', payload.get('processingMode', 'auto')),
        'manual_moments': payload.get('manual_moments', payload.get('manualMoments')),
        'target_player_box': payload.get('target_player_box', payload.get('targetPlayerBox')),
        'effective_target_player_box': payload.get(
            'effective_target_player_box',
            payload.get('effectiveTargetPlayerBox'),
        ),
        'metadata_path': get_task_state_path(task_id),
    }
    return task


def load_persisted_task(task_id, mark_incomplete_as_failed=True):
    metadata_path = get_task_state_path(task_id)
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
        payload = json.load(metadata_file)

    task = _build_task_from_payload(task_id, payload if isinstance(payload, dict) else {})
    persisted_status = task.get('status')

    if mark_incomplete_as_failed and persisted_status not in TERMINAL_TASK_STATUSES:
        task['status'] = 'failed'
        task['stage'] = '处理已中断'
        task['error'] = '服务已重启，未完成的本地处理任务已中断，请重新发起处理'
        task['updated_at'] = time.time()

    processing_tasks[task_id] = task

    if (
        mark_incomplete_as_failed
        and persisted_status not in TERMINAL_TASK_STATUSES
    ):
        persist_task_state(task_id)

    return task


def get_task(task_id, load_if_missing=True):
    task = processing_tasks.get(task_id)
    if task or not load_if_missing:
        return task
    return load_persisted_task(task_id)

@app.errorhandler(413)
def request_entity_too_large(error):
    """处理文件过大错误"""
    logger.warning(f"文件上传失败: 文件过大")
    return jsonify({
        'success': False,
        'error': f'文件大小超过限制 ({MAX_FILE_SIZE / (1024*1024):.0f}MB)'
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    """处理服务器内部错误"""
    logger.error(f"服务器内部错误: {error}")
    return jsonify({
        'success': False,
        'error': '服务器内部错误，请稍后重试'
    }), 500

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上传视频文件"""
    
    try:
        logger.info("收到视频上传请求")
        
        if 'video' not in request.files:
            logger.warning("上传请求中未找到视频文件")
            return jsonify({
                'success': False,
                'error': '未找到视频文件'
            }), 400
        
        file = request.files['video']
        
        if file.filename == '':
            logger.warning("上传请求中文件名为空")
            return jsonify({
                'success': False,
                'error': '未选择文件'
            }), 400
        
        if not allowed_file(file.filename):
            logger.warning(f"不支持的文件格式: {file.filename}")
            return jsonify({
                'success': False,
                'error': '不支持的文件格式，请上传mp4、avi、mov或mkv格式'
            }), 400
        
        # 生成唯一文件ID
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        
        logger.info(f"开始保存文件: {filename} -> {file_path}")
        
        # 保存文件
        file.save(file_path)
        
        # 验证文件大小
        file_size = validate_file_size(file_path)
        
        logger.info(f"文件上传成功: {filename}, 大小: {file_size / (1024*1024):.1f}MB, ID: {file_id}")
        
        return jsonify({
            'success': True,
            'fileId': file_id,
            'filename': filename,
            'fileSize': file_size,
            'message': '视频上传成功'
        })
    
    except ValueError as e:
        # 文件大小验证错误
        logger.error(f"文件验证失败: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'上传失败: {str(e)}'
        }), 500

@app.route('/api/upload/init', methods=['POST'])
def init_upload():
    """初始化分块上传"""
    try:
        data = request.get_json(silent=True)
        if data is None:
             return jsonify({'success': False, 'error': 'Invalid JSON body'}), 400
             
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
            
        file_id = str(uuid.uuid4())
        upload_dir = os.path.join(app.config['CHUNKS_FOLDER'], file_id)
        os.makedirs(upload_dir, exist_ok=True)
        logger.info(f"分块上传已初始化: file_id={file_id}, upload_dir={upload_dir}")
        
        return jsonify({
            'success': True,
            'fileId': file_id,
            'message': 'Upload initialized'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Init upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/chunk', methods=['POST'])
def upload_chunk():
    """上传文件块"""
    try:
        if 'chunk' not in request.files:
            return jsonify({'success': False, 'error': 'No chunk file'}), 400
            
        chunk = request.files['chunk']
        file_id = request.form.get('fileId')
        chunk_index = request.form.get('chunkIndex')
        
        if not file_id or chunk_index is None:
            return jsonify({'success': False, 'error': 'Missing fileId or chunkIndex'}), 400
            
        save_path = os.path.join(app.config['CHUNKS_FOLDER'], file_id, chunk_index)
        chunk.save(save_path)
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Upload chunk failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload/complete', methods=['POST'])
def complete_upload():
    """完成分块上传并合并文件"""
    try:
        data = request.get_json()
        file_id = data.get('fileId')
        filename = data.get('filename')
        total_chunks = data.get('totalChunks')
        
        if not file_id or not filename or total_chunks is None:
            return jsonify({'success': False, 'error': 'Missing parameters'}), 400
            
        chunks_dir = os.path.join(app.config['CHUNKS_FOLDER'], file_id)
        if not os.path.exists(chunks_dir):
            return jsonify({'success': False, 'error': 'Upload session not found'}), 404
            
        # Verify all chunks exist
        for i in range(total_chunks):
            if not os.path.exists(os.path.join(chunks_dir, str(i))):
                return jsonify({'success': False, 'error': f'Missing chunk {i}'}), 400
        
        # Merge chunks
        safe_filename = secure_filename(filename)
        final_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{safe_filename}")
        
        with open(final_path, 'wb') as outfile:
            for i in range(total_chunks):
                chunk_path = os.path.join(chunks_dir, str(i))
                with open(chunk_path, 'rb') as infile:
                    shutil.copyfileobj(infile, outfile)
                    
        # Clean up chunks
        shutil.rmtree(chunks_dir)
        
        validate_file_size(final_path)
        file_size = os.path.getsize(final_path)
        logger.info(f"File merged successfully: {final_path}, size: {file_size}")
        
        return jsonify({
            'success': True,
            'fileId': file_id,
            'filename': safe_filename,
            'fileSize': file_size,
            'message': 'Upload completed successfully'
        })
        
    except Exception as e:
        logger.error(f"Complete upload failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_video():
    """启动视频处理任务"""
    
    try:
        logger.info("收到视频处理请求")
        
        data = request.get_json()
        
        if not data or 'fileId' not in data:
            logger.warning("处理请求中缺少文件ID")
            return jsonify({
                'success': False,
                'error': '缺少文件ID'
            }), 400
        
        file_id = data['fileId']
        before_seconds = data.get('beforeSeconds', DEFAULT_CLIP_BEFORE_SECONDS)
        after_seconds = data.get('afterSeconds', DEFAULT_CLIP_AFTER_SECONDS)
        requested_mode = str(data.get('mode') or '').strip().lower()
        processing_mode = requested_mode if requested_mode in {'manual', 'auto'} else (
            'manual' if data.get('manualMoments') else 'auto'
        )
        target_player_box = validate_target_player_box(data.get('targetPlayerBox'))
        
        # 验证参数
        if not isinstance(before_seconds, (int, float)) or before_seconds < 0 or before_seconds > 30:
            return jsonify({
                'success': False,
                'error': '片段前保留时间必须在0-30秒之间'
            }), 400
            
        if not isinstance(after_seconds, (int, float)) or after_seconds < 0 or after_seconds > 10:
            return jsonify({
                'success': False,
                'error': '片段后保留时间必须在0-10秒之间'
            }), 400
        
        uploaded_file_path = find_uploaded_file_path(file_id)
        uploaded_filename = os.path.basename(uploaded_file_path) if uploaded_file_path else None

        if not uploaded_file_path or not uploaded_filename:
            logger.warning(f"找不到文件ID对应的文件: {file_id}")
            return jsonify({
                'success': False,
                'error': '找不到上传的文件'
            }), 404

        input_path = uploaded_file_path
        
        # 验证文件是否存在且可读
        if not os.path.exists(input_path) or not os.access(input_path, os.R_OK):
            logger.error(f"文件不存在或不可读: {input_path}")
            return jsonify({
                'success': False,
                'error': '文件不存在或不可读'
            }), 404
        
        manual_moments = []
        if processing_mode == 'manual':
            video_metadata = get_video_metadata(input_path)
            manual_moments = validate_manual_moments(
                data.get('manualMoments'),
                video_metadata['duration'],
            )

        # 生成任务ID
        task_id = str(uuid.uuid4())
        task_metadata = {
            'taskId': task_id,
            'fileId': file_id,
            'beforeSeconds': before_seconds,
            'afterSeconds': after_seconds,
            'processingMode': processing_mode,
            'manualMoments': manual_moments,
            'targetPlayerBox': target_player_box,
            'createdAt': datetime.now().isoformat(),
        }
        metadata_path = save_task_metadata(task_id, task_metadata)
        
        logger.info(
            f"创建处理任务: {task_id}, 文件: {uploaded_filename}, mode={processing_mode}, "
            f"before={before_seconds}s, after={after_seconds}s"
        )
        
        # 初始化任务状态
        processing_tasks[task_id] = {
            'status': 'starting',
            'progress': 0,
            'stage': '准备处理',
            'result': None,
            'error': None,
            'created_at': time.time(),
            'updated_at': time.time(),
            'file_id': file_id,
            'input_path': input_path,
            'before_seconds': before_seconds,
            'after_seconds': after_seconds,
            'processing_mode': processing_mode,
            'manual_moments': manual_moments,
            'target_player_box': target_player_box,
            'metadata_path': metadata_path,
        }
        persist_task_state(task_id)
        
        # 启动后台处理线程
        thread = threading.Thread(
            target=process_video_background,
            args=(task_id, input_path, before_seconds, after_seconds)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'taskId': task_id,
            'message': '处理任务已启动'
        })
    
    except Exception as e:
        logger.error(f"启动处理任务失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'启动处理失败: {str(e)}'
        }), 500

def update_task_progress(task_id, **kwargs):
    """更新任务进度的辅助函数"""
    task = get_task(task_id, load_if_missing=False)
    if task is not None:
        normalized_updates = {
            key: normalize_result_payload(value) if key == 'result' else value
            for key, value in kwargs.items()
            if key != 'log'
        }
        task.update(normalized_updates)
        task['updated_at'] = time.time()
        logger.info(f"任务 {task_id} 进度更新: {kwargs}")
        persist_task_state(task_id)

        # 通过WebSocket发送进度更新
        try:
            task_data = build_progress_response(task)
            # 如果此次更新包含即时日志，不持久化但在事件中携带
            if 'log' in kwargs:
                task_data['log'] = kwargs['log']
            socketio.emit('task_progress', {
                'taskId': task_id,
                'data': task_data
            })
        except Exception as e:
            logger.error(f"WebSocket发送失败: {e}")


def build_manual_tracking_summary(total_frames=0):
    return {
        'enabled': False,
        'activeFrames': 0,
        'totalFrames': int(total_frames or 0),
        'coverage': 0.0,
        'missingFrames': 0,
        'lostFrames': 0,
        'reacquiredCount': 0,
        'guardedSwitches': 0,
        'latestStatus': 'disabled',
        'startFrame': 0,
        'startTime': 0.0,
        'primedReferenceSamples': 0,
        'runtimeReferenceSamples': 0,
        'error': None,
    }


def build_manual_shots(manual_moments, fps):
    shots = []
    safe_fps = float(fps or 0.0)
    for index, moment in enumerate(manual_moments, start=1):
        timestamp = round(float(moment), 3)
        frame = int(round(timestamp * safe_fps)) if safe_fps > 0 else 0
        shots.append({
            'frame': frame,
            'timestamp': timestamp,
            'made': False,
            'clip_export': True,
            'owner': 'manual',
            'owner_confidence': 1.0,
            'target_visible': True,
            'highlight_role': 'manual',
            'highlight_confidence': 1.0,
            'manual_index': index,
        })
    return shots


def process_manual_video_background(task_id, input_path, before_seconds, after_seconds):
    task = processing_tasks.get(task_id, {})
    manual_moments = task.get('manual_moments') or []
    if not manual_moments:
        raise ValueError('当前没有可处理的手动时间点')

    logger.info(f"开始手动剪辑任务: {task_id}, moments={manual_moments}")
    update_task_progress(
        task_id,
        status='generating',
        progress=15,
        stage='正在按你选择的时间点导出片段...',
    )

    metadata = get_video_metadata(input_path)
    manual_shots = build_manual_shots(manual_moments, metadata['fps'])

    for shot in manual_shots:
        update_task_progress(
            task_id,
            log=(
                f"手动时间点 #{shot['manual_index']} - "
                f"{float(shot['timestamp']):.2f}s"
            ),
        )

    output_filename = f"{task_id}_highlight.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    task_temp_dir = os.path.join(app.config['TEMP_FOLDER'], task_id)
    processor = VideoProcessor(temp_dir=task_temp_dir)
    video_result = processor.process_video_full_pipeline(
        video_path=input_path,
        timestamps=manual_shots,
        output_path=output_path,
        before=before_seconds,
        after=after_seconds,
        log_callback=lambda message: update_task_progress(task_id, log=message),
        clip_output_dir=app.config['OUTPUT_FOLDER'],
        clip_filename_prefix=f"{task_id}_clip",
        keep_clips=True,
    )

    if not video_result['success']:
        raise Exception(video_result['error'])

    clip_segments = [
        {
            'filename': clip['filename'],
            'index': clip['index'],
            'start': clip['start'],
            'end': clip['end'],
            'duration': clip['duration'],
            'shotFrame': clip['shot_frame'],
            'shotTimestamp': clip['shot_timestamp'],
            'highlightRole': 'manual',
            'candidateReason': None,
            'candidateSource': 'manual_selection',
            'highlightConfidence': 1.0,
        }
        for clip in video_result.get('clips', [])
    ]
    confirmed_clip_count = len(clip_segments)
    highlight_output_path = video_result.get('output_file')
    highlight_video_filename = (
        os.path.basename(highlight_output_path)
        if highlight_output_path and os.path.exists(highlight_output_path)
        else None
    )
    file_size = (
        os.path.getsize(highlight_output_path)
        if highlight_output_path and os.path.exists(highlight_output_path)
        else 0
    )
    completed_at_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

    update_task_progress(
        task_id,
        status='completed',
        progress=100,
        stage='处理完成',
        result={
            'processingMode': 'manual',
            'manualMoments': manual_moments,
            'totalShots': confirmed_clip_count,
            'madeShots': confirmed_clip_count,
            'targetShots': 0,
            'targetScores': 0,
            'targetAssists': 0,
            'targetHighlights': confirmed_clip_count,
            'possibleHighlights': 0,
            'relatedHighlights': confirmed_clip_count,
            'reviewCandidateHighlights': 0,
            'accuracy': 100 if confirmed_clip_count > 0 else 0,
            'highlightVideo': highlight_video_filename,
            'annotatedVideo': None,
            'timestamps': manual_shots,
            'debugTimestamps': [],
            'allMadeTimestamps': manual_shots,
            'clips': clip_segments,
            'debugClips': [],
            'fileSize': file_size,
            'targetPlayerBox': None,
            'effectiveTargetPlayerBox': None,
            'tracking': build_manual_tracking_summary(metadata['total_frames']),
            'message': (
                f"已按你选择的 {confirmed_clip_count} 个时间点导出片段。"
                if confirmed_clip_count > 0
                else '当前没有成功导出的片段，请检查时间点和源视频。'
            ),
            'selectionSummary': {
                'mode': 'manual',
                'confirmed': confirmed_clip_count,
                'possible': 0,
            },
            'diagnostics': {
                'outcome': 'manual_selection',
                'summary': '当前结果完全来自你手动选择的时间点，系统没有再做自动找球或人物归因。',
                'recommendedActions': [
                    '如果片段起止不合适，回首页调整前后保留时间再重跑。',
                ],
                'counts': {
                    'selectedClips': confirmed_clip_count,
                },
                'trackingCoverage': 0.0,
            },
            'pipeline': {
                'scan': {
                    'mode': 'manual_timestamps',
                    'fullVideoScanned': False,
                    'trackerEnabled': False,
                    'trackingStartTime': None,
                    'trackingStartFrame': None,
                    'totalShotEvents': len(manual_moments),
                    'madeShotEvents': len(manual_moments),
                    'targetVisibleEvents': 0,
                },
                'attribution': {
                    'selectionMode': 'manual',
                    'confirmedHighlights': confirmed_clip_count,
                    'possibleHighlights': 0,
                    'confirmedScores': 0,
                    'confirmedAssists': 0,
                    'reviewCandidates': 0,
                    'trackingCoverage': 0.0,
                },
                'export': {
                    'selectedClipCount': confirmed_clip_count,
                    'selectedHighlights': confirmed_clip_count,
                    'clipWindowBeforeSeconds': before_seconds,
                    'clipWindowAfterSeconds': after_seconds,
                    'scoreClips': 0,
                    'assistClips': 0,
                    'possibleClips': 0,
                },
            },
            'completed_at': completed_at_iso,
        },
    )

def process_video_background(task_id, input_path, before_seconds, after_seconds):
    """后台处理视频的函数"""
    
    try:
        logger.info(f"开始后台处理任务: {task_id}")
        task = processing_tasks.get(task_id, {})
        if str(task.get('processing_mode') or 'auto') == 'manual':
            process_manual_video_background(task_id, input_path, before_seconds, after_seconds)
            return
        
        # 更新状态：开始检测
        update_task_progress(task_id, 
            status='detecting',
            progress=10,
            stage='正在检测进球时刻...'
        )
        
        # 检查模型文件是否存在
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"AI模型文件不存在: {model_path}")
        
        # 初始化检测器
        logger.info(f"初始化篮球检测器，模型: {model_path}")
        detector = BasketballShotDetector(model_path=model_path)
        task = processing_tasks.get(task_id, {})
        target_player_box = task.get('target_player_box')
        
        # 进度回调函数
        progress_callback = build_detection_progress_callback(task_id)
        
        # 检测进球
        logger.info(f"开始检测进球，文件: {input_path}")
        annotated_output_path = None
        annotate_video = should_generate_annotated_video(target_player_box)
        if annotate_video:
            annotated_filename = f"{task_id}_annotated.mp4"
            annotated_output_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_filename)

        result = detector.detect_shots_with_clips(
            input_path,
            before_seconds=before_seconds,
            after_seconds=after_seconds,
            progress_callback=progress_callback,
            annotate=annotate_video,
            annotated_output_path=annotated_output_path,
            target_player_box=target_player_box,
        )
        update_task_progress(
            task_id,
            status='attributing',
            progress=72,
            stage='正在归因目标球员并整理相关片段...',
        )

        selected_shots = result.get('selected_shots', result.get('selected_made_shots', result.get('made_shots', [])))
        confirmed_shots, debug_shots = _split_highlight_dicts(selected_shots, 'highlight_role')
        tracking_summary = result.get('tracking', {'enabled': False})
        target_scores = result['stats'].get('target_scores', 0)
        target_assists = result['stats'].get('target_assists', 0)
        target_highlights = result['stats'].get('target_highlights', target_scores + target_assists)
        possible_highlights = result['stats'].get('possible_highlights', 0)
        related_highlights = result['stats'].get('related_highlights', len(confirmed_shots))
        review_candidate_highlights = result['stats'].get('review_candidate_highlights', 0)
        selection_summary = result.get('selection_summary', {})
        diagnostics = result.get('diagnostics', {})
        pipeline_summary = result.get('pipeline', {})
        auto_retry = result.get('auto_retry')
        effective_target_player_box = target_player_box or result.get('target_player_box')
        if (
            target_player_box
            and result.get('target_player_box')
            and result.get('target_player_box') != target_player_box
        ):
            logger.warning("检测结果返回了不同的目标人物框，已回退到用户原始框选")
        if effective_target_player_box is not None:
            update_task_progress(task_id, effective_target_player_box=effective_target_player_box)
        keep_annotated_video, annotated_video_reason = should_keep_annotated_video(
            target_player_box=effective_target_player_box,
            diagnostics=diagnostics,
            tracking_summary=tracking_summary,
            possible_highlights=possible_highlights,
        )
        annotated_video_path = result.get('annotated_video')
        if annotated_video_path and not keep_annotated_video:
            try:
                if os.path.exists(annotated_video_path):
                    os.remove(annotated_video_path)
                    logger.info(f"删除无需保留的标注视频: {annotated_video_path}")
            except OSError as error:
                logger.warning(f"删除标注视频失败: {annotated_video_path}, error={error}")
            finally:
                annotated_video_path = None
                result['annotated_video'] = None

        annotated_video_filename = (
            os.path.basename(annotated_video_path)
            if annotated_video_path
            else None
        )
        debug_result = {
            'debugArtifactsKept': DEBUG_KEEP_ARTIFACTS,
        }
        if DEBUG_KEEP_ARTIFACTS:
            debug_result['allShots'] = result.get('shots', [])
        if annotated_video_filename:
            debug_result['annotatedVideoReason'] = annotated_video_reason

        if selected_shots:
            selected_shot_lookup = {
                (
                    int(shot.get('frame') or 0),
                    round(float(shot.get('timestamp') or 0.0), 3),
                ): shot
                for shot in selected_shots
            }
            for i, t in enumerate(selected_shots, start=1):
                try:
                    role = t.get('highlight_role')
                    role_label = (
                        '目标球员进球'
                        if role == 'score'
                        else '目标球员助攻'
                        if role == 'assist'
                        else '系统补充片段'
                    )
                    msg = (
                        f"检测到高光 #{i} - 帧: {t['frame']}, 时间: {float(t['timestamp']):.2f}s, {role_label}"
                    )
                    update_task_progress(task_id, log=msg)
                except Exception:
                    pass
            if review_candidate_highlights > 0 and target_highlights == 0:
                update_task_progress(
                    task_id,
                    log=f"当前没有确认到目标球员进球或助攻，已额外保留 {review_candidate_highlights} 个系统补充回合供你快速检查",
                )
        elif target_player_box and result.get('made_shots'):
            update_task_progress(task_id, log='检测到全场进球，当前正在尽量保留可能与目标球员相关的片段')
        
        logger.info(f"检测完成，结果: 总投篮 {result['stats']['total_attempts']}, 进球 {result['stats']['total_makes']}, 命中率 {result['stats']['accuracy']:.1f}%")
        
        # 更新状态：开始生成集锦
        update_task_progress(task_id,
            status='generating',
            progress=75,
            stage='正在生成集锦视频...'
        )
        
        # 生成输出文件名
        output_filename = f"{task_id}_highlight.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # 处理视频：生成集锦
        task_temp_dir = os.path.join(app.config['TEMP_FOLDER'], task_id)
        processor = VideoProcessor(temp_dir=task_temp_dir)
        
        if selected_shots:
            # 有进球，生成集锦
            logger.info(f"生成集锦视频，片段数量: {len(selected_shots)}")
            # 集锦始终从原视频剪辑，保留原始音频；标注视频仅用于调试。
            source_video_for_clips = input_path
            has_confirmed_highlights = any(
                str(shot.get('highlight_role') or '') in {'score', 'assist'}
                for shot in selected_shots
            )
            video_result = processor.process_video_full_pipeline(
                video_path=source_video_for_clips,
                timestamps=selected_shots,
                output_path=output_path,
                before=before_seconds,
                after=after_seconds,
                log_callback=lambda m: update_task_progress(task_id, log=m),
                clip_output_dir=app.config['OUTPUT_FOLDER'],
                clip_filename_prefix=f"{task_id}_clip",
                keep_clips=True,
            )
            
            if not video_result['success']:
                raise Exception(video_result['error'])

            highlight_output_path = video_result.get('output_file')
            if has_confirmed_highlights:
                if not highlight_output_path or not os.path.exists(highlight_output_path):
                    raise Exception("已确认片段拼接视频生成失败，输出文件不存在")
                file_size = os.path.getsize(highlight_output_path)
                output_filename = os.path.basename(highlight_output_path)
                logger.info(f"集锦视频生成成功: {highlight_output_path}, 大小: {file_size / (1024*1024):.1f}MB")
            else:
                output_filename = None
                file_size = 0
                logger.info("当前没有已确认的进球或助攻片段，仅导出独立片段供验收")

            clip_segments = [
                {
                    'filename': clip['filename'],
                    'index': clip['index'],
                    'start': clip['start'],
                    'end': clip['end'],
                    'duration': clip['duration'],
                    'shotFrame': clip['shot_frame'],
                    'shotTimestamp': clip['shot_timestamp'],
                    'highlightRole': clip['highlight_role'],
                    'candidateReason': clip.get('candidate_reason'),
                    'candidateSource': clip.get('candidate_source')
                    or selected_shot_lookup.get(
                        (
                            int(clip.get('shot_frame') or 0),
                            round(float(clip.get('shot_timestamp') or 0.0), 3),
                        ),
                        {},
                    ).get('candidate_source'),
                    'highlightConfidence': clip.get('highlight_confidence')
                    or selected_shot_lookup.get(
                        (
                            int(clip.get('shot_frame') or 0),
                            round(float(clip.get('shot_timestamp') or 0.0), 3),
                        ),
                        {},
                    ).get('highlight_confidence'),
                }
                for clip in video_result.get('clips', [])
            ]
            confirmed_clips, debug_clips = _split_highlight_dicts(clip_segments, 'highlightRole')
            
            confirmed_clip_count = len(confirmed_clips)
            debug_clip_count = len(debug_clips)

            if debug_clip_count > 0 and target_highlights == 0:
                result_message = (
                    f"当前还没有确认到目标球员进球或助攻。系统已把 {debug_clip_count} 个系统补充回合放到高级排错区，供你按需检查。"
                )
            elif debug_clip_count == 0:
                result_message = f"已自动导出 {confirmed_clip_count} 个已确认片段。"
            else:
                result_message = (
                    f"已自动导出 {confirmed_clip_count} 个已确认片段。另有 {debug_clip_count} 个系统补充回合已移入高级排错区，只有怀疑漏剪时再看。"
                )
            completed_at_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

            # 更新状态：完成
            update_task_progress(task_id,
                status='completed',
                progress=100,
                stage='处理完成',
                result={
                    'totalShots': result['stats']['total_attempts'],
                    'madeShots': result['stats']['total_makes'],
                    'accuracy': result['stats']['accuracy'],
                    'targetShots': target_scores,
                    'targetScores': target_scores,
                    'targetAssists': target_assists,
                    'targetHighlights': target_highlights,
                    'possibleHighlights': debug_clip_count,
                    'relatedHighlights': confirmed_clip_count,
                    'reviewCandidateHighlights': review_candidate_highlights,
                    'highlightVideo': output_filename,
                    'annotatedVideo': annotated_video_filename,
                    'timestamps': confirmed_shots,
                    'debugTimestamps': debug_shots,
                    'allMadeTimestamps': result['made_shots'],
                    'clips': confirmed_clips,
                    'debugClips': debug_clips,
                    'fileSize': file_size,
                    'targetPlayerBox': target_player_box,
                    'effectiveTargetPlayerBox': effective_target_player_box,
                    'tracking': tracking_summary,
                    'message': result_message,
                    'selectionSummary': selection_summary,
                    'diagnostics': diagnostics,
                    'pipeline': pipeline_summary,
                    'autoRetry': auto_retry,
                    'completed_at': completed_at_iso,
                    **debug_result,
                }
            )
        else:
            # 没有检测到进球
            logger.info("未检测到可用于集锦的相关片段")
            message = '未检测到进球，请检查视频内容或调整参数'
            if target_player_box:
                if result['made_shots']:
                    message = '检测到全场进球，但当前还没有锁定与目标球员相关的片段，请检查框选时机或人物清晰度'
                else:
                    message = '人物跟踪正常，但当前未识别出全场进球，请检查机位、清晰度或进球识别规则'
            if diagnostics.get('summary'):
                message = diagnostics['summary']
            completed_at_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            update_task_progress(task_id,
                status='completed',
                progress=100,
                stage='处理完成',
                result={
                    'totalShots': result['stats']['total_attempts'],
                    'madeShots': result['stats']['total_makes'],
                    'accuracy': result['stats']['accuracy'],
                    'targetShots': target_scores,
                    'targetScores': target_scores,
                    'targetAssists': target_assists,
                    'targetHighlights': target_highlights,
                    'possibleHighlights': possible_highlights,
                    'relatedHighlights': len(confirmed_shots),
                    'reviewCandidateHighlights': review_candidate_highlights,
                    'highlightVideo': None,
                    'annotatedVideo': annotated_video_filename,
                    'message': message,
                    'targetPlayerBox': target_player_box,
                    'tracking': tracking_summary,
                    'timestamps': confirmed_shots,
                    'debugTimestamps': debug_shots,
                    'allMadeTimestamps': result['made_shots'],
                    'clips': [],
                    'debugClips': [],
                    'selectionSummary': selection_summary,
                    'effectiveTargetPlayerBox': effective_target_player_box,
                    'diagnostics': diagnostics,
                    'pipeline': pipeline_summary,
                    'autoRetry': auto_retry,
                    'completed_at': completed_at_iso,
                    **debug_result,
                }
            )

        if DEBUG_KEEP_ARTIFACTS:
            logger.info("DEBUG_KEEP_ARTIFACTS 已开启，保留上传文件用于排查")
        else:
            logger.info("保留上传文件，支持后续重新框选并重跑")
            
    except Exception as e:
        # 处理错误
        error_msg = str(e)
        logger.error(f"任务 {task_id} 处理失败: {error_msg}")
        update_task_progress(task_id,
            status='failed',
            progress=0,
            stage='处理失败',
            error=error_msg
        )

@app.route('/api/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    """获取处理进度"""
    
    try:
        task = get_task(task_id)
        if task is None:
            logger.warning(f"查询不存在的任务: {task_id}")
            return jsonify({
                'success': False,
                'error': '任务不存在'
            }), 404

        return jsonify(build_progress_response(task))
    
    except Exception as e:
        logger.error(f"获取任务进度失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取进度失败'
        }), 500


@app.route('/api/upload/candidates/<file_id>', methods=['GET'])
def get_upload_selection_candidates(file_id):
    """为已上传视频生成智能推荐截图，前端本地候选仍可作为兜底。"""
    try:
        if not file_id:
            return jsonify({'success': False, 'error': '缺少 fileId'}), 400

        input_path = find_uploaded_file_path(file_id)
        if not input_path or not os.path.exists(input_path):
            return jsonify({'success': False, 'error': '找不到上传的文件'}), 404

        candidates = build_selection_frame_candidates(file_id, input_path)
        return jsonify({
            'success': True,
            'fileId': file_id,
            'candidateFrames': candidates,
        })
    except Exception as error:
        logger.error(f"生成智能推荐截图失败: file_id={file_id}, error={error}")
        return jsonify({
            'success': False,
            'error': '生成智能推荐截图失败',
        }), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_video(filename):
    """下载生成的集锦视频"""
    
    try:
        # 安全检查文件名
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            logger.warning(f"非法文件名访问: {filename}")
            return jsonify({
                'success': False,
                'error': '非法文件名'
            }), 400
        
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"下载文件不存在: {file_path}")
            return jsonify({
                'success': False,
                'error': '文件不存在'
            }), 404
        
        logger.info(f"开始下载文件: {filename}")
        
        return send_file(
            file_path, 
            as_attachment=True,
            download_name=f"basketball_highlight_{filename}"
        )
    
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': '下载失败'
        }), 500

@app.route('/api/download/clips/<task_id>', methods=['POST'])
def download_selected_clips(task_id):
    """将用户选择的高光片段打包下载。"""
    try:
        task = get_task(task_id)
        if not task or task.get('status') != 'completed' or not task.get('result'):
            return jsonify({'success': False, 'error': '任务结果不存在'}), 404

        normalized_result = normalize_result_payload(task.get('result') or {})
        confirmed_clips = normalized_result.get('clips', [])
        debug_clips = normalized_result.get('debugClips', [])

        data = request.get_json(silent=True) or {}
        requested_filenames = data.get('filenames')
        archive_scope = str(data.get('scope') or '').strip().lower()

        if isinstance(requested_filenames, list) and requested_filenames:
            requested_set = {
                filename for filename in requested_filenames
                if isinstance(filename, str)
            }
            clip_pool = [*confirmed_clips, *debug_clips]
            selected_clips = [
                clip for clip in clip_pool
                if clip.get('filename') in requested_set
            ]
        elif archive_scope == 'debug':
            selected_clips = list(debug_clips)
        elif archive_scope == 'all':
            selected_clips = [*confirmed_clips, *debug_clips]
        else:
            selected_clips = list(confirmed_clips)

        if not selected_clips:
            return jsonify({'success': False, 'error': '所选片段不存在'}), 404

        archive_buffer = io.BytesIO()
        archived_count = 0
        archive_group_counts = {
            'score': 0,
            'assist': 0,
            'manual': 0,
            'review': 0,
            'other': 0,
        }
        manifest_clips = []
        with zipfile.ZipFile(archive_buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
            for clip in selected_clips:
                filename = clip['filename']
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if not os.path.exists(file_path):
                    continue

                highlight_role = str(clip.get('highlightRole') or 'other')
                if highlight_role == 'score':
                    archive_group = 'score'
                    archive_filename = f"score_{int(clip.get('index', 0)):03d}.mp4"
                elif highlight_role == 'assist':
                    archive_group = 'assist'
                    archive_filename = f"assist_{int(clip.get('index', 0)):03d}.mp4"
                elif highlight_role == 'manual':
                    archive_group = 'manual'
                    archive_filename = f"manual_{int(clip.get('index', 0)):03d}.mp4"
                elif highlight_role == 'possible':
                    archive_group = 'review'
                    archive_filename = f"review_{int(clip.get('index', 0)):03d}.mp4"
                else:
                    archive_group = 'other'
                    archive_filename = f"clip_{int(clip.get('index', 0)):03d}.mp4"

                archive_name = f"{archive_group}/{archive_filename}"
                archive.write(file_path, arcname=archive_name)
                manifest_clips.append({
                    'archiveName': archive_name,
                    'archiveGroup': archive_group,
                    'filename': filename,
                    'index': int(clip.get('index', 0) or 0),
                    'start': clip.get('start'),
                    'end': clip.get('end'),
                    'duration': clip.get('duration'),
                    'shotFrame': clip.get('shotFrame'),
                    'shotTimestamp': clip.get('shotTimestamp'),
                    'highlightRole': clip.get('highlightRole'),
                    'candidateReason': clip.get('candidateReason'),
                    'candidateSource': clip.get('candidateSource'),
                    'highlightConfidence': clip.get('highlightConfidence'),
                })
                archived_count += 1
                archive_group_counts[archive_group] = archive_group_counts.get(archive_group, 0) + 1

            manifest_scope = (
                'confirmed'
                if archive_group_counts['review'] == 0 and (
                    archive_group_counts['score'] > 0
                    or archive_group_counts['assist'] > 0
                    or archive_group_counts['manual'] > 0
                )
                else 'debug'
                if (
                    archive_group_counts['score'] == 0
                    and archive_group_counts['assist'] == 0
                    and archive_group_counts['manual'] == 0
                )
                else 'mixed'
            )
            manifest = {
                'taskId': task_id,
                'exportedAt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'archiveScope': manifest_scope,
                'selectedClipCount': archived_count,
                'clipGroups': archive_group_counts,
                'selectionSummary': normalized_result.get('selectionSummary'),
                'diagnostics': normalized_result.get('diagnostics'),
                'pipeline': normalized_result.get('pipeline'),
                'autoRetry': normalized_result.get('autoRetry'),
                'clips': manifest_clips,
            }
            archive.writestr(
                'manifest.json',
                json.dumps(manifest, ensure_ascii=False, indent=2),
            )

        if archived_count == 0:
            return jsonify({'success': False, 'error': '所选片段文件不存在'}), 404

        archive_buffer.seek(0)
        return send_file(
            archive_buffer,
            as_attachment=True,
            download_name=f"hoopcut_related_clips_{task_id}.zip",
            mimetype='application/zip',
        )
    except Exception as e:
        logger.error(f"片段打包下载失败: {str(e)}")
        return jsonify({'success': False, 'error': '片段打包下载失败'}), 500


@app.route('/api/tasks/<task_id>/source', methods=['GET'])
def get_task_source(task_id):
    """获取任务对应的可复用源视频信息。"""
    try:
        task = get_task(task_id)
        if task is None:
            return jsonify({'success': False, 'error': '任务不存在'}), 404

        payload = build_task_source_payload(task_id, task)
        return jsonify({
            'success': True,
            **payload,
        })
    except FileNotFoundError as error:
        logger.warning(f"任务源视频不存在: task_id={task_id}, error={error}")
        return jsonify({'success': False, 'error': str(error)}), 404
    except Exception as error:
        logger.error(f"获取任务源视频失败: task_id={task_id}, error={error}")
        return jsonify({'success': False, 'error': '获取任务源视频失败'}), 500


@app.route('/api/tasks/<task_id>/source/stream', methods=['GET'])
def stream_task_source(task_id):
    """流式返回任务原始上传视频，用于重新框选并重跑。"""
    try:
        task = get_task(task_id)
        if task is None:
            return jsonify({'success': False, 'error': '任务不存在'}), 404

        input_path = resolve_task_input_path(task)
        if not input_path or not os.path.exists(input_path):
            return jsonify({'success': False, 'error': '源视频文件不存在'}), 404

        filename = get_original_upload_filename(task.get('file_id'), input_path)
        return send_file(
            input_path,
            as_attachment=False,
            mimetype=guess_video_mime_type(filename),
            conditional=True,
        )
    except Exception as error:
        logger.error(f"源视频流媒体传输失败: task_id={task_id}, error={error}")
        return jsonify({'success': False, 'error': '源视频流媒体传输失败'}), 500

@app.route('/api/stream/<filename>', methods=['GET'])
def stream_video(filename):
    try:
        if not filename or '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({'success': False, 'error': '非法文件名'}), 400

        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': '文件不存在'}), 404

        return send_file(file_path, as_attachment=False, mimetype='video/mp4', conditional=True)
    except Exception:
        return jsonify({'success': False, 'error': '流媒体传输失败'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        # 检查关键组件
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'message': '篮球集锦生成服务运行正常',
            'components': {
                'upload_folder': os.path.exists(UPLOAD_FOLDER),
                'output_folder': os.path.exists(OUTPUT_FOLDER),
                'model_file': os.path.exists(MODEL_PATH),
                'active_tasks': len(processing_tasks)
            },
            'runtime': {
                'apiVersion': API_RUNTIME_VERSION,
                'backendHost': BACKEND_HOST,
                'backendPort': BACKEND_PORT,
                'frontendPort': FRONTEND_PORT,
                'startedAt': SERVER_STARTED_AT.isoformat().replace('+00:00', 'Z'),
                'supports': {
                    'uploadCandidates': True,
                    'scopedClipArchive': True,
                },
            },
        }
        
        # 检查是否有组件异常
        storage_ok = (
            health_status['components']['upload_folder']
            and health_status['components']['output_folder']
            and health_status['components']['model_file']
        )
        if not storage_ok:
            health_status['status'] = 'degraded'
            health_status['message'] = '部分组件异常'
        
        return jsonify(health_status)
    
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'message': f'健康检查失败: {str(e)}'
        }), 500

# 清理已结束任务的定时任务
def cleanup_old_tasks():
    """清理保留期已过的已结束任务，运行中的长任务不参与清理。"""
    try:
        current_time = time.time()
        all_tasks = {
            task_id: dict(task)
            for task_id, task in processing_tasks.items()
        }
        expired_tasks = set(
            task_id for task_id, task in all_tasks.items()
            if task.get('status') in TERMINAL_TASK_STATUSES
            and current_time - task.get('updated_at', task.get('created_at', current_time)) > TASK_RETENTION_SECONDS
        )

        for filename in os.listdir(TASK_METADATA_FOLDER):
            if not filename.endswith('.json'):
                continue

            task_id = filename[:-5]
            if task_id in expired_tasks:
                continue

            metadata_path = os.path.join(TASK_METADATA_FOLDER, filename)
            try:
                with open(metadata_path, 'r', encoding='utf-8') as metadata_file:
                    payload = json.load(metadata_file)
            except (OSError, json.JSONDecodeError) as error:
                logger.warning(f"读取任务状态失败，跳过清理: {metadata_path}, error={error}")
                continue

            task = _build_task_from_payload(task_id, payload if isinstance(payload, dict) else {})
            all_tasks.setdefault(task_id, task)
            if task.get('status') not in TERMINAL_TASK_STATUSES:
                continue

            updated_at = task.get('updated_at', task.get('created_at', current_time))
            if current_time - updated_at > TASK_RETENTION_SECONDS:
                expired_tasks.add(task_id)

        referenced_file_ids = {
            task.get('file_id')
            for task_id, task in all_tasks.items()
            if task_id not in expired_tasks and task.get('file_id')
        }
        expired_file_ids = {
            task.get('file_id')
            for task_id, task in all_tasks.items()
            if task_id in expired_tasks and task.get('file_id')
        }
        
        for task_id in expired_tasks:
            if processing_tasks.pop(task_id, None) is not None:
                logger.info(f"清理过期任务: {task_id}")
            try:
                delete_task_state(task_id)
            except OSError as error:
                logger.warning(f"删除任务状态文件失败: {task_id}, error={error}")

        for file_id in expired_file_ids:
            if file_id not in referenced_file_ids:
                delete_uploaded_files(file_id)
        
        if expired_tasks:
            logger.info(f"清理了 {len(expired_tasks)} 个过期任务")
    
    except Exception as e:
        logger.error(f"清理过期任务失败: {str(e)}")

# 启动清理定时器
def start_cleanup_timer():
    cleanup_old_tasks()
    timer = threading.Timer(1800, start_cleanup_timer)  # 每30分钟清理一次
    timer.daemon = True
    timer.start()

if __name__ == '__main__':
    start_cleanup_timer()
    logger.info("Basketball Highlight Generator Service Starting...")
    logger.info(f"API Service: http://{BACKEND_HOST}:{BACKEND_PORT}")
    logger.info(f"Health Check: http://{BACKEND_HOST}:{BACKEND_PORT}/api/health")
    socketio.run(
        app,
        debug=BACKEND_DEBUG,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        allow_unsafe_werkzeug=True,
        use_reloader=BACKEND_DEBUG,
    )
