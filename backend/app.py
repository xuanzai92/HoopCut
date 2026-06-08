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
from datetime import datetime
from shot_detector_video import BasketballShotDetector
from video_processor import VideoProcessor

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
TASK_RETENTION_SECONDS = int(os.getenv('TASK_RETENTION_SECONDS', str(24 * 60 * 60)))

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

def save_task_metadata(task_id, metadata):
    metadata_path = os.path.join(TASK_METADATA_FOLDER, f'{task_id}.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
    return metadata_path

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
        before_seconds = data.get('beforeSeconds', 3)
        after_seconds = data.get('afterSeconds', 1)
        target_player_box = validate_target_player_box(data.get('targetPlayerBox'))
        
        # 验证参数
        if not isinstance(before_seconds, (int, float)) or before_seconds < 1 or before_seconds > 30:
            return jsonify({
                'success': False,
                'error': '进球前保留时间必须在1-30秒之间'
            }), 400
            
        if not isinstance(after_seconds, (int, float)) or after_seconds < 1 or after_seconds > 10:
            return jsonify({
                'success': False,
                'error': '进球后保留时间必须在1-10秒之间'
            }), 400
        
        # 查找上传的文件
        uploaded_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
        
        if not uploaded_files:
            logger.warning(f"找不到文件ID对应的文件: {file_id}")
            return jsonify({
                'success': False,
                'error': '找不到上传的文件'
            }), 404
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_files[0])
        
        # 验证文件是否存在且可读
        if not os.path.exists(input_path) or not os.access(input_path, os.R_OK):
            logger.error(f"文件不存在或不可读: {input_path}")
            return jsonify({
                'success': False,
                'error': '文件不存在或不可读'
            }), 404
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        task_metadata = {
            'taskId': task_id,
            'fileId': file_id,
            'beforeSeconds': before_seconds,
            'afterSeconds': after_seconds,
            'targetPlayerBox': target_player_box,
            'createdAt': datetime.now().isoformat(),
        }
        metadata_path = save_task_metadata(task_id, task_metadata)
        
        logger.info(f"创建处理任务: {task_id}, 文件: {uploaded_files[0]}, 参数: before={before_seconds}s, after={after_seconds}s")
        
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
            'target_player_box': target_player_box,
            'metadata_path': metadata_path,
        }
        
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
    if task_id in processing_tasks:
        processing_tasks[task_id].update(kwargs)
        processing_tasks[task_id]['updated_at'] = time.time()
        logger.info(f"任务 {task_id} 进度更新: {kwargs}")
        
        # 通过WebSocket发送进度更新
        try:
            task_data = processing_tasks[task_id].copy()
            # 添加完成状态标记
            task_data['completed'] = task_data['status'] in ['completed', 'failed']
            # 如果此次更新包含即时日志，不持久化但在事件中携带
            if 'log' in kwargs:
                task_data['log'] = kwargs['log']
            socketio.emit('task_progress', {
                'taskId': task_id,
                'data': task_data
            })
        except Exception as e:
            logger.error(f"WebSocket发送失败: {e}")

def process_video_background(task_id, input_path, before_seconds, after_seconds):
    """后台处理视频的函数"""
    
    try:
        logger.info(f"开始后台处理任务: {task_id}")
        
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
        def progress_callback(current_frame, total_frames):
            if task_id in processing_tasks:
                progress = 10 + int((current_frame / total_frames) * 60)  # 10-70%
                update_task_progress(task_id,
                    progress=progress,
                    stage=f'正在分析视频... ({current_frame}/{total_frames})'
                )
        
        # 检测进球
        logger.info(f"开始检测进球，文件: {input_path}")
        annotated_output_path = None
        if DEBUG_KEEP_ARTIFACTS:
            annotated_filename = f"{task_id}_annotated.mp4"
            annotated_output_path = os.path.join(app.config['OUTPUT_FOLDER'], annotated_filename)

        result = detector.detect_shots_with_clips(
            input_path,
            before_seconds=before_seconds,
            after_seconds=after_seconds,
            progress_callback=progress_callback,
            annotate=DEBUG_KEEP_ARTIFACTS,
            annotated_output_path=annotated_output_path,
            target_player_box=target_player_box,
        )

        selected_made_shots = result.get('selected_made_shots', result.get('made_shots', []))
        tracking_summary = result.get('tracking', {'enabled': False})
        target_scores = result['stats'].get('target_scores', 0)
        target_assists = result['stats'].get('target_assists', 0)
        target_highlights = result['stats'].get('target_highlights', target_scores + target_assists)
        debug_result = {
            'debugArtifactsKept': DEBUG_KEEP_ARTIFACTS,
        }
        if DEBUG_KEEP_ARTIFACTS:
            debug_result['allShots'] = result.get('shots', [])
            debug_result['annotatedVideo'] = (
                os.path.basename(result['annotated_video'])
                if result.get('annotated_video')
                else None
            )

        if selected_made_shots:
            for i, t in enumerate(selected_made_shots, start=1):
                try:
                    role = t.get('highlight_role')
                    role_label = '你的进球' if role == 'score' else '你的助攻' if role == 'assist' else '个人高光'
                    msg = (
                        f"检测到高光 #{i} - 帧: {t['frame']}, 时间: {float(t['timestamp']):.2f}s, {role_label}"
                    )
                    update_task_progress(task_id, log=msg)
                except Exception:
                    pass
        elif target_player_box and result.get('made_shots'):
            update_task_progress(task_id, log='检测到进球，但未归因到你的进球或助攻')
        
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
        processor = VideoProcessor()
        
        if selected_made_shots:
            # 有进球，生成集锦
            logger.info(f"生成集锦视频，片段数量: {len(selected_made_shots)}")
            # 集锦始终从原视频剪辑，保留原始音频；标注视频仅用于调试。
            source_video_for_clips = input_path
            video_result = processor.process_video_full_pipeline(
                video_path=source_video_for_clips,
                timestamps=selected_made_shots,
                output_path=output_path,
                before=before_seconds,
                after=after_seconds,
                log_callback=lambda m: update_task_progress(task_id, log=m)
            )
            
            if not video_result['success']:
                raise Exception(video_result['error'])
            
            # 验证输出文件
            if not os.path.exists(output_path):
                raise Exception("集锦视频生成失败，输出文件不存在")
            
            file_size = os.path.getsize(output_path)
            logger.info(f"集锦视频生成成功: {output_path}, 大小: {file_size / (1024*1024):.1f}MB")
            
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
                    'highlightVideo': output_filename,
                    'annotatedVideo': os.path.basename(result['annotated_video']) if result.get('annotated_video') else None,
                    'timestamps': selected_made_shots,
                    'allMadeTimestamps': result['made_shots'],
                    'fileSize': file_size,
                    'targetPlayerBox': target_player_box,
                    'tracking': tracking_summary,
                    **debug_result,
                }
            )
        else:
            # 没有检测到进球
            logger.info("未检测到可用于集锦的个人高光")
            message = '未检测到进球，请检查视频内容或调整参数'
            if target_player_box:
                if result['made_shots']:
                    message = '检测到全场进球，但当前规则未将其归因到你的进球或助攻，请检查选区或人物出场时机'
                else:
                    message = '人物跟踪正常，但当前未识别出全场进球，请检查机位、清晰度或进球识别规则'
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
                    'highlightVideo': None,
                    'annotatedVideo': os.path.basename(result['annotated_video']) if result.get('annotated_video') else None,
                    'message': message,
                    'targetPlayerBox': target_player_box,
                    'tracking': tracking_summary,
                    'timestamps': selected_made_shots,
                    'allMadeTimestamps': result['made_shots'],
                    **debug_result,
                }
            )

        # 清理上传的文件
        if DEBUG_KEEP_ARTIFACTS:
            logger.info("DEBUG_KEEP_ARTIFACTS 已开启，保留上传文件用于排查")
        else:
            try:
                os.remove(input_path)
                logger.info(f"清理上传文件: {input_path}")
            except Exception as e:
                logger.warning(f"清理上传文件失败: {e}")
            
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
        if task_id not in processing_tasks:
            logger.warning(f"查询不存在的任务: {task_id}")
            return jsonify({
                'success': False,
                'error': '任务不存在'
            }), 404
        
        task = processing_tasks[task_id]
        
        response = {
            'progress': task['progress'],
            'stage': task['stage'],
            'status': task['status'],
            'completed': task['status'] in ['completed', 'failed']
        }
        
        if task['status'] == 'completed' and task['result']:
            response['result'] = task['result']
        elif task['status'] == 'failed' and task['error']:
            response['error'] = task['error']
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"获取任务进度失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': '获取进度失败'
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
            'timestamp': datetime.now().isoformat(),
            'message': '篮球集锦生成服务运行正常',
            'components': {
                'upload_folder': os.path.exists(UPLOAD_FOLDER),
                'output_folder': os.path.exists(OUTPUT_FOLDER),
                'model_file': os.path.exists(MODEL_PATH),
                'active_tasks': len(processing_tasks)
            }
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
        expired_tasks = [
            task_id for task_id, task in list(processing_tasks.items())
            if task.get('status') in {'completed', 'failed'}
            and current_time - task.get('updated_at', task.get('created_at', current_time)) > TASK_RETENTION_SECONDS
        ]
        
        for task_id in expired_tasks:
            if processing_tasks.pop(task_id, None) is not None:
                logger.info(f"清理过期任务: {task_id}")
        
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
