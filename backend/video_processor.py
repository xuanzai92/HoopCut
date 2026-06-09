# video_processor.py - 视频处理模块
import cv2
import subprocess
import os
import tempfile
from typing import List, Dict, Optional
import shutil

class VideoProcessor:
    """
    视频剪辑和拼接处理器
    """
    CONFIRMED_HIGHLIGHT_ROLES = {'score', 'assist'}
    DEFAULT_INVOLVEMENT_LEAD_SECONDS = 1.0
    ASSIST_INVOLVEMENT_LEAD_SECONDS = 2.0
    TARGET_RELATED_REVIEW_LEAD_SECONDS = 1.5
    
    def __init__(self, temp_dir=None):
        """
        初始化视频处理器
        
        Args:
            temp_dir: 临时文件目录，如果为None则使用系统临时目录
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 检查FFmpeg是否可用
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查FFmpeg是否已安装"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ FFmpeg 已就绪")
            else:
                raise Exception("FFmpeg 未正确安装")
        except FileNotFoundError:
            raise Exception("未找到 FFmpeg，请先安装 FFmpeg")
        except Exception as e:
            raise Exception(f"FFmpeg 检查失败: {str(e)}")
    
    @classmethod
    def _resolve_involvement_lead_seconds(cls, shot: Dict) -> float:
        role = str(shot.get('highlight_role') or '')
        reason = str(shot.get('candidate_reason') or '')
        source = str(shot.get('candidate_source') or '')

        if role == 'assist' or 'assist' in reason:
            return cls.ASSIST_INVOLVEMENT_LEAD_SECONDS

        if source == 'target_attempt_fallback' or reason.startswith('attempt_'):
            return cls.TARGET_RELATED_REVIEW_LEAD_SECONDS

        return cls.DEFAULT_INVOLVEMENT_LEAD_SECONDS

    @classmethod
    def _calculate_clip_bounds(cls, shot: Dict, duration: float, before: float, after: float) -> Dict:
        shot_time = float(shot['timestamp'])
        start_time = max(0, shot_time - before)

        involvement_start = shot.get('involvement_start_timestamp')
        if isinstance(involvement_start, (int, float)):
            involvement_lead_seconds = cls._resolve_involvement_lead_seconds(shot)
            start_time = min(start_time, max(0, float(involvement_start) - involvement_lead_seconds))

        end_time = min(duration, shot_time + after)
        return {
            'start': round(start_time, 3),
            'end': round(end_time, 3),
            'duration': round(max(end_time - start_time, 0), 3),
        }

    def extract_clips(self, video_path: str, timestamps: List[Dict], 
                     before: float = 8, after: float = 2, 
                     progress_callback=None, log_callback=None,
                     output_dir: Optional[str] = None,
                     filename_prefix: str = 'clip') -> List[Dict]:
        """
        提取每个进球的视频片段
        
        Args:
            video_path: 原始视频路径
            timestamps: 进球时间戳列表 [{'frame': x, 'timestamp': y, 'made': True}, ...]
            before: 进球前保留的秒数
            after: 进球后保留的秒数
            progress_callback: 进度回调函数
        
        Returns:
            剪辑片段元数据列表
        """
        exportable_shots = [
            ts for ts in timestamps
            if ts.get('made', False) or ts.get('clip_export', False)
        ]

        if not exportable_shots:
            print("⚠️  没有可导出的相关片段")
            return []
        
        if log_callback:
            log_callback(f"开始提取 {len(exportable_shots)} 个相关片段...")
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise ValueError("无法读取视频帧率")
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        
        clip_segments = []
        clip_output_dir = output_dir or self.temp_dir
        os.makedirs(clip_output_dir, exist_ok=True)
        
        for idx, shot in enumerate(exportable_shots):
            # 计算剪辑时间
            bounds = self._calculate_clip_bounds(shot, duration, before, after)
            start_time = bounds['start']
            end_time = bounds['end']
            clip_duration = bounds['duration']
            
            # 生成临时文件名
            clip_filename = f"{filename_prefix}_{idx + 1:03d}_{shot['frame']}.mp4"
            clip_path = os.path.join(clip_output_dir, clip_filename)
            
            if log_callback:
                log_callback(f"提取片段 {idx + 1}/{len(exportable_shots)}: {start_time:.2f}s - {end_time:.2f}s (时长: {clip_duration:.2f}s)")
            
            try:
                # 单片段导出优先保证起止点准确，避免关键帧截断丢掉传球或起手动作。
                input_path = os.path.abspath(video_path)
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss', str(start_time),
                    '-i', input_path,
                    '-t', str(clip_duration),
                    '-map', '0:v:0',
                    '-map', '0:a?',
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',
                    '-crf', '18',
                    '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-movflags', '+faststart',
                    clip_path
                ]
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=60,  # 超时设置
                    check=True
                )
                
                # 验证文件是否生成
                if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                    clip_segments.append({
                        'path': clip_path,
                        'filename': os.path.basename(clip_path),
                        'index': idx + 1,
                        'start': start_time,
                        'end': end_time,
                        'duration': clip_duration,
                        'shot_frame': shot['frame'],
                        'shot_timestamp': shot['timestamp'],
                        'highlight_role': shot.get('highlight_role', 'score'),
                        'candidate_reason': shot.get('candidate_reason'),
                        'candidate_source': shot.get('candidate_source'),
                        'highlight_confidence': shot.get('highlight_confidence'),
                    })
                    if log_callback:
                        log_callback(f"✓ 片段 {idx + 1} 提取成功")
                else:
                    if log_callback:
                        log_callback(f"✗ 片段 {idx + 1} 生成失败")
                
                # 进度回调
                if progress_callback:
                    progress_callback(idx + 1, len(exportable_shots))
                    
            except subprocess.TimeoutExpired:
                if log_callback:
                    log_callback(f"✗ 片段 {idx + 1} 处理超时")
            except subprocess.CalledProcessError as e:
                stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
                if log_callback:
                    log_callback(f"✗ 片段 {idx + 1} FFmpeg错误: {stderr_output[:2000]}")
            except Exception as e:
                if log_callback:
                    log_callback(f"✗ 片段 {idx + 1} 未知错误: {str(e)}")
        
        if log_callback:
            log_callback(f"✓ 成功提取 {len(clip_segments)}/{len(exportable_shots)} 个片段")
        return clip_segments
    
    def concatenate_clips(self, clips: List[str], output_path: str,
                         add_transitions: bool = False, log_callback=None) -> bool:
        """
        拼接所有视频片段
        
        Args:
            clips: 片段文件路径列表
            output_path: 输出文件路径
            add_transitions: 是否添加转场效果（淡入淡出）
        
        Returns:
            是否成功
        """
        if not clips:
            print("⚠️  没有可拼接的片段")
            return False
        
        if log_callback:
            log_callback(f"开始拼接 {len(clips)} 个片段...")
        
        # 创建文件列表
        list_handle = tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            suffix='.txt',
            prefix='concat_',
            dir=self.temp_dir,
            delete=False,
        )
        list_file = list_handle.name
        list_handle.close()
        
        try:
            with open(list_file, 'w', encoding='utf-8') as f:
                for clip in clips:
                    abs_path = os.path.abspath(clip)
                    f.write(f"file '{abs_path}'\n")
            
            
            
            # 调试：打印文件列表内容
            with open(list_file, 'r', encoding='utf-8') as f:
                print("  文件列表内容:")
                for line in f:
                    print(f"    {line.strip()}")
            
            # 拼接命令
            if add_transitions:
                # 使用xfade滤镜添加转场（更复杂，暂时不实现）
                print("  注意：转场效果暂未实现，使用直接拼接")
            
            # 如果只有一个片段，直接复制文件
            if len(clips) == 1:
                print("  只有一个片段，直接复制...")
                shutil.copy2(clips[0], output_path)
                
                if os.path.exists(output_path):
                    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"✓ 复制完成: {output_path}")
                    print(f"  文件大小: {file_size_mb:.2f} MB")
                    return True
                else:
                    print("✗ 复制失败")
                    return False
            
            # 多个片段时使用concat并直接拷贝，不重新编码
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                output_path
            ]
            
            
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300,  # 5分钟超时
            )
            
            # 显示完整的FFmpeg输出（用于调试）
            if result.returncode != 0:
                stderr_output = result.stderr.decode('utf-8', errors='ignore')
                print(f"\n完整FFmpeg错误输出:")
                print(stderr_output)
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
            
            # 验证输出文件
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                if log_callback:
                    log_callback("✓ 拼接完成")
                return True
            else:
                if log_callback:
                    log_callback("✗ 拼接失败：输出文件无效")
                return False
                
        except subprocess.TimeoutExpired:
            if log_callback:
                log_callback("✗ 拼接超时")
            return False
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
            if log_callback:
                log_callback("✗ FFmpeg拼接错误")
            return False
        except Exception as e:
            if log_callback:
                log_callback(f"✗ 拼接失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理文件列表
            if os.path.exists(list_file):
                os.remove(list_file)
    
    def cleanup_clips(self, clips: List[str]):
        """清理临时片段文件"""
        print("\n清理临时文件...")
        cleaned = 0
        for clip in clips:
            try:
                if os.path.exists(clip):
                    os.remove(clip)
                    cleaned += 1
            except Exception as e:
                print(f"  ⚠️  无法删除 {clip}: {str(e)}")
        
        print(f"✓ 清理了 {cleaned}/{len(clips)} 个临时文件")
    
    def process_video_full_pipeline(self, video_path: str, timestamps: List[Dict],
                                    output_path: str, before: float = 8, after: float = 2,
                                    log_callback=None, clip_output_dir: Optional[str] = None,
                                    clip_filename_prefix: str = 'clip',
                                    keep_clips: bool = False) -> Dict:
        """
        完整的处理流程：检测 -> 剪辑 -> 拼接
        
        Args:
            video_path: 输入视频路径
            timestamps: 进球时间戳列表
            output_path: 输出视频路径
            before: 进球前保留秒数
            after: 进球后保留秒数
        
        Returns:
            处理结果字典
        """
        print("=" * 60)
        print("开始完整视频处理流程")
        print("=" * 60)
        
        result = {
            'success': False,
            'clips_extracted': 0,
            'clips': [],
            'output_file': None,
            'error': None
        }
        
        try:
            # 步骤1: 提取片段
            clip_segments = self.extract_clips(
                video_path,
                timestamps,
                before,
                after,
                None,
                log_callback,
                output_dir=clip_output_dir,
                filename_prefix=clip_filename_prefix,
            )
            clips = [segment['path'] for segment in clip_segments]
            result['clips_extracted'] = len(clips)
            result['clips'] = [
                {key: value for key, value in segment.items() if key != 'path'}
                for segment in clip_segments
            ]
            
            if not clips:
                result['error'] = "没有成功提取任何片段"
                return result

            confirmed_clips = [
                segment['path']
                for segment in clip_segments
                if str(segment.get('highlight_role') or '') in self.CONFIRMED_HIGHLIGHT_ROLES
            ]

            if not confirmed_clips:
                if log_callback:
                    log_callback("当前没有已确认的进球或助攻片段，跳过拼接视频生成")
                result['success'] = True
                result['output_file'] = None
                if not keep_clips:
                    self.cleanup_clips(clips)
                return result
            
            # 步骤2: 拼接片段
            success = self.concatenate_clips(confirmed_clips, output_path, False, log_callback)
            
            if success:
                result['success'] = True
                result['output_file'] = output_path
            else:
                result['error'] = "拼接失败"
            
            # 步骤3: 清理临时文件
            if not keep_clips:
                self.cleanup_clips(clips)
            
        except Exception as e:
            result['error'] = str(e)
            print(f"\n✗ 处理失败: {str(e)}")
        
        print("=" * 60)
        return result


# 测试代码
if __name__ == "__main__":
    # 模拟测试数据
    test_timestamps = [
        {'frame': 120, 'timestamp': 4.0, 'made': True},
        {'frame': 360, 'timestamp': 12.0, 'made': True},
        {'frame': 600, 'timestamp': 20.0, 'made': True},
    ]
    
    processor = VideoProcessor()
    
    # 测试完整流程
    result = processor.process_video_full_pipeline(
        video_path='test_video.mp4',
        timestamps=test_timestamps,
        output_path='highlight_output.mp4',
        before=8,
        after=2
    )
    
    print("\n最终结果:")
    print(f"  成功: {result['success']}")
    print(f"  提取片段数: {result['clips_extracted']}")
    print(f"  输出文件: {result['output_file']}")
    if result['error']:
        print(f"  错误: {result['error']}")
