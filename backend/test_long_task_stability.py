import os
import tempfile
import unittest
import zipfile
import io
import json
from unittest.mock import Mock, patch

import cv2

import app as app_module
from shot_detector_video import BasketballShotDetector
from video_processor import VideoProcessor


class FakeVideoCapture:
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 300
        if prop == cv2.CAP_PROP_FPS:
            return 30
        return 0

    def release(self):
        pass


class LongTaskStabilityTests(unittest.TestCase):
    def test_resolve_detection_progress_keeps_monotonic_value_for_retry_attempts(self):
        initial_progress = app_module._resolve_detection_progress(
            current_frame=120,
            total_frames=120,
            attempt_index=1,
            previous_progress=10,
        )
        retry_progress = app_module._resolve_detection_progress(
            current_frame=30,
            total_frames=120,
            attempt_index=2,
            previous_progress=initial_progress,
        )
        late_retry_progress = app_module._resolve_detection_progress(
            current_frame=120,
            total_frames=120,
            attempt_index=2,
            previous_progress=retry_progress,
        )

        self.assertEqual(initial_progress, 60)
        self.assertGreaterEqual(retry_progress, initial_progress)
        self.assertEqual(late_retry_progress, 65)

    def test_build_detection_progress_callback_marks_retry_stage_without_rollback(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                task_id = 'retry-progress-task'
                app_module.processing_tasks.clear()
                app_module.processing_tasks[task_id] = {
                    'status': 'detecting',
                    'progress': 10,
                    'stage': '准备处理',
                    'result': None,
                    'error': None,
                    'created_at': 0,
                    'updated_at': 0,
                    'metadata_path': os.path.join(metadata_folder, f'{task_id}.json'),
                }

                callback = app_module.build_detection_progress_callback(task_id)
                progress_updates = []
                stage_updates = []

                for current_frame in (60, 90, 30, 60):
                    callback(current_frame, 120)
                    task = app_module.processing_tasks[task_id]
                    progress_updates.append(task['progress'])
                    stage_updates.append(task['stage'])

                self.assertEqual(progress_updates, sorted(progress_updates))
                self.assertEqual(progress_updates[0], 35)
                self.assertEqual(progress_updates[1], 47)
                self.assertEqual(progress_updates[2], 61)
                self.assertEqual(progress_updates[3], 62)
                self.assertEqual(stage_updates[0], '正在分析视频... (60/120)')
                self.assertEqual(stage_updates[1], '正在分析视频... (90/120)')
                self.assertEqual(stage_updates[2], '正在自动补跑分析... (第 2 轮 30/120)')
                self.assertEqual(stage_updates[3], '正在自动补跑分析... (第 2 轮 60/120)')
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_video_processor_concatenates_only_confirmed_highlights(self):
        processor = object.__new__(VideoProcessor)
        processor.temp_dir = tempfile.gettempdir()
        processor.extract_clips = Mock(return_value=[
            {
                'path': '/tmp/score.mp4',
                'filename': 'score.mp4',
                'index': 1,
                'start': 1.0,
                'end': 9.0,
                'duration': 8.0,
                'shot_frame': 120,
                'shot_timestamp': 4.0,
                'highlight_role': 'score',
                'candidate_reason': None,
                'candidate_source': None,
                'highlight_confidence': 0.91,
            },
            {
                'path': '/tmp/review.mp4',
                'filename': 'review.mp4',
                'index': 2,
                'start': 5.0,
                'end': 13.0,
                'duration': 8.0,
                'shot_frame': 240,
                'shot_timestamp': 8.0,
                'highlight_role': 'possible',
                'candidate_reason': 'attempt_local_score_window',
                'candidate_source': 'attempt_review',
                'highlight_confidence': 0.64,
            },
            {
                'path': '/tmp/assist.mp4',
                'filename': 'assist.mp4',
                'index': 3,
                'start': 9.0,
                'end': 17.0,
                'duration': 8.0,
                'shot_frame': 360,
                'shot_timestamp': 12.0,
                'highlight_role': 'assist',
                'candidate_reason': None,
                'candidate_source': None,
                'highlight_confidence': 0.87,
            },
        ])
        processor.concatenate_clips = Mock(return_value=True)
        processor.cleanup_clips = Mock()

        result = processor.process_video_full_pipeline(
            video_path='video.mp4',
            timestamps=[],
            output_path='highlight.mp4',
            keep_clips=True,
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['clips_extracted'], 3)
        self.assertEqual(result['output_file'], 'highlight.mp4')
        processor.concatenate_clips.assert_called_once_with(
            ['/tmp/score.mp4', '/tmp/assist.mp4'],
            'highlight.mp4',
            False,
            None,
        )

    def test_video_processor_skips_concat_when_only_review_clips_exist(self):
        processor = object.__new__(VideoProcessor)
        processor.temp_dir = tempfile.gettempdir()
        processor.extract_clips = Mock(return_value=[
            {
                'path': '/tmp/review.mp4',
                'filename': 'review.mp4',
                'index': 1,
                'start': 5.0,
                'end': 13.0,
                'duration': 8.0,
                'shot_frame': 240,
                'shot_timestamp': 8.0,
                'highlight_role': 'possible',
                'candidate_reason': 'attempt_local_score_window',
                'candidate_source': 'attempt_review',
                'highlight_confidence': 0.64,
            },
        ])
        processor.concatenate_clips = Mock(return_value=True)
        processor.cleanup_clips = Mock()

        result = processor.process_video_full_pipeline(
            video_path='video.mp4',
            timestamps=[],
            output_path='highlight.mp4',
            keep_clips=True,
        )

        self.assertTrue(result['success'])
        self.assertEqual(result['clips_extracted'], 1)
        self.assertIsNone(result['output_file'])
        processor.concatenate_clips.assert_not_called()

    def test_process_video_uses_backend_default_clip_window_when_client_omits_params(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as metadata_folder, tempfile.TemporaryDirectory() as upload_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder

                stored_filename = 'default-file_game.mp4'
                input_path = os.path.join(upload_folder, stored_filename)
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'video-bytes')

                app_module.processing_tasks.clear()
                with patch.object(app_module.threading, 'Thread') as thread_cls:
                    response = app_module.app.test_client().post('/api/process', json={
                        'fileId': 'default-file',
                    })
                    payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload['success'])
                task_id = payload['taskId']
                self.assertIn(task_id, app_module.processing_tasks)
                self.assertEqual(
                    app_module.processing_tasks[task_id]['before_seconds'],
                    app_module.DEFAULT_CLIP_BEFORE_SECONDS,
                )
                self.assertEqual(
                    app_module.processing_tasks[task_id]['after_seconds'],
                    app_module.DEFAULT_CLIP_AFTER_SECONDS,
                )
                self.assertEqual(
                    thread_cls.call_args.kwargs['args'][2:],
                    (app_module.DEFAULT_CLIP_BEFORE_SECONDS, app_module.DEFAULT_CLIP_AFTER_SECONDS),
                )
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_progress_loads_persisted_completed_task(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                persisted_path = os.path.join(metadata_folder, 'persisted.json')
                with open(persisted_path, 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'persisted',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'result': {'message': 'ok', 'clips': []},
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/progress/persisted')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload['status'], 'completed')
                self.assertTrue(payload['completed'])
                self.assertEqual(payload['createdAt'], '1970-01-01T00:00:00Z')
                self.assertEqual(payload['updatedAt'], '1970-01-01T00:00:00Z')
                self.assertEqual(payload['result']['message'], 'ok')
                self.assertIn('persisted', app_module.processing_tasks)
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_upload_selection_candidates_returns_smart_frames(self):
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as upload_folder:
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder
                stored_filename = 'smart-file_game.mp4'
                input_path = os.path.join(upload_folder, stored_filename)
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'video-bytes')

                with patch.object(app_module, 'build_selection_frame_candidates', return_value=[
                    {
                        'imageUrl': 'data:image/jpeg;base64,abc',
                        'width': 1280,
                        'height': 720,
                        'time': 1.2,
                        'frame': 36,
                        'source': 'smart',
                        'recommended': True,
                        'recommendationScore': 0.88,
                        'suggestedBox': {
                            'x': 200,
                            'y': 120,
                            'width': 80,
                            'height': 190,
                            'frameWidth': 1280,
                            'frameHeight': 720,
                            'selectionTime': 1.2,
                            'selectionFrame': 36,
                        },
                    },
                ]) as builder:
                    response = app_module.app.test_client().get('/api/upload/candidates/smart-file')
                    payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload['success'])
                self.assertEqual(len(payload['candidateFrames']), 1)
                self.assertEqual(payload['candidateFrames'][0]['source'], 'smart')
                builder.assert_called_once_with('smart-file', input_path)
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder

    def test_get_upload_selection_candidates_returns_404_when_upload_missing(self):
        response = app_module.app.test_client().get('/api/upload/candidates/missing-file')
        payload = response.get_json()

        self.assertEqual(response.status_code, 404)
        self.assertFalse(payload['success'])
        self.assertEqual(payload['error'], '找不到上传的文件')

    def test_get_progress_normalizes_legacy_review_copy_for_completed_task(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                persisted_path = os.path.join(metadata_folder, 'legacy-copy.json')
                with open(persisted_path, 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'legacy-copy',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'result': {
                            'message': '已自动导出 2 个相关片段，其中 1 个为待确认候选，优先减少漏剪。',
                            'diagnostics': {
                                'summary': '已导出 1 个已确认片段，并额外保留 1 个待确认回合，优先减少漏剪。',
                                'reasons': ['部分目标球员相关回合仍未通过最终进球确认，暂作为待确认片段保留'],
                                'recommendedActions': ['优先检查待确认片段，确认是否还存在漏判进球或助攻'],
                            },
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/progress/legacy-copy')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    payload['result']['message'],
                    '已自动导出 2 个相关片段，其中 1 个为系统补充候选，优先减少漏剪。',
                )
                self.assertEqual(
                    payload['result']['diagnostics']['summary'],
                    '已导出 1 个已确认片段，并额外保留 1 个系统补充回合，优先减少漏剪。',
                )
                self.assertEqual(
                    payload['result']['diagnostics']['reasons'],
                    ['部分目标球员相关回合仍未通过最终进球确认，暂作为系统补充片段保留'],
                )
                self.assertEqual(
                    payload['result']['diagnostics']['recommendedActions'],
                    ['优先检查系统补充片段，确认是否还存在漏判进球或助攻'],
                )
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_progress_marks_persisted_running_task_as_interrupted(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                persisted_path = os.path.join(metadata_folder, 'running.json')
                with open(persisted_path, 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'running',
                        'status': 'detecting',
                        'progress': 42,
                        'stage': '正在分析视频...',
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/progress/running')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload['status'], 'failed')
                self.assertTrue(payload['completed'])
                self.assertEqual(payload['createdAt'], '1970-01-01T00:00:00Z')
                self.assertIsInstance(payload['updatedAt'], str)
                self.assertTrue(payload['updatedAt'].endswith('Z'))
                self.assertIn('服务已重启', payload['error'])

                with open(persisted_path, 'r', encoding='utf-8') as task_file:
                    persisted_payload = json.load(task_file)
                self.assertEqual(persisted_payload['status'], 'failed')
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_progress_splits_legacy_mixed_result_into_confirmed_and_debug_groups(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                persisted_path = os.path.join(metadata_folder, 'mixed-result.json')
                with open(persisted_path, 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'mixed-result',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'result': {
                            'targetScores': 1,
                            'targetAssists': 0,
                            'targetHighlights': 1,
                            'possibleHighlights': 1,
                            'relatedHighlights': 2,
                            'selectionSummary': {
                                'mode': 'mixed_with_review_candidates',
                                'confirmed': 1,
                                'possible': 1,
                            },
                            'timestamps': [
                                {
                                    'frame': 348,
                                    'timestamp': 11.59,
                                    'made': True,
                                    'highlight_role': 'score',
                                },
                                {
                                    'frame': 240,
                                    'timestamp': 8.0,
                                    'made': False,
                                    'highlight_role': 'possible',
                                    'candidate_reason': 'attempt_local_score_window',
                                },
                            ],
                            'clips': [
                                {
                                    'filename': 'task_clip_001_348.mp4',
                                    'index': 1,
                                    'start': 5.59,
                                    'end': 13.59,
                                    'duration': 8.0,
                                    'shotFrame': 348,
                                    'shotTimestamp': 11.59,
                                    'highlightRole': 'score',
                                },
                                {
                                    'filename': 'task_clip_002_240.mp4',
                                    'index': 2,
                                    'start': 2.0,
                                    'end': 10.0,
                                    'duration': 8.0,
                                    'shotFrame': 240,
                                    'shotTimestamp': 8.0,
                                    'highlightRole': 'possible',
                                    'candidateReason': 'attempt_local_score_window',
                                },
                            ],
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/progress/mixed-result')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(payload['result']['clips']), 1)
                self.assertEqual(payload['result']['clips'][0]['highlightRole'], 'score')
                self.assertEqual(len(payload['result']['debugClips']), 1)
                self.assertEqual(payload['result']['debugClips'][0]['highlightRole'], 'possible')
                self.assertEqual(len(payload['result']['timestamps']), 1)
                self.assertEqual(len(payload['result']['debugTimestamps']), 1)
                self.assertEqual(payload['result']['relatedHighlights'], 1)
                self.assertEqual(payload['result']['selectionSummary']['confirmed'], 1)
                self.assertEqual(payload['result']['selectionSummary']['possible'], 1)
                self.assertEqual(payload['result']['pipeline']['export']['selectedClipCount'], 1)
                self.assertEqual(payload['result']['pipeline']['export']['possibleClips'], 1)
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_cleanup_keeps_running_tasks(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as metadata_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                finished_path = os.path.join(metadata_folder, 'finished.json')
                with open(finished_path, 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'finished',
                        'status': 'completed',
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                app_module.processing_tasks.update({
                    'running': {
                        'status': 'detecting',
                        'created_at': 0,
                        'updated_at': 0,
                    },
                    'finished': {
                        'status': 'completed',
                        'created_at': 0,
                        'updated_at': 0,
                    },
                })

                with (
                    patch.object(app_module, 'TASK_RETENTION_SECONDS', 3600),
                    patch.object(app_module.time, 'time', return_value=7200),
                ):
                    app_module.cleanup_old_tasks()

                self.assertIn('running', app_module.processing_tasks)
                self.assertNotIn('finished', app_module.processing_tasks)
                self.assertFalse(os.path.exists(finished_path))
        finally:
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_task_source_returns_reusable_metadata(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as metadata_folder, tempfile.TemporaryDirectory() as upload_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder

                file_id = 'reusable-file'
                stored_filename = f'{file_id}_game.mp4'
                input_path = os.path.join(upload_folder, stored_filename)
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'original-video-bytes')

                with open(os.path.join(metadata_folder, 'reuse-task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'reuse-task',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'file_id': file_id,
                        'input_path': input_path,
                        'target_player_box': {
                            'x': 120,
                            'y': 80,
                            'width': 90,
                            'height': 180,
                            'frameWidth': 1280,
                            'frameHeight': 720,
                            'selectionTime': 14.5,
                            'selectionFrame': 435,
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/tasks/reuse-task/source')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload['success'])
                self.assertEqual(payload['taskId'], 'reuse-task')
                self.assertEqual(payload['fileId'], file_id)
                self.assertEqual(payload['filename'], 'game.mp4')
                self.assertEqual(payload['fileSize'], len(b'original-video-bytes'))
                self.assertEqual(payload['mimeType'], 'video/mp4')
                self.assertEqual(payload['sourceStreamUrl'], '/api/tasks/reuse-task/source/stream')
                self.assertEqual(payload['targetPlayerBox']['selectionFrame'], 435)
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_get_task_source_prefers_original_task_target_box_when_available(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as metadata_folder, tempfile.TemporaryDirectory() as upload_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder

                file_id = 'reusable-file'
                stored_filename = f'{file_id}_game.mp4'
                input_path = os.path.join(upload_folder, stored_filename)
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'original-video-bytes')

                with open(os.path.join(metadata_folder, 'reuse-task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'reuse-task',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'file_id': file_id,
                        'input_path': input_path,
                        'target_player_box': {
                            'x': 120,
                            'y': 80,
                            'width': 90,
                            'height': 180,
                            'frameWidth': 1280,
                            'frameHeight': 720,
                            'selectionTime': 14.5,
                            'selectionFrame': 435,
                        },
                        'result': {
                            'targetPlayerBox': {
                                'x': 96,
                                'y': 64,
                                'width': 88,
                                'height': 168,
                                'frameWidth': 1280,
                                'frameHeight': 720,
                                'selectionTime': 8.2,
                                'selectionFrame': 246,
                            },
                            'effectiveTargetPlayerBox': {
                                'x': 96,
                                'y': 64,
                                'width': 88,
                                'height': 168,
                                'frameWidth': 1280,
                                'frameHeight': 720,
                                'selectionTime': 8.2,
                                'selectionFrame': 246,
                            },
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                response = app_module.app.test_client().get('/api/tasks/reuse-task/source')
                payload = response.get_json()

                self.assertEqual(response.status_code, 200)
                self.assertTrue(payload['success'])
                self.assertEqual(payload['targetPlayerBox']['selectionTime'], 14.5)
                self.assertEqual(payload['targetPlayerBox']['selectionFrame'], 435)
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_stream_task_source_returns_original_video(self):
        original_tasks = app_module.processing_tasks.copy()
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as upload_folder:
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder
                input_path = os.path.join(upload_folder, 'source-file_game.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'video-stream-data')

                app_module.processing_tasks.clear()
                app_module.processing_tasks['source-task'] = {
                    'status': 'completed',
                    'progress': 100,
                    'stage': '处理完成',
                    'created_at': 0,
                    'updated_at': 0,
                    'file_id': 'source-file',
                    'input_path': input_path,
                }

                response = app_module.app.test_client().get('/api/tasks/source-task/source/stream')

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.data, b'video-stream-data')
                self.assertEqual(response.mimetype, 'video/mp4')
                response.close()
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_cleanup_deletes_unreferenced_upload_file(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as metadata_folder, tempfile.TemporaryDirectory() as upload_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder

                input_path = os.path.join(upload_folder, 'expired-file_game.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'expired-video')

                with open(os.path.join(metadata_folder, 'expired-task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'expired-task',
                        'status': 'completed',
                        'file_id': 'expired-file',
                        'input_path': input_path,
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                app_module.processing_tasks.clear()
                with (
                    patch.object(app_module, 'TASK_RETENTION_SECONDS', 3600),
                    patch.object(app_module.time, 'time', return_value=7200),
                ):
                    app_module.cleanup_old_tasks()

                self.assertFalse(os.path.exists(os.path.join(metadata_folder, 'expired-task.json')))
                self.assertFalse(os.path.exists(input_path))
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_cleanup_preserves_shared_upload_file_when_other_task_still_references_it(self):
        original_tasks = app_module.processing_tasks.copy()
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        original_upload_folder = app_module.app.config['UPLOAD_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as metadata_folder, tempfile.TemporaryDirectory() as upload_folder:
                app_module.TASK_METADATA_FOLDER = metadata_folder
                app_module.app.config['UPLOAD_FOLDER'] = upload_folder

                input_path = os.path.join(upload_folder, 'shared-file_game.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'shared-video')

                with open(os.path.join(metadata_folder, 'expired-task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'expired-task',
                        'status': 'completed',
                        'file_id': 'shared-file',
                        'input_path': input_path,
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                with open(os.path.join(metadata_folder, 'recent-task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'recent-task',
                        'status': 'completed',
                        'file_id': 'shared-file',
                        'input_path': input_path,
                        'created_at': 7000,
                        'updated_at': 7100,
                    }, task_file)

                app_module.processing_tasks.clear()
                with (
                    patch.object(app_module, 'TASK_RETENTION_SECONDS', 3600),
                    patch.object(app_module.time, 'time', return_value=7200),
                ):
                    app_module.cleanup_old_tasks()

                self.assertFalse(os.path.exists(os.path.join(metadata_folder, 'expired-task.json')))
                self.assertTrue(os.path.exists(os.path.join(metadata_folder, 'recent-task.json')))
                self.assertTrue(os.path.exists(input_path))
        finally:
            app_module.app.config['UPLOAD_FOLDER'] = original_upload_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_detect_shots_returns_existing_annotated_video(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=([], {'enabled': False}))

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as annotated_file:
            annotated_file.write(b'annotated-video')
            annotated_path = annotated_file.name

        try:
            with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
                result = detector.detect_shots_with_clips(
                    'video.mp4',
                    annotate=True,
                    annotated_output_path=annotated_path,
                )

            self.assertEqual(result['annotated_video'], annotated_path)
        finally:
            os.remove(annotated_path)

    def test_download_selected_clips_returns_zip(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as metadata_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.TASK_METADATA_FOLDER = metadata_folder
                clip_path = os.path.join(output_folder, 'task_clip_001_100.mp4')
                with open(clip_path, 'wb') as clip_file:
                    clip_file.write(b'clip-data')

                app_module.processing_tasks.clear()
                with open(os.path.join(metadata_folder, 'task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'task',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'result': {
                            'clips': [{
                                'filename': 'task_clip_001_100.mp4',
                                'index': 1,
                                'start': 5.5,
                                'end': 13.5,
                                'duration': 8.0,
                                'shotFrame': 100,
                                'shotTimestamp': 9.5,
                                'highlightRole': 'score',
                                'candidateReason': None,
                                'candidateSource': None,
                                'highlightConfidence': 0.91,
                            }],
                            'selectionSummary': {
                                'mode': 'mixed',
                                'confirmed': 1,
                                'possible': 0,
                            },
                            'diagnostics': {
                                'outcome': 'confirmed_highlights',
                            },
                            'pipeline': {
                                'scan': {
                                    'totalShotEvents': 3,
                                    'madeShotEvents': 1,
                                },
                                'attribution': {
                                    'confirmedHighlights': 1,
                                    'possibleHighlights': 0,
                                },
                                'export': {
                                    'selectedClipCount': 1,
                                    'scoreClips': 1,
                                    'assistClips': 0,
                                    'possibleClips': 0,
                                },
                            },
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                response = app_module.app.test_client().post(
                    '/api/download/clips/task',
                    json={'filenames': ['task_clip_001_100.mp4']},
                )

                self.assertEqual(response.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(archive.read('score/score_001.mp4'), b'clip-data')
                    manifest = json.loads(archive.read('manifest.json').decode('utf-8'))
                    self.assertEqual(manifest['taskId'], 'task')
                    self.assertEqual(manifest['archiveScope'], 'confirmed')
                    self.assertEqual(manifest['selectedClipCount'], 1)
                    self.assertEqual(manifest['clipGroups']['score'], 1)
                    self.assertEqual(manifest['selectionSummary']['confirmed'], 1)
                    self.assertEqual(manifest['pipeline']['scan']['totalShotEvents'], 3)
                    self.assertEqual(manifest['pipeline']['export']['selectedClipCount'], 1)
                    self.assertEqual(manifest['clips'][0]['archiveName'], 'score/score_001.mp4')
                    self.assertEqual(manifest['clips'][0]['archiveGroup'], 'score')
                    self.assertEqual(manifest['clips'][0]['highlightRole'], 'score')
                    self.assertIsNone(manifest['clips'][0]['candidateSource'])
                    self.assertEqual(manifest['clips'][0]['highlightConfidence'], 0.91)
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_download_selected_clips_returns_debug_zip_for_debug_scope(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_metadata_folder = app_module.TASK_METADATA_FOLDER
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as metadata_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.TASK_METADATA_FOLDER = metadata_folder
                clip_path = os.path.join(output_folder, 'task_clip_002_240.mp4')
                with open(clip_path, 'wb') as clip_file:
                    clip_file.write(b'debug-clip-data')

                app_module.processing_tasks.clear()
                with open(os.path.join(metadata_folder, 'task.json'), 'w', encoding='utf-8') as task_file:
                    json.dump({
                        'taskId': 'task',
                        'status': 'completed',
                        'progress': 100,
                        'stage': '处理完成',
                        'result': {
                            'clips': [],
                            'debugClips': [{
                                'filename': 'task_clip_002_240.mp4',
                                'index': 2,
                                'start': 2.0,
                                'end': 10.0,
                                'duration': 8.0,
                                'shotFrame': 240,
                                'shotTimestamp': 8.0,
                                'highlightRole': 'possible',
                                'candidateReason': 'attempt_local_score_window',
                                'candidateSource': 'attempt_review',
                                'highlightConfidence': 0.64,
                            }],
                            'selectionSummary': {
                                'mode': 'review_candidates_fallback',
                                'confirmed': 0,
                                'possible': 1,
                            },
                        },
                        'created_at': 0,
                        'updated_at': 0,
                    }, task_file)

                response = app_module.app.test_client().post(
                    '/api/download/clips/task',
                    json={'scope': 'debug'},
                )

                self.assertEqual(response.status_code, 200)
                with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                    self.assertEqual(archive.read('review/review_002.mp4'), b'debug-clip-data')
                    manifest = json.loads(archive.read('manifest.json').decode('utf-8'))
                    self.assertEqual(manifest['archiveScope'], 'debug')
                    self.assertEqual(manifest['selectedClipCount'], 1)
                    self.assertEqual(manifest['clipGroups']['review'], 1)
                    self.assertEqual(manifest['clips'][0]['archiveGroup'], 'review')
                    self.assertEqual(manifest['clips'][0]['highlightRole'], 'possible')
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.TASK_METADATA_FOLDER = original_metadata_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_process_video_background_persists_confirmed_target_score_result(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_temp_folder = app_module.app.config['TEMP_FOLDER']
        original_debug_keep_artifacts = app_module.DEBUG_KEEP_ARTIFACTS
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as temp_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.app.config['TEMP_FOLDER'] = temp_folder
                app_module.DEBUG_KEEP_ARTIFACTS = False

                input_path = os.path.join(temp_folder, 'input.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'input-video')

                task_id = 'confirmed-score-task'
                app_module.processing_tasks.clear()
                app_module.processing_tasks[task_id] = {
                    'status': 'starting',
                    'progress': 0,
                    'stage': '准备处理',
                    'result': None,
                    'error': None,
                    'created_at': 0,
                    'updated_at': 0,
                    'file_id': 'file-id',
                    'input_path': input_path,
                    'before_seconds': 6,
                    'after_seconds': 2,
                    'target_player_box': {
                        'x': 110,
                        'y': 70,
                        'width': 90,
                        'height': 170,
                        'frameWidth': 480,
                        'frameHeight': 256,
                        'selectionTime': 10.0,
                    },
                    'metadata_path': os.path.join(temp_folder, 'task.json'),
                }

                fake_detection_result = {
                    'made_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.887,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.887,
                    }],
                    'selected_made_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.887,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.887,
                    }],
                    'selected_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.887,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.887,
                    }],
                    'review_candidates': [],
                    'clips': [{
                        'start': 5.59,
                        'end': 13.59,
                        'shot_frame': 348,
                        'shot_timestamp': 11.59,
                        'highlight_role': 'score',
                        'candidate_reason': None,
                        'candidate_source': None,
                        'highlight_confidence': 0.887,
                    }],
                    'stats': {
                        'total_attempts': 2,
                        'total_makes': 1,
                        'accuracy': 50.0,
                        'target_scores': 1,
                        'target_assists': 0,
                        'target_highlights': 1,
                        'possible_highlights': 0,
                        'related_highlights': 1,
                        'review_candidate_highlights': 0,
                    },
                    'tracking': {
                        'enabled': True,
                        'coverage': 1.0,
                        'reacquiredCount': 0,
                        'guardedSwitches': 0,
                        'latestStatus': 'tracking',
                        'referenceFrames': [300, 311, 321],
                    },
                    'selection_summary': {
                        'mode': 'mixed',
                        'confirmed': 1,
                        'possible': 0,
                    },
                    'diagnostics': {
                        'outcome': 'confirmed_highlights',
                        'summary': '已导出与目标球员相关的进球和助攻片段。',
                        'reasons': [],
                        'recommendedActions': [],
                        'counts': {
                            'attempts': 2,
                            'madeShots': 1,
                            'selectedClips': 1,
                            'reviewCandidates': 0,
                            'possibleHighlights': 0,
                        },
                        'trackingCoverage': 1.0,
                    },
                    'pipeline': {
                        'scan': {
                            'mode': 'full_video_single_pass',
                            'fullVideoScanned': True,
                            'trackerEnabled': True,
                            'trackingStartTime': 10.0,
                            'trackingStartFrame': 300,
                            'totalShotEvents': 2,
                            'madeShotEvents': 1,
                            'targetVisibleEvents': 1,
                        },
                        'attribution': {
                            'selectionMode': 'mixed',
                            'confirmedHighlights': 1,
                            'possibleHighlights': 0,
                            'confirmedScores': 1,
                            'confirmedAssists': 0,
                            'reviewCandidates': 0,
                            'trackingCoverage': 1.0,
                        },
                        'export': {
                            'selectedClipCount': 1,
                            'selectedHighlights': 1,
                            'clipWindowBeforeSeconds': 6.0,
                            'clipWindowAfterSeconds': 2.0,
                            'scoreClips': 1,
                            'assistClips': 0,
                            'possibleClips': 0,
                        },
                    },
                    'target_player_box': {
                        'x': 96,
                        'y': 64,
                        'width': 88,
                        'height': 168,
                        'frameWidth': 480,
                        'frameHeight': 256,
                        'selectionTime': 8.2,
                        'selectionFrame': 246,
                    },
                    'annotated_video': os.path.join(output_folder, f'{task_id}_annotated.mp4'),
                }

                highlight_path = os.path.join(output_folder, f'{task_id}_highlight.mp4')
                clip_path = os.path.join(output_folder, f'{task_id}_clip_001_348.mp4')
                annotated_path = fake_detection_result['annotated_video']

                fake_processor = Mock()

                def fake_pipeline(**kwargs):
                    with open(highlight_path, 'wb') as highlight_file:
                        highlight_file.write(b'highlight-video')
                    with open(clip_path, 'wb') as clip_file:
                        clip_file.write(b'clip-video')
                    return {
                        'success': True,
                        'clips': [{
                            'filename': os.path.basename(clip_path),
                            'index': 1,
                            'start': 5.59,
                            'end': 13.59,
                            'duration': 8.0,
                            'shot_frame': 348,
                            'shot_timestamp': 11.59,
                            'highlight_role': 'score',
                            'candidate_reason': None,
                            'candidate_source': None,
                            'highlight_confidence': 0.887,
                        }],
                        'output_file': highlight_path,
                    }

                fake_processor.process_video_full_pipeline.side_effect = fake_pipeline

                with (
                    patch.object(app_module, 'MODEL_PATH', input_path),
                    patch.object(app_module, 'BasketballShotDetector') as detector_cls,
                    patch.object(app_module, 'VideoProcessor', return_value=fake_processor),
                ):
                    with open(annotated_path, 'wb') as annotated_file:
                        annotated_file.write(b'annotated-video')
                    detector_cls.return_value.detect_shots_with_clips.return_value = fake_detection_result
                    app_module.process_video_background(task_id, input_path, 6, 2)
                    detector_cls.return_value.detect_shots_with_clips.assert_called_once()
                    self.assertTrue(detector_cls.return_value.detect_shots_with_clips.call_args.kwargs['annotate'])

                task = app_module.processing_tasks[task_id]
                self.assertEqual(task['status'], 'completed')
                self.assertEqual(task['result']['targetScores'], 1)
                self.assertEqual(task['result']['possibleHighlights'], 0)
                self.assertEqual(task['result']['reviewCandidateHighlights'], 0)
                self.assertEqual(task['result']['diagnostics']['outcome'], 'confirmed_highlights')
                self.assertIsNone(task['result']['annotatedVideo'])
                self.assertEqual(task['result']['pipeline']['scan']['madeShotEvents'], 1)
                self.assertEqual(task['result']['pipeline']['export']['scoreClips'], 1)
                self.assertEqual(task['result']['targetPlayerBox']['selectionTime'], 10.0)
                self.assertIsNone(task['result']['targetPlayerBox'].get('selectionFrame'))
                self.assertEqual(task['result']['effectiveTargetPlayerBox']['selectionTime'], 8.2)
                self.assertEqual(task['result']['effectiveTargetPlayerBox']['selectionFrame'], 246)
                self.assertEqual(task['target_player_box']['selectionTime'], 10.0)
                self.assertIsNone(task['target_player_box'].get('selectionFrame'))
                self.assertEqual(task['effective_target_player_box']['selectionTime'], 8.2)
                self.assertEqual(task['effective_target_player_box']['selectionFrame'], 246)
                self.assertEqual(task['result']['clips'][0]['highlightRole'], 'score')
                self.assertEqual(task['result']['clips'][0]['highlightConfidence'], 0.887)
                self.assertEqual(task['result']['timestamps'][0]['highlight_role'], 'score')
                self.assertTrue(os.path.exists(highlight_path))
                self.assertTrue(os.path.exists(clip_path))
                self.assertTrue(os.path.exists(input_path))
                self.assertFalse(os.path.exists(annotated_path))
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.app.config['TEMP_FOLDER'] = original_temp_folder
            app_module.DEBUG_KEEP_ARTIFACTS = original_debug_keep_artifacts
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_process_video_background_keeps_annotated_video_for_tracking_review(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_temp_folder = app_module.app.config['TEMP_FOLDER']
        original_debug_keep_artifacts = app_module.DEBUG_KEEP_ARTIFACTS
        original_auto_keep_review_annotations = app_module.AUTO_KEEP_REVIEW_ANNOTATIONS
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as temp_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.app.config['TEMP_FOLDER'] = temp_folder
                app_module.DEBUG_KEEP_ARTIFACTS = False
                app_module.AUTO_KEEP_REVIEW_ANNOTATIONS = True

                input_path = os.path.join(temp_folder, 'input.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'input-video')

                task_id = 'tracking-review-task'
                app_module.processing_tasks.clear()
                app_module.processing_tasks[task_id] = {
                    'status': 'starting',
                    'progress': 0,
                    'stage': '准备处理',
                    'result': None,
                    'error': None,
                    'created_at': 0,
                    'updated_at': 0,
                    'file_id': 'file-id',
                    'input_path': input_path,
                    'before_seconds': 6,
                    'after_seconds': 2,
                    'target_player_box': {
                        'x': 110,
                        'y': 70,
                        'width': 90,
                        'height': 170,
                        'frameWidth': 480,
                        'frameHeight': 256,
                        'selectionTime': 10.0,
                    },
                    'metadata_path': os.path.join(temp_folder, 'task.json'),
                }

                annotated_path = os.path.join(output_folder, f'{task_id}_annotated.mp4')
                fake_detection_result = {
                    'made_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'unknown',
                        'owner_confidence': 0.1,
                        'target_visible': False,
                        'highlight_role': 'none',
                        'highlight_confidence': 0.0,
                    }],
                    'selected_made_shots': [],
                    'selected_shots': [],
                    'review_candidates': [],
                    'clips': [],
                    'stats': {
                        'total_attempts': 2,
                        'total_makes': 1,
                        'accuracy': 50.0,
                        'target_scores': 0,
                        'target_assists': 0,
                        'target_highlights': 0,
                        'possible_highlights': 0,
                        'related_highlights': 0,
                        'review_candidate_highlights': 0,
                    },
                    'tracking': {
                        'enabled': True,
                        'coverage': 0.24,
                        'reacquiredCount': 2,
                        'guardedSwitches': 1,
                        'latestStatus': 'lost',
                        'referenceFrames': [300, 311, 321],
                    },
                    'selection_summary': {
                        'mode': 'no_target_highlights',
                        'confirmed': 0,
                        'possible': 0,
                    },
                    'diagnostics': {
                        'outcome': 'global_makes_without_target',
                        'summary': '检测到了全场进球，但没有稳定归因到目标球员。',
                        'reasons': ['目标球员与进球回合之间的关联证据不足'],
                        'recommendedActions': ['重新框选并重跑'],
                        'counts': {
                            'attempts': 2,
                            'madeShots': 1,
                            'selectedClips': 0,
                            'reviewCandidates': 0,
                            'possibleHighlights': 0,
                        },
                        'trackingCoverage': 0.24,
                    },
                    'pipeline': {
                        'scan': {
                            'mode': 'full_video_single_pass',
                            'fullVideoScanned': True,
                            'trackerEnabled': True,
                            'trackingStartTime': 10.0,
                            'trackingStartFrame': 300,
                            'totalShotEvents': 2,
                            'madeShotEvents': 1,
                            'targetVisibleEvents': 0,
                        },
                        'attribution': {
                            'selectionMode': 'no_target_highlights',
                            'confirmedHighlights': 0,
                            'possibleHighlights': 0,
                            'confirmedScores': 0,
                            'confirmedAssists': 0,
                            'reviewCandidates': 0,
                            'trackingCoverage': 0.24,
                        },
                        'export': {
                            'selectedClipCount': 0,
                            'selectedHighlights': 0,
                            'clipWindowBeforeSeconds': 6.0,
                            'clipWindowAfterSeconds': 2.0,
                            'scoreClips': 0,
                            'assistClips': 0,
                            'possibleClips': 0,
                        },
                    },
                    'annotated_video': annotated_path,
                }

                with (
                    patch.object(app_module, 'MODEL_PATH', input_path),
                    patch.object(app_module, 'BasketballShotDetector') as detector_cls,
                    patch.object(app_module, 'VideoProcessor', return_value=Mock()),
                ):
                    with open(annotated_path, 'wb') as annotated_file:
                        annotated_file.write(b'annotated-video')
                    detector_cls.return_value.detect_shots_with_clips.return_value = fake_detection_result
                    app_module.process_video_background(task_id, input_path, 6, 2)
                    detector_cls.return_value.detect_shots_with_clips.assert_called_once()
                    self.assertTrue(detector_cls.return_value.detect_shots_with_clips.call_args.kwargs['annotate'])

                task = app_module.processing_tasks[task_id]
                self.assertEqual(task['status'], 'completed')
                self.assertIsNone(task['result']['highlightVideo'])
                self.assertEqual(task['result']['annotatedVideo'], os.path.basename(annotated_path))
                self.assertEqual(task['result']['annotatedVideoReason'], 'tracking_low_coverage')
                self.assertEqual(task['result']['diagnostics']['outcome'], 'global_makes_without_target')
                self.assertTrue(os.path.exists(annotated_path))
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.app.config['TEMP_FOLDER'] = original_temp_folder
            app_module.DEBUG_KEEP_ARTIFACTS = original_debug_keep_artifacts
            app_module.AUTO_KEEP_REVIEW_ANNOTATIONS = original_auto_keep_review_annotations
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_process_video_background_keeps_review_clips_without_combined_highlight_video(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_temp_folder = app_module.app.config['TEMP_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as temp_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.app.config['TEMP_FOLDER'] = temp_folder

                input_path = os.path.join(temp_folder, 'input.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'input-video')

                task_id = 'review-only-task'
                app_module.processing_tasks.clear()
                app_module.processing_tasks[task_id] = {
                    'status': 'starting',
                    'progress': 0,
                    'stage': '准备处理',
                    'result': None,
                    'error': None,
                    'created_at': 0,
                    'updated_at': 0,
                    'file_id': 'file-id',
                    'input_path': input_path,
                    'before_seconds': 6,
                    'after_seconds': 2,
                    'target_player_box': {
                        'x': 96,
                        'y': 64,
                        'width': 88,
                        'height': 168,
                        'frameWidth': 480,
                        'frameHeight': 256,
                        'selectionTime': 8.2,
                        'selectionFrame': 246,
                    },
                    'metadata_path': os.path.join(temp_folder, 'task.json'),
                }

                fake_detection_result = {
                    'made_shots': [],
                    'selected_made_shots': [],
                    'selected_shots': [{
                        'frame': 240,
                        'timestamp': 8.0,
                        'made': False,
                        'owner': 'target',
                        'owner_confidence': 0.67,
                        'target_visible': True,
                        'highlight_role': 'possible',
                        'highlight_confidence': 0.64,
                        'candidate_reason': 'attempt_local_score_window',
                        'candidate_source': 'attempt_review',
                        'clip_export': True,
                    }],
                    'review_candidates': [],
                    'clips': [],
                    'stats': {
                        'total_attempts': 1,
                        'total_makes': 0,
                        'accuracy': 0.0,
                        'target_scores': 0,
                        'target_assists': 0,
                        'target_highlights': 0,
                        'possible_highlights': 1,
                        'related_highlights': 1,
                        'review_candidate_highlights': 1,
                    },
                    'tracking': {
                        'enabled': True,
                        'coverage': 1.0,
                        'reacquiredCount': 0,
                        'guardedSwitches': 0,
                        'latestStatus': 'tracking',
                        'referenceFrames': [246, 258],
                    },
                    'selection_summary': {
                        'mode': 'review_candidates_fallback',
                        'confirmed': 0,
                        'possible': 1,
                    },
                    'diagnostics': {
                        'outcome': 'review_candidates',
                        'summary': '当前没有确认到目标球员进球或助攻，已先导出系统补充片段供你快速检查。',
                        'reasons': ['当前只保留了系统补充片段'],
                        'recommendedActions': ['先快速检查系统补充片段'],
                        'counts': {
                            'attempts': 1,
                            'madeShots': 0,
                            'selectedClips': 1,
                            'reviewCandidates': 1,
                            'possibleHighlights': 1,
                        },
                        'trackingCoverage': 1.0,
                    },
                    'pipeline': {
                        'scan': {
                            'mode': 'full_video_single_pass',
                            'fullVideoScanned': True,
                            'trackerEnabled': True,
                            'trackingStartTime': 8.2,
                            'trackingStartFrame': 246,
                            'totalShotEvents': 1,
                            'madeShotEvents': 0,
                            'targetVisibleEvents': 1,
                        },
                        'attribution': {
                            'selectionMode': 'review_candidates_fallback',
                            'confirmedHighlights': 0,
                            'possibleHighlights': 1,
                            'confirmedScores': 0,
                            'confirmedAssists': 0,
                            'reviewCandidates': 1,
                            'trackingCoverage': 1.0,
                        },
                        'export': {
                            'selectedClipCount': 1,
                            'selectedHighlights': 1,
                            'clipWindowBeforeSeconds': 6.0,
                            'clipWindowAfterSeconds': 2.0,
                            'scoreClips': 0,
                            'assistClips': 0,
                            'possibleClips': 1,
                        },
                    },
                    'annotated_video': None,
                }

                fake_processor = Mock()
                fake_processor.process_video_full_pipeline.return_value = {
                    'success': True,
                    'clips_extracted': 1,
                    'clips': [{
                        'filename': 'review-only-task_clip_001_240.mp4',
                        'index': 1,
                        'start': 2.0,
                        'end': 10.0,
                        'duration': 8.0,
                        'shot_frame': 240,
                        'shot_timestamp': 8.0,
                        'highlight_role': 'possible',
                        'candidate_reason': 'attempt_local_score_window',
                        'candidate_source': 'attempt_review',
                        'highlight_confidence': 0.64,
                    }],
                    'output_file': None,
                    'error': None,
                }

                with (
                    patch.object(app_module, 'MODEL_PATH', input_path),
                    patch.object(app_module, 'BasketballShotDetector') as detector_cls,
                    patch.object(app_module, 'VideoProcessor', return_value=fake_processor),
                ):
                    detector_cls.return_value.detect_shots_with_clips.return_value = fake_detection_result
                    app_module.process_video_background(task_id, input_path, 6, 2)

                task = app_module.processing_tasks[task_id]
                self.assertEqual(task['status'], 'completed')
                self.assertIsNone(task['result']['highlightVideo'])
                self.assertEqual(task['result']['fileSize'], 0)
                self.assertEqual(task['result']['possibleHighlights'], 1)
                self.assertEqual(task['result']['clips'], [])
                self.assertEqual(task['result']['debugClips'][0]['highlightRole'], 'possible')
                self.assertEqual(task['result']['debugClips'][0]['candidateSource'], 'attempt_review')
                self.assertEqual(task['result']['timestamps'], [])
                self.assertEqual(task['result']['debugTimestamps'][0]['highlight_role'], 'possible')
                self.assertEqual(task['result']['pipeline']['export']['selectedClipCount'], 0)
                self.assertEqual(task['result']['pipeline']['export']['selectedHighlights'], 0)
                self.assertEqual(task['result']['pipeline']['export']['possibleClips'], 1)
                fake_processor.process_video_full_pipeline.assert_called_once()
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.app.config['TEMP_FOLDER'] = original_temp_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)

    def test_process_video_background_prioritizes_confirmed_clips_before_review_guidance(self):
        original_tasks = app_module.processing_tasks.copy()
        original_output_folder = app_module.app.config['OUTPUT_FOLDER']
        original_temp_folder = app_module.app.config['TEMP_FOLDER']
        try:
            with tempfile.TemporaryDirectory() as output_folder, tempfile.TemporaryDirectory() as temp_folder:
                app_module.app.config['OUTPUT_FOLDER'] = output_folder
                app_module.app.config['TEMP_FOLDER'] = temp_folder

                input_path = os.path.join(temp_folder, 'input.mp4')
                with open(input_path, 'wb') as input_file:
                    input_file.write(b'input-video')

                task_id = 'confirmed-plus-review-task'
                app_module.processing_tasks.clear()
                app_module.processing_tasks[task_id] = {
                    'status': 'starting',
                    'progress': 0,
                    'stage': '准备处理',
                    'result': None,
                    'error': None,
                    'created_at': 0,
                    'updated_at': 0,
                    'file_id': 'file-id',
                    'input_path': input_path,
                    'before_seconds': 6,
                    'after_seconds': 2,
                    'target_player_box': {
                        'x': 96,
                        'y': 64,
                        'width': 88,
                        'height': 168,
                        'frameWidth': 480,
                        'frameHeight': 256,
                        'selectionTime': 8.2,
                        'selectionFrame': 246,
                    },
                    'metadata_path': os.path.join(temp_folder, 'task.json'),
                }

                highlight_output_path = os.path.join(output_folder, f'{task_id}_highlight.mp4')
                with open(highlight_output_path, 'wb') as highlight_file:
                    highlight_file.write(b'highlight-video')

                fake_detection_result = {
                    'made_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.91,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.91,
                    }],
                    'selected_made_shots': [{
                        'frame': 348,
                        'timestamp': 11.59,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.91,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.91,
                    }],
                    'selected_shots': [
                        {
                            'frame': 348,
                            'timestamp': 11.59,
                            'made': True,
                            'owner': 'target',
                            'owner_confidence': 0.91,
                            'target_visible': True,
                            'highlight_role': 'score',
                            'highlight_confidence': 0.91,
                        },
                        {
                            'frame': 240,
                            'timestamp': 8.0,
                            'made': False,
                            'owner': 'target',
                            'owner_confidence': 0.67,
                            'target_visible': True,
                            'highlight_role': 'possible',
                            'highlight_confidence': 0.64,
                            'candidate_reason': 'attempt_local_score_window',
                            'candidate_source': 'attempt_review',
                            'clip_export': True,
                        },
                    ],
                    'review_candidates': [{
                        'frame': 240,
                        'timestamp': 8.0,
                        'made': False,
                        'owner': 'target',
                        'owner_confidence': 0.67,
                        'target_visible': True,
                        'highlight_role': 'possible',
                        'highlight_confidence': 0.64,
                        'candidate_reason': 'attempt_local_score_window',
                        'candidate_source': 'attempt_review',
                        'clip_export': True,
                    }],
                    'clips': [],
                    'stats': {
                        'total_attempts': 2,
                        'total_makes': 1,
                        'accuracy': 50.0,
                        'target_scores': 1,
                        'target_assists': 0,
                        'target_highlights': 1,
                        'possible_highlights': 1,
                        'related_highlights': 2,
                        'review_candidate_highlights': 1,
                    },
                    'tracking': {
                        'enabled': True,
                        'coverage': 1.0,
                        'reacquiredCount': 0,
                        'guardedSwitches': 0,
                        'latestStatus': 'tracking',
                        'referenceFrames': [246, 258],
                    },
                    'selection_summary': {
                        'mode': 'mixed_with_review_candidates',
                        'confirmed': 1,
                        'possible': 1,
                    },
                    'diagnostics': {
                        'outcome': 'confirmed_with_review_candidates',
                        'summary': '已导出 1 个已确认片段，并额外保留 1 个系统补充回合。建议先验收已确认片段，如怀疑漏剪，再检查系统补充片段。',
                        'reasons': ['部分目标球员相关回合仍未通过最终进球确认，暂作为系统补充片段保留'],
                        'recommendedActions': ['先验收已确认片段，如怀疑漏剪，再检查系统补充片段。'],
                        'counts': {
                            'attempts': 2,
                            'madeShots': 1,
                            'selectedClips': 2,
                            'reviewCandidates': 1,
                            'possibleHighlights': 1,
                        },
                        'trackingCoverage': 1.0,
                    },
                    'pipeline': {
                        'scan': {
                            'mode': 'full_video_single_pass',
                            'fullVideoScanned': True,
                            'trackerEnabled': True,
                            'trackingStartTime': 8.2,
                            'trackingStartFrame': 246,
                            'totalShotEvents': 2,
                            'madeShotEvents': 1,
                            'targetVisibleEvents': 2,
                        },
                        'attribution': {
                            'selectionMode': 'mixed_with_review_candidates',
                            'confirmedHighlights': 1,
                            'possibleHighlights': 1,
                            'confirmedScores': 1,
                            'confirmedAssists': 0,
                            'reviewCandidates': 1,
                            'trackingCoverage': 1.0,
                        },
                        'export': {
                            'selectedClipCount': 2,
                            'selectedHighlights': 2,
                            'clipWindowBeforeSeconds': 6.0,
                            'clipWindowAfterSeconds': 2.0,
                            'scoreClips': 1,
                            'assistClips': 0,
                            'possibleClips': 1,
                        },
                    },
                }

                processor = Mock()
                processor.process_video_full_pipeline.return_value = {
                    'success': True,
                    'output_file': highlight_output_path,
                    'clips': [
                        {
                            'filename': f'{task_id}_clip_001_348.mp4',
                            'index': 1,
                            'start': 5.59,
                            'end': 13.59,
                            'duration': 8.0,
                            'shot_frame': 348,
                            'shot_timestamp': 11.59,
                            'highlight_role': 'score',
                            'candidate_reason': None,
                            'candidate_source': None,
                            'highlight_confidence': 0.91,
                        },
                        {
                            'filename': f'{task_id}_clip_002_240.mp4',
                            'index': 2,
                            'start': 2.0,
                            'end': 10.0,
                            'duration': 8.0,
                            'shot_frame': 240,
                            'shot_timestamp': 8.0,
                            'highlight_role': 'possible',
                            'candidate_reason': 'attempt_local_score_window',
                            'candidate_source': 'attempt_review',
                            'highlight_confidence': 0.64,
                        },
                    ],
                }

                with (
                    patch.object(app_module, 'MODEL_PATH', input_path),
                    patch.object(app_module, 'BasketballShotDetector') as detector_cls,
                    patch.object(app_module, 'VideoProcessor', return_value=processor),
                ):
                    detector_cls.return_value.detect_shots_with_clips.return_value = fake_detection_result
                    app_module.process_video_background(task_id, input_path, 6, 2)

                task = app_module.processing_tasks[task_id]
                self.assertEqual(task['status'], 'completed')
                self.assertEqual(
                    task['result']['message'],
                    '已自动导出 1 个已确认片段。另有 1 个系统补充回合已移入高级排错区，只有怀疑漏剪时再看。',
                )
                self.assertEqual(
                    task['result']['diagnostics']['recommendedActions'],
                    ['先验收已确认片段，如怀疑漏剪，再检查系统补充片段。'],
                )
                self.assertEqual(len(task['result']['clips']), 1)
                self.assertEqual(task['result']['clips'][0]['highlightRole'], 'score')
                self.assertEqual(len(task['result']['debugClips']), 1)
                self.assertEqual(task['result']['debugClips'][0]['highlightRole'], 'possible')
                self.assertEqual(task['result']['relatedHighlights'], 1)
                self.assertEqual(task['result']['pipeline']['export']['selectedClipCount'], 1)
                self.assertEqual(task['result']['pipeline']['export']['selectedHighlights'], 1)
                self.assertEqual(task['result']['pipeline']['export']['possibleClips'], 1)
        finally:
            app_module.app.config['OUTPUT_FOLDER'] = original_output_folder
            app_module.app.config['TEMP_FOLDER'] = original_temp_folder
            app_module.processing_tasks.clear()
            app_module.processing_tasks.update(original_tasks)


if __name__ == '__main__':
    unittest.main()
