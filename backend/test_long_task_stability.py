import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2

import app as app_module
from shot_detector_video import BasketballShotDetector


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
    def test_cleanup_keeps_running_tasks(self):
        original_tasks = app_module.processing_tasks.copy()
        try:
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
        finally:
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


if __name__ == '__main__':
    unittest.main()
