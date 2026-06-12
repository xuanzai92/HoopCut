import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from player_tracker import TargetPlayerTracker, classify_shot_involvement, review_shot_with_local_window
from shot_detector_video import BasketballShotDetector
from utils import find_recent_down_frame, find_recent_score_event, find_recent_up_frame, score
from video_processor import VideoProcessor


class FakeTracker:
    def get_box_at_frame(self, frame_index, max_gap=12):
        return (80, 80, 80, 180)


class FakeVideoCapture:
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 300
        if prop == cv2.CAP_PROP_FPS:
            return 30
        return 0

    def release(self):
        pass


class VideoLogicTests(unittest.TestCase):
    def _build_local_review_tracker(self):
        tracker = object.__new__(TargetPlayerTracker)
        tracker.initial_bbox = (60, 40, 60, 140)
        tracker.last_bbox = tracker.initial_bbox
        tracker.current_bbox = tracker.initial_bbox
        tracker.history = []
        tracker.reference_hist = None
        tracker.adaptive_hist = None
        tracker.reference_template = None
        tracker.reference_hists = []
        tracker.reference_templates = []
        tracker.reference_sample_frames = []
        tracker.tracker_type = 'MIL'
        tracker.active_frames = 0
        tracker.total_frames = 0
        tracker.missing_frames = 0
        tracker.reacquired_count = 0
        tracker.guarded_switches = 0
        tracker.lost_frames = 0
        tracker.latest_status = 'idle'

        reference_frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(reference_frame, (60, 40), (120, 180), (20, 80, 220), -1)
        cv2.rectangle(reference_frame, (78, 70), (102, 108), (255, 255, 255), -1)
        reference_hist, reference_template = tracker._extract_appearance(reference_frame, tracker.initial_bbox)
        tracker.reference_hist = reference_hist
        tracker.adaptive_hist = reference_hist.copy()
        tracker.reference_template = reference_template
        tracker.reference_hists = [reference_hist.copy()]
        tracker.reference_templates = [reference_template.copy()]
        tracker.reference_sample_frames = [0]
        tracker.get_box_at_frame = lambda frame_index, max_gap=12: None
        return tracker

    def _build_local_review_frame(self, ball_point=None, width=640, height=360):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (60, 40), (120, 180), (20, 80, 220), -1)
        cv2.rectangle(frame, (78, 70), (102, 108), (255, 255, 255), -1)
        if ball_point is not None:
            cv2.circle(frame, ball_point, 8, (0, 160, 255), -1)
        return frame

    def _build_scaled_target_frame(self, bbox, ball_point=None, width=240, height=240):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        x, y, bbox_width, bbox_height = bbox
        cv2.rectangle(frame, (x, y), (x + bbox_width, y + bbox_height), (20, 80, 220), -1)

        chest_left = x + int(round(bbox_width * 0.3))
        chest_top = y + int(round(bbox_height * 0.22))
        chest_right = x + int(round(bbox_width * 0.7))
        chest_bottom = y + int(round(bbox_height * 0.48))
        cv2.rectangle(frame, (chest_left, chest_top), (chest_right, chest_bottom), (255, 255, 255), -1)

        if ball_point is not None:
            cv2.circle(frame, ball_point, 8, (0, 160, 255), -1)
        return frame

    def test_score_requires_downward_crossing_through_rim(self):
        hoop_positions = [((100, 100), 10, 40, 20, 0.9)]
        made_trajectory = [
            ((94, 78), 8, 12, 12, 0.9),
            ((98, 92), 9, 12, 12, 0.9),
            ((101, 108), 10, 12, 12, 0.9),
            ((102, 118), 12, 12, 12, 0.9),
        ]
        outside_trajectory = [
            ((125, 78), 8, 12, 12, 0.9),
            ((126, 92), 9, 12, 12, 0.9),
            ((127, 108), 10, 12, 12, 0.9),
        ]

        self.assertTrue(score(made_trajectory, hoop_positions))
        self.assertFalse(score(outside_trajectory, hoop_positions))

    def test_recent_score_event_returns_scoring_frames(self):
        hoop_positions = [((100, 100), 10, 40, 20, 0.9)]
        made_trajectory = [
            ((76, 84), 5, 12, 12, 0.92),
            ((93, 79), 8, 12, 12, 0.94),
            ((99, 92), 10, 12, 12, 0.95),
            ((101, 108), 11, 12, 12, 0.95),
            ((102, 118), 14, 12, 12, 0.91),
        ]

        event = find_recent_score_event(made_trajectory, hoop_positions)

        self.assertIsNotNone(event)
        self.assertEqual(event['up_frame'], 8)
        self.assertEqual(event['cross_frame'], 10)
        self.assertEqual(event['down_frame'], 11)

    def test_recent_score_event_recovers_occluded_crossing_gap(self):
        hoop_positions = [((278, 119), 0, 40, 20, 0.9)]
        occluded_trajectory = [
            ((241, 69), 325, 12, 12, 0.55),
            ((251, 78), 328, 12, 12, 0.28),
            ((280, 120), 338, 12, 12, 0.40),
            ((280, 118), 339, 12, 12, 0.36),
            ((281, 128), 348, 12, 12, 0.67),
        ]

        event = find_recent_score_event(occluded_trajectory, hoop_positions)

        self.assertIsNotNone(event)
        self.assertEqual(event['up_frame'], 328)
        self.assertGreaterEqual(event['cross_frame'], 329)
        self.assertLessEqual(event['cross_frame'], 338)
        self.assertEqual(event['down_frame'], 348)

    def test_recent_up_and_down_frames_scan_window_not_only_latest_sample(self):
        hoop_positions = [((100, 100), 10, 40, 20, 0.9)]
        ball_positions = [
            ((90, 82), 18, 12, 12, 0.93),
            ((98, 111), 21, 12, 12, 0.94),
            ((160, 40), 24, 12, 12, 0.40),
        ]

        self.assertEqual(find_recent_up_frame(ball_positions, hoop_positions), 18)
        self.assertEqual(find_recent_down_frame(ball_positions, hoop_positions, after_frame=18), 21)

    def test_assist_attribution_returns_pass_evidence_window(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((500, 120), 35, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=35,
        )

        self.assertEqual(attribution['highlight_role'], 'assist')
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 16)

    def test_assist_attribution_accepts_coherent_receiver_trajectory(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((260, 148), 22, 12, 12, 0.9),
            ((320, 145), 26, 12, 12, 0.9),
            ((380, 140), 30, 12, 12, 0.9),
            ((430, 132), 34, 12, 12, 0.9),
            ((470, 124), 38, 12, 12, 0.9),
            ((500, 118), 40, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=40,
        )

        self.assertEqual(attribution['highlight_role'], 'assist')
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 16)

    def test_assist_attribution_prefers_latest_control_window(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((122, 151), 40, 12, 12, 0.9),
            ((132, 156), 43, 12, 12, 0.9),
            ((500, 120), 70, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=70,
        )

        self.assertEqual(attribution['highlight_role'], 'assist')
        self.assertEqual(attribution['involvement_start_frame'], 40)
        self.assertEqual(attribution['involvement_end_frame'], 43)

    def test_assist_attribution_requires_handoff_separation(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((230, 250), 28, 12, 12, 0.9),
            ((230, 250), 31, 12, 12, 0.9),
            ((230, 250), 40, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=40,
        )

        self.assertEqual(attribution['highlight_role'], 'none')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.42)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 16)

    def test_assist_attribution_requires_post_handoff_continuity(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((520, 120), 24, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=40,
        )

        self.assertEqual(attribution['highlight_role'], 'none')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.42)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 16)

    def test_assist_attribution_requires_terminal_release_window(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((420, 160), 24, 12, 12, 0.9),
            ((440, 150), 28, 12, 12, 0.9),
            ((470, 140), 30, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=40,
        )

        self.assertEqual(attribution['highlight_role'], 'none')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.42)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 16)

    def test_assist_attribution_requires_coherent_receiver_trajectory(self):
        ball_positions = [
            ((120, 150), 10, 12, 12, 0.9),
            ((130, 155), 13, 12, 12, 0.9),
            ((140, 160), 16, 12, 12, 0.9),
            ((220, 148), 22, 12, 12, 0.9),
            ((440, 260), 26, 12, 12, 0.9),
            ((260, 280), 30, 12, 12, 0.9),
            ((470, 160), 34, 12, 12, 0.9),
            ((480, 125), 38, 12, 12, 0.9),
            ((492, 120), 39, 12, 12, 0.9),
            ((500, 118), 40, 12, 12, 0.9),
        ]

        attribution = classify_shot_involvement(
            ball_positions,
            FakeTracker(),
            shot_release_frame=40,
        )

        self.assertEqual(attribution['highlight_role'], 'none')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.42)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 22)

    def test_assist_attribution_preserves_partial_global_evidence_below_confirm_threshold(self):
        with patch('player_tracker._collect_target_ball_samples') as collect_samples_mock, \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._handoff_confidence', return_value=0.34), \
             patch('player_tracker._post_handoff_continuity_confidence', return_value=0.27), \
             patch('player_tracker._terminal_release_window_confidence', return_value=0.30), \
             patch('player_tracker._receiver_trajectory_confidence', return_value=0.31), \
             patch('player_tracker._control_window_confidence', return_value=0.62), \
             patch('player_tracker._collect_ball_points_by_frame', return_value=[]):
            collect_samples_mock.side_effect = [
                [],
                [
                    {'frame': 10, 'inside': True, 'score': 0.70},
                    {'frame': 13, 'inside': True, 'score': 0.69},
                ],
                [],
            ]
            build_windows_mock.side_effect = [
                [],
                [{
                    'start_frame': 10,
                    'end_frame': 13,
                    'sample_count': 2,
                    'span': 3,
                    'best_score': 0.70,
                    'mean_score': 0.695,
                }],
            ]

            attribution = classify_shot_involvement(
                [((120, 150), 10, 12, 12, 0.9)],
                FakeTracker(),
                shot_release_frame=21,
            )

        self.assertEqual(attribution['highlight_role'], 'none')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.42)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 13)

    def test_assist_attribution_confirms_strong_aggregate_when_one_tail_signal_is_slightly_low(self):
        with patch('player_tracker._collect_target_ball_samples') as collect_samples_mock, \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._handoff_confidence', return_value=0.47), \
             patch('player_tracker._post_handoff_continuity_confidence', return_value=0.22), \
             patch('player_tracker._terminal_release_window_confidence', return_value=0.43), \
             patch('player_tracker._receiver_trajectory_confidence', return_value=0.36), \
             patch('player_tracker._control_window_confidence', return_value=0.74), \
             patch('player_tracker._collect_ball_points_by_frame', return_value=[]):
            collect_samples_mock.side_effect = [
                [],
                [
                    {'frame': 10, 'inside': True, 'score': 0.72},
                    {'frame': 13, 'inside': True, 'score': 0.71},
                ],
                [],
            ]
            build_windows_mock.side_effect = [
                [],
                [{
                    'start_frame': 10,
                    'end_frame': 13,
                    'sample_count': 2,
                    'span': 3,
                    'best_score': 0.72,
                    'mean_score': 0.715,
                }],
            ]

            attribution = classify_shot_involvement(
                [((120, 150), 10, 12, 12, 0.9)],
                FakeTracker(),
                shot_release_frame=21,
            )

        self.assertEqual(attribution['highlight_role'], 'assist')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.52)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 13)

    def test_assist_attribution_confirms_high_anchor_chain_when_one_tail_signal_dips_below_tolerance(self):
        with patch('player_tracker._collect_target_ball_samples') as collect_samples_mock, \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._handoff_confidence', return_value=0.49), \
             patch('player_tracker._post_handoff_continuity_confidence', return_value=0.21), \
             patch('player_tracker._terminal_release_window_confidence', return_value=0.46), \
             patch('player_tracker._receiver_trajectory_confidence', return_value=0.39), \
             patch('player_tracker._control_window_confidence', return_value=0.79), \
             patch('player_tracker._collect_ball_points_by_frame', return_value=[]):
            collect_samples_mock.side_effect = [
                [],
                [
                    {'frame': 10, 'inside': True, 'score': 0.74},
                    {'frame': 13, 'inside': True, 'score': 0.72},
                ],
                [],
            ]
            build_windows_mock.side_effect = [
                [],
                [{
                    'start_frame': 10,
                    'end_frame': 13,
                    'sample_count': 2,
                    'span': 3,
                    'best_score': 0.74,
                    'mean_score': 0.73,
                }],
            ]

            attribution = classify_shot_involvement(
                [((120, 150), 10, 12, 12, 0.9)],
                FakeTracker(),
                shot_release_frame=21,
            )

        self.assertEqual(attribution['highlight_role'], 'assist')
        self.assertGreaterEqual(attribution['highlight_confidence'], 0.54)
        self.assertEqual(attribution['involvement_start_frame'], 10)
        self.assertEqual(attribution['involvement_end_frame'], 13)

    def test_clip_starts_before_assist_evidence(self):
        processor = object.__new__(VideoProcessor)
        bounds = processor._calculate_clip_bounds(
            {
                'timestamp': 15.0,
                'highlight_role': 'assist',
                'involvement_start_timestamp': 9.0,
            },
            duration=30.0,
            before=3.0,
            after=1.0,
        )

        self.assertEqual(bounds['start'], 7.0)
        self.assertEqual(bounds['end'], 16.0)

    def test_clip_keeps_standard_involvement_preroll_for_score(self):
        processor = object.__new__(VideoProcessor)
        bounds = processor._calculate_clip_bounds(
            {
                'timestamp': 15.0,
                'highlight_role': 'score',
                'involvement_start_timestamp': 9.0,
            },
            duration=30.0,
            before=3.0,
            after=1.0,
        )

        self.assertEqual(bounds['start'], 8.0)
        self.assertEqual(bounds['end'], 16.0)

    @patch('video_processor.os.path.getsize', return_value=2048)
    @patch('video_processor.os.path.exists', return_value=True)
    @patch('video_processor.subprocess.run')
    @patch('video_processor.cv2.VideoCapture', return_value=FakeVideoCapture())
    def test_extract_clips_uses_precise_ffmpeg_reencode_settings(
        self,
        _video_capture,
        mock_run,
        _path_exists,
        _getsize,
    ):
        processor = object.__new__(VideoProcessor)
        processor.temp_dir = '/tmp'
        mock_run.return_value = Mock(returncode=0, stdout=b'', stderr=b'')

        clips = processor.extract_clips(
            'video.mp4',
            [{
                'frame': 120,
                'timestamp': 15.0,
                'made': True,
                'highlight_role': 'assist',
                'involvement_start_timestamp': 9.0,
            }],
            before=3.0,
            after=1.0,
            output_dir='/tmp',
        )

        cmd = mock_run.call_args.args[0]

        self.assertEqual(len(clips), 1)
        self.assertIn('libx264', cmd)
        self.assertIn('aac', cmd)
        self.assertIn('-map', cmd)
        self.assertNotIn('copy', cmd)

    def test_detect_shots_with_clips_exports_review_candidates_when_no_makes(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [{
                'frame': 120,
                'timestamp': 4.0,
                'made': False,
                'owner': 'target',
                'owner_confidence': 0.64,
                'target_visible': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.58,
                'local_highlight_role': 'score',
                'local_highlight_confidence': 0.71,
                'local_involvement_start_frame': 100,
                'local_involvement_end_frame': 118,
                'local_involvement_start_timestamp': 3.33,
                'local_involvement_end_timestamp': 3.93,
                'score_event_detected': False,
            }],
            {
                'enabled': True,
                'coverage': 0.82,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(result['selected_made_shots'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][0]['candidate_reason'], 'attempt_local_score_window')
        self.assertTrue(result['selected_shots'][0]['clip_export'])
        self.assertEqual(result['stats']['review_candidate_highlights'], 1)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['diagnostics']['outcome'], 'review_candidates')
        self.assertEqual(
            result['diagnostics']['recommendedActions'],
            [
                '先快速检查系统补充片段，确认是否存在漏检进球',
                '尽量把框选起点放在目标球员更清晰、身体完整的一帧',
            ],
        )
        self.assertEqual(result['selection_summary']['mode'], 'review_candidates_fallback')
        self.assertEqual(result['clips'][0]['candidate_reason'], 'attempt_local_score_window')
        self.assertEqual(result['pipeline']['scan']['totalShotEvents'], 1)
        self.assertEqual(result['pipeline']['scan']['madeShotEvents'], 0)
        self.assertEqual(result['pipeline']['attribution']['reviewCandidates'], 1)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 1)

    def test_detect_shots_with_clips_exports_global_assist_review_candidate_when_score_not_confirmed(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [{
                'frame': 210,
                'timestamp': 7.0,
                'made': False,
                'owner': 'unknown',
                'owner_confidence': 0.18,
                'target_visible': True,
                'attribution_highlight_role': 'assist',
                'attribution_highlight_confidence': 0.64,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'involvement_start_frame': 162,
                'involvement_end_frame': 194,
                'involvement_start_timestamp': 5.40,
                'involvement_end_timestamp': 6.47,
                'local_target_visible': False,
                'local_owner_confidence': 0.0,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.0,
                'local_involvement_start_frame': None,
                'local_involvement_end_frame': None,
                'local_involvement_start_timestamp': None,
                'local_involvement_end_timestamp': None,
                'score_event_detected': False,
            }],
            {
                'enabled': True,
                'coverage': 0.82,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(result['selected_made_shots'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][0]['candidate_reason'], 'global_assist_window')
        self.assertEqual(result['review_candidates'][0]['candidate_reason'], 'global_assist_window')
        self.assertGreaterEqual(result['review_candidates'][0]['highlight_confidence'], 0.64)
        self.assertEqual(result['stats']['review_candidate_highlights'], 1)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['selection_summary']['mode'], 'review_candidates_fallback')
        self.assertEqual(result['clips'][0]['candidate_reason'], 'global_assist_window')

    def test_detect_shots_with_clips_falls_back_to_target_attempts_when_review_candidates_are_empty(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [{
                'frame': 210,
                'timestamp': 7.0,
                'made': False,
                'owner': 'unknown',
                'owner_confidence': 0.12,
                'target_visible': False,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'local_target_visible': False,
                'local_owner_confidence': 0.31,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.0,
                'local_involvement_start_frame': None,
                'local_involvement_end_frame': None,
                'local_involvement_start_timestamp': None,
                'local_involvement_end_timestamp': None,
                'involvement_start_frame': None,
                'involvement_end_frame': None,
                'involvement_start_timestamp': None,
                'involvement_end_timestamp': None,
                'score_event_detected': False,
            }],
            {
                'enabled': True,
                'coverage': 0.82,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(result['selected_made_shots'], [])
        self.assertEqual(result['review_candidates'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][0]['candidate_source'], 'target_attempt_fallback')
        self.assertEqual(result['selection_summary']['mode'], 'target_attempt_fallback')
        self.assertEqual(result['stats']['review_candidate_highlights'], 0)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['diagnostics']['outcome'], 'target_attempt_fallback')
        self.assertEqual(result['pipeline']['attribution']['selectionMode'], 'target_attempt_fallback')
        self.assertEqual(result['pipeline']['attribution']['reviewCandidates'], 0)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 1)
        self.assertEqual(result['clips'][0]['candidate_reason'], 'attempt_target_context')

    def test_select_target_review_candidates_skips_low_signal_release_context(self):
        detector = object.__new__(BasketballShotDetector)

        review_candidates = detector._select_target_review_candidates(
            [
                {
                    'frame': 210,
                    'timestamp': 7.0,
                    'made': False,
                    'owner': 'target',
                    'owner_confidence': 0.67,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.61,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': 186,
                    'local_involvement_end_frame': 206,
                    'local_involvement_start_timestamp': 6.2,
                    'local_involvement_end_timestamp': 6.87,
                    'involvement_start_frame': 186,
                    'involvement_end_frame': 206,
                    'involvement_start_timestamp': 6.2,
                    'involvement_end_timestamp': 6.87,
                    'score_event_detected': False,
                },
            ],
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.82},
        )

        self.assertEqual(review_candidates, [])

    def test_select_target_review_candidates_skips_conflicting_partial_local_assist_context(self):
        detector = object.__new__(BasketballShotDetector)

        review_candidates = detector._select_target_review_candidates(
            [
                {
                    'frame': 228,
                    'timestamp': 7.59,
                    'made': False,
                    'owner': 'unknown',
                    'owner_confidence': 0.284,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.70,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.698,
                    'local_involvement_start_frame': 194,
                    'local_involvement_end_frame': 197,
                    'local_involvement_start_timestamp': 6.46,
                    'local_involvement_end_timestamp': 6.56,
                    'involvement_start_frame': 194,
                    'involvement_end_frame': 197,
                    'involvement_start_timestamp': 6.46,
                    'involvement_end_timestamp': 6.56,
                    'score_event_detected': False,
                },
            ],
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 1.0},
        )

        self.assertEqual(review_candidates, [])

    def test_detect_shots_with_clips_uses_attempt_fallback_when_only_low_signal_review_evidence_exists(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [{
                'frame': 210,
                'timestamp': 7.0,
                'made': False,
                'owner': 'target',
                'owner_confidence': 0.67,
                'target_visible': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.61,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.0,
                'local_involvement_start_frame': 186,
                'local_involvement_end_frame': 206,
                'local_involvement_start_timestamp': 6.2,
                'local_involvement_end_timestamp': 6.87,
                'involvement_start_frame': 186,
                'involvement_end_frame': 206,
                'involvement_start_timestamp': 6.2,
                'involvement_end_timestamp': 6.87,
                'score_event_detected': False,
            }],
            {
                'enabled': True,
                'coverage': 0.82,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(result['review_candidates'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][0]['candidate_source'], 'target_attempt_fallback')
        self.assertEqual(result['selected_shots'][0]['candidate_reason'], 'attempt_target_release')
        self.assertEqual(result['selection_summary']['mode'], 'target_attempt_fallback')
        self.assertEqual(result['diagnostics']['outcome'], 'target_attempt_fallback')

    def test_detect_shots_with_clips_keeps_user_selected_target_when_retry_candidates_exist(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(side_effect=[
            (
                [{
                    'frame': 210,
                    'timestamp': 7.0,
                    'made': True,
                    'owner': 'unknown',
                    'owner_confidence': 0.12,
                    'target_visible': False,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.31,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': None,
                    'local_involvement_end_frame': None,
                    'local_involvement_start_timestamp': None,
                    'local_involvement_end_timestamp': None,
                    'involvement_start_frame': None,
                    'involvement_end_frame': None,
                    'involvement_start_timestamp': None,
                    'involvement_end_timestamp': None,
                    'score_event_detected': True,
                }],
                {
                    'enabled': True,
                    'coverage': 0.28,
                },
            ),
            (
                [{
                    'frame': 240,
                    'timestamp': 8.0,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.87,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.87,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.87,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.87,
                    'local_involvement_start_frame': 225,
                    'local_involvement_end_frame': 238,
                    'local_involvement_start_timestamp': 7.5,
                    'local_involvement_end_timestamp': 7.93,
                    'involvement_start_frame': 225,
                    'involvement_end_frame': 238,
                    'involvement_start_timestamp': 7.5,
                    'involvement_end_timestamp': 7.93,
                    'score_event_detected': True,
                }],
                {
                    'enabled': True,
                    'coverage': 0.91,
                },
            ),
        ])

        retry_target_box = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.2,
            'selectionFrame': 186,
        }

        with (
            patch.object(
                detector,
                '_discover_retry_target_boxes',
                return_value=[{
                    'target_player_box': retry_target_box,
                    'score': 0.81,
                }],
            ) as discover_retry_mock,
            patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={
                    'x': 0,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'frameWidth': 320,
                    'frameHeight': 180,
                    'selectionTime': 5.0,
                    'selectionFrame': 150,
                },
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        discover_retry_mock.assert_not_called()
        self.assertFalse(result['auto_retry']['used'])
        self.assertEqual(result['auto_retry']['attempted'], 0)
        self.assertEqual(result['auto_retry']['initialSelectionTime'], 5.0)
        self.assertEqual(result['auto_retry']['finalSelectionTime'], 5.0)
        self.assertEqual(
            detector.detect_shots.call_args.kwargs['target_player_box']['selectionTime'],
            5.0,
        )

    def test_detect_shots_with_clips_skips_auto_retry_even_when_initial_tracking_is_weak(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(side_effect=[
            (
                [{
                    'frame': 210,
                    'timestamp': 7.0,
                    'made': True,
                    'owner': 'unknown',
                    'owner_confidence': 0.12,
                    'target_visible': False,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.31,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': None,
                    'local_involvement_end_frame': None,
                    'local_involvement_start_timestamp': None,
                    'local_involvement_end_timestamp': None,
                    'involvement_start_frame': None,
                    'involvement_end_frame': None,
                    'involvement_start_timestamp': None,
                    'involvement_end_timestamp': None,
                    'score_event_detected': True,
                }],
                {
                    'enabled': True,
                    'coverage': 0.28,
                },
            ),
        ])

        retry_target_box_1 = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.2,
            'selectionFrame': 186,
        }
        retry_target_box_2 = {
            **retry_target_box_1,
            'selectionTime': 7.1,
            'selectionFrame': 213,
        }

        with (
            patch.object(
                detector,
                '_discover_retry_target_boxes',
                return_value=[
                    {
                        'target_player_box': retry_target_box_1,
                        'score': 0.81,
                    },
                    {
                        'target_player_box': retry_target_box_2,
                        'score': 0.79,
                    },
                ],
            ) as discover_retry_mock,
            patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={
                    'x': 0,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'frameWidth': 320,
                    'frameHeight': 180,
                    'selectionTime': 5.0,
                    'selectionFrame': 150,
                },
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        discover_retry_mock.assert_not_called()
        self.assertFalse(result['auto_retry']['used'])
        self.assertEqual(result['auto_retry']['attempted'], 0)
        self.assertEqual(result['auto_retry']['finalSelectionTime'], 5.0)

    def test_retry_sample_frames_include_farther_backward_search(self):
        detector = object.__new__(BasketballShotDetector)

        sample_frames = detector._build_retry_sample_frames(
            center_frame=300,
            fps=30.0,
            total_frames=600,
        )

        self.assertIn(252, sample_frames)
        self.assertIn(228, sample_frames)
        self.assertIn(162, sample_frames)

    def test_prime_target_reference_window_uses_dense_sampling_when_coarse_samples_fail(self):
        detector = object.__new__(BasketballShotDetector)

        class FakePrimeCapture:
            def __init__(self):
                self.position = 140
                self.frames = {
                    frame_index: np.zeros((48, 48, 3), dtype=np.uint8)
                    for frame_index in range(60, 141)
                }

            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    return self.position
                return 0

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.position = int(value)

            def read(self):
                frame = self.frames.get(self.position)
                if frame is None:
                    return False, None
                return True, frame.copy()

        tracker = Mock()
        tracker.initial_bbox = (20, 10, 14, 28)
        tracker.last_bbox = tracker.initial_bbox
        tracker.current_bbox = tracker.initial_bbox
        dense_success_frames = {96, 104, 88, 112}

        def add_reference_sample(_frame, frame_index, anchor_bbox=None):
            if frame_index in dense_success_frames:
                return {'bbox': anchor_bbox or tracker.initial_bbox}
            return None

        tracker.add_reference_sample = Mock(side_effect=add_reference_sample)
        cap = FakePrimeCapture()

        successful_samples = detector._prime_target_reference_window(
            cap,
            tracker,
            center_frame=100,
            fps=30.0,
            total_frames=240,
        )

        attempted_frames = [call.args[1] for call in tracker.add_reference_sample.call_args_list]

        self.assertGreaterEqual(successful_samples, detector.REFERENCE_PRIME_MIN_SUCCESSFUL_SAMPLES)
        self.assertIn(96, attempted_frames)
        self.assertIn(104, attempted_frames)
        self.assertEqual(cap.position, 140)

    def test_prime_target_reference_window_stops_after_minimum_dense_samples(self):
        detector = object.__new__(BasketballShotDetector)

        class FakePrimeCapture:
            def __init__(self):
                self.position = 40
                self.frames = {
                    frame_index: np.zeros((48, 48, 3), dtype=np.uint8)
                    for frame_index in range(0, 121)
                }

            def get(self, prop):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    return self.position
                return 0

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self.position = int(value)

            def read(self):
                frame = self.frames.get(self.position)
                if frame is None:
                    return False, None
                return True, frame.copy()

        tracker = Mock()
        tracker.initial_bbox = (20, 10, 14, 28)
        tracker.last_bbox = tracker.initial_bbox
        tracker.current_bbox = tracker.initial_bbox
        dense_success_frames = {56, 64, 52, 68}

        def add_reference_sample(_frame, frame_index, anchor_bbox=None):
            if frame_index in dense_success_frames:
                return {'bbox': anchor_bbox or tracker.initial_bbox}
            return None
        
        tracker.add_reference_sample = Mock(side_effect=add_reference_sample)
        cap = FakePrimeCapture()

        successful_samples = detector._prime_target_reference_window(
            cap,
            tracker,
            center_frame=60,
            fps=30.0,
            total_frames=180,
        )

        attempted_frames = [call.args[1] for call in tracker.add_reference_sample.call_args_list]

        self.assertEqual(successful_samples, detector.REFERENCE_PRIME_MIN_SUCCESSFUL_SAMPLES)
        self.assertNotIn(48, attempted_frames)
        self.assertEqual(cap.position, 40)

    def test_select_prealigned_target_player_box_keeps_original_user_selection(self):
        detector = object.__new__(BasketballShotDetector)
        original_box = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.0,
            'selectionFrame': 180,
        }
        earlier_box = {
            **original_box,
            'selectionTime': 4.8,
            'selectionFrame': 144,
        }

        selected_box = detector._select_prealigned_target_player_box_from_candidates(
            original_box,
            [
                {'target_player_box': earlier_box, 'score': 0.79},
                {'target_player_box': {**original_box, 'selectionTime': 5.8, 'selectionFrame': 174}, 'score': 0.91},
            ],
        )

        self.assertEqual(selected_box, original_box)

    def test_select_prealigned_target_player_box_keeps_original_when_candidate_is_not_stable_enough(self):
        detector = object.__new__(BasketballShotDetector)
        original_box = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.0,
            'selectionFrame': 180,
        }

        selected_box = detector._select_prealigned_target_player_box_from_candidates(
            original_box,
            [
                {
                    'target_player_box': {**original_box, 'selectionTime': 5.6, 'selectionFrame': 168},
                    'score': 0.68,
                },
            ],
        )

        self.assertEqual(selected_box, original_box)

    def test_detect_shots_with_clips_uses_original_target_box_for_initial_run(self):
        detector = object.__new__(BasketballShotDetector)
        original_box = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.0,
            'selectionFrame': 180,
        }
        earlier_box = {
            **original_box,
            'selectionTime': 4.8,
            'selectionFrame': 144,
        }

        detector.detect_shots = Mock(return_value=([], {'enabled': True, 'coverage': 1.0}))
        detector._build_detection_output = Mock(return_value={
            'selection_summary': {'mode': 'mixed', 'confirmed': 1, 'possible': 0},
            'stats': {
                'target_highlights': 1,
                'related_highlights': 1,
                'possible_highlights': 0,
            },
            'tracking': {'enabled': True, 'coverage': 1.0},
            'diagnostics': {'outcome': 'confirmed_highlights'},
            'pipeline': {'attribution': {'trackingCoverage': 1.0}},
            'made_shots': [],
            'selected_shots': [],
            'review_candidates': [],
            'clips': [],
            'target_player_box': earlier_box,
        })

        with (
            patch.object(
                detector,
                '_discover_retry_target_boxes',
                return_value=[{'target_player_box': earlier_box, 'score': 0.79}],
            ),
            patch.object(detector, '_should_auto_retry_target_detection', return_value=False),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box=original_box,
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        self.assertEqual(detector.detect_shots.call_args.kwargs['target_player_box'], original_box)
        self.assertEqual(result['auto_retry']['initialSelectionTime'], 6.0)
        self.assertEqual(result['auto_retry']['finalSelectionTime'], 6.0)

    def test_retry_candidate_ranking_prefers_earlier_frame_when_scores_are_close(self):
        detector = object.__new__(BasketballShotDetector)
        earlier_candidate = {
            'score': 0.78,
            'target_player_box': {
                'selectionFrame': 180,
            },
        }
        later_candidate = {
            'score': 0.81,
            'target_player_box': {
                'selectionFrame': 315,
            },
        }

        earlier_rank = detector._rank_retry_candidate(
            candidate=earlier_candidate,
            center_frame=300,
            fps=30.0,
        )
        later_rank = detector._rank_retry_candidate(
            candidate=later_candidate,
            center_frame=300,
            fps=30.0,
        )

        self.assertGreater(earlier_rank, later_rank)

    def test_score_detection_output_prefers_more_related_highlights_over_mode_rank(self):
        detector = object.__new__(BasketballShotDetector)

        stable_output = {
            'selection_summary': {
                'mode': 'mixed',
                'confirmed': 1,
                'possible': 0,
            },
            'stats': {
                'target_highlights': 1,
                'related_highlights': 1,
                'possible_highlights': 0,
            },
            'tracking': {
                'coverage': 0.92,
            },
        }
        broader_output = {
            'selection_summary': {
                'mode': 'mixed_with_review_candidates',
                'confirmed': 1,
                'possible': 1,
            },
            'stats': {
                'target_highlights': 1,
                'related_highlights': 2,
                'possible_highlights': 1,
            },
            'tracking': {
                'coverage': 0.74,
            },
        }

        self.assertGreater(
            detector._score_detection_output(broader_output),
            detector._score_detection_output(stable_output),
        )

    def test_detect_shots_with_clips_skips_auto_retry_when_confirmed_result_is_stable(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [{
                'frame': 240,
                'timestamp': 8.0,
                'made': True,
                'owner': 'target',
                'owner_confidence': 0.87,
                'target_visible': True,
                'highlight_role': 'score',
                'highlight_confidence': 0.87,
                'local_target_visible': True,
                'local_owner_confidence': 0.87,
                'local_highlight_role': 'score',
                'local_highlight_confidence': 0.87,
                'local_involvement_start_frame': 225,
                'local_involvement_end_frame': 238,
                'local_involvement_start_timestamp': 7.5,
                'local_involvement_end_timestamp': 7.93,
                'involvement_start_frame': 225,
                'involvement_end_frame': 238,
                'involvement_start_timestamp': 7.5,
                'involvement_end_timestamp': 7.93,
                'score_event_detected': True,
            }],
            {
                'enabled': True,
                'coverage': 0.91,
            },
        ))

        with (
            patch.object(detector, '_discover_retry_target_boxes', return_value=[]) as discover_retry_mock,
            patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={
                    'x': 0,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'frameWidth': 320,
                    'frameHeight': 180,
                    'selectionTime': 5.0,
                    'selectionFrame': 150,
                },
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        discover_retry_mock.assert_not_called()
        self.assertFalse(result['auto_retry']['used'])
        self.assertEqual(result['auto_retry']['attempted'], 0)
        self.assertEqual(result['selection_summary']['confirmed'], 1)

    def test_detect_shots_with_clips_keeps_user_selected_target_when_possible_highlights_exist(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(side_effect=[
            (
                [
                    {
                        'frame': 120,
                        'timestamp': 4.0,
                        'made': True,
                        'owner': 'target',
                        'owner_confidence': 0.88,
                        'target_visible': True,
                        'highlight_role': 'score',
                        'highlight_confidence': 0.88,
                        'involvement_start_frame': 110,
                        'involvement_end_frame': 120,
                        'involvement_start_timestamp': 3.67,
                        'involvement_end_timestamp': 4.0,
                        'local_target_visible': True,
                        'local_owner_confidence': 0.82,
                        'local_highlight_role': 'score',
                        'local_highlight_confidence': 0.82,
                        'local_involvement_start_frame': 108,
                        'local_involvement_end_frame': 120,
                        'local_involvement_start_timestamp': 3.60,
                        'local_involvement_end_timestamp': 4.0,
                        'score_event_detected': True,
                    },
                    {
                        'frame': 240,
                        'timestamp': 8.0,
                        'made': False,
                        'owner': 'target',
                        'owner_confidence': 0.67,
                        'target_visible': True,
                        'highlight_role': 'none',
                        'highlight_confidence': 0.0,
                        'local_target_visible': True,
                        'local_owner_confidence': 0.61,
                        'local_highlight_role': 'score',
                        'local_highlight_confidence': 0.74,
                        'local_involvement_start_frame': 222,
                        'local_involvement_end_frame': 238,
                        'local_involvement_start_timestamp': 7.40,
                        'local_involvement_end_timestamp': 7.93,
                        'score_event_detected': False,
                    },
                ],
                {
                    'enabled': True,
                    'coverage': 0.91,
                },
            ),
        ])

        retry_target_box = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.2,
            'selectionFrame': 186,
        }

        with (
            patch.object(
                detector,
                '_discover_retry_target_boxes',
                return_value=[{
                    'target_player_box': retry_target_box,
                    'score': 0.81,
                }],
            ) as discover_retry_mock,
            patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={
                    'x': 0,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'frameWidth': 320,
                    'frameHeight': 180,
                    'selectionTime': 5.0,
                    'selectionFrame': 150,
                },
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        discover_retry_mock.assert_not_called()
        self.assertFalse(result['auto_retry']['used'])
        self.assertEqual(result['auto_retry']['attempted'], 0)
        self.assertEqual(result['auto_retry']['finalSelectionTime'], 5.0)
        self.assertGreaterEqual(result['stats']['possible_highlights'], 0)

    def test_detect_shots_with_clips_does_not_retry_with_alternate_target_candidates(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(side_effect=[
            (
                [{
                    'frame': 210,
                    'timestamp': 7.0,
                    'made': True,
                    'owner': 'unknown',
                    'owner_confidence': 0.12,
                    'target_visible': False,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.31,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': None,
                    'local_involvement_end_frame': None,
                    'local_involvement_start_timestamp': None,
                    'local_involvement_end_timestamp': None,
                    'involvement_start_frame': None,
                    'involvement_end_frame': None,
                    'involvement_start_timestamp': None,
                    'involvement_end_timestamp': None,
                    'score_event_detected': True,
                }],
                {
                    'enabled': True,
                    'coverage': 0.28,
                },
            ),
        ])

        retry_target_box_1 = {
            'x': 10,
            'y': 12,
            'width': 48,
            'height': 120,
            'frameWidth': 320,
            'frameHeight': 180,
            'selectionTime': 6.2,
            'selectionFrame': 186,
        }
        retry_target_box_2 = {
            **retry_target_box_1,
            'selectionTime': 7.1,
            'selectionFrame': 213,
        }

        with (
            patch.object(
                detector,
                '_discover_retry_target_boxes',
                return_value=[
                    {
                        'target_player_box': retry_target_box_1,
                        'score': 0.81,
                    },
                    {
                        'target_player_box': retry_target_box_2,
                        'score': 0.79,
                    },
                ],
            ) as discover_retry_mock,
            patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()),
        ):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={
                    'x': 0,
                    'y': 0,
                    'width': 1,
                    'height': 1,
                    'frameWidth': 320,
                    'frameHeight': 180,
                    'selectionTime': 5.0,
                    'selectionFrame': 150,
                },
            )

        self.assertEqual(detector.detect_shots.call_count, 1)
        discover_retry_mock.assert_not_called()
        self.assertFalse(result['auto_retry']['used'])
        self.assertEqual(result['auto_retry']['attempted'], 0)
        self.assertEqual(result['auto_retry']['finalSelectionTime'], 5.0)

    def test_detect_shots_with_clips_keeps_confirmed_and_extra_review_candidates(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 120,
                    'timestamp': 4.0,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.88,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.88,
                    'involvement_start_frame': 110,
                    'involvement_end_frame': 120,
                    'involvement_start_timestamp': 3.67,
                    'involvement_end_timestamp': 4.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.82,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.82,
                    'local_involvement_start_frame': 108,
                    'local_involvement_end_frame': 120,
                    'local_involvement_start_timestamp': 3.60,
                    'local_involvement_end_timestamp': 4.0,
                    'score_event_detected': True,
                },
                {
                    'frame': 240,
                    'timestamp': 8.0,
                    'made': False,
                    'owner': 'target',
                    'owner_confidence': 0.67,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.61,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.74,
                    'local_involvement_start_frame': 222,
                    'local_involvement_end_frame': 238,
                    'local_involvement_start_timestamp': 7.40,
                    'local_involvement_end_timestamp': 7.93,
                    'score_event_detected': False,
                },
            ],
            {
                'enabled': True,
                'coverage': 0.91,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 1)
        self.assertEqual(len(result['selected_shots']), 2)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['selected_shots'][1]['highlight_role'], 'possible')
        self.assertEqual(result['review_candidates'][0]['candidate_reason'], 'attempt_local_score_window')
        self.assertEqual(result['stats']['review_candidate_highlights'], 1)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['selection_summary']['mode'], 'mixed_with_review_candidates')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 1)
        self.assertEqual(result['diagnostics']['outcome'], 'confirmed_with_review_candidates')
        self.assertEqual(
            result['diagnostics']['summary'],
            '已导出 1 个已确认片段，并额外保留 1 个系统补充回合。建议先验收已确认片段，如怀疑漏剪，再检查系统补充片段。',
        )
        self.assertEqual(
            result['diagnostics']['recommendedActions'],
            ['先验收已确认片段，如怀疑漏剪，再检查系统补充片段。'],
        )
        self.assertEqual(result['pipeline']['scan']['totalShotEvents'], 2)
        self.assertEqual(result['pipeline']['scan']['madeShotEvents'], 1)
        self.assertEqual(result['pipeline']['attribution']['confirmedScores'], 1)
        self.assertEqual(result['pipeline']['export']['selectedClipCount'], 2)

    def test_detect_shots_with_clips_skips_conflicting_local_assist_review_when_target_keeps_ball_to_release(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 228,
                    'timestamp': 7.59,
                    'release_frame': 215,
                    'release_timestamp': 7.16,
                    'made': False,
                    'score_event_detected': False,
                    'owner': 'unknown',
                    'owner_confidence': 0.284,
                    'target_visible': True,
                    'attribution_highlight_role': 'none',
                    'attribution_highlight_confidence': 0.0,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'involvement_start_frame': 194,
                    'involvement_end_frame': 197,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.70,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.698,
                    'local_involvement_start_frame': 194,
                    'local_involvement_end_frame': 197,
                    'involvement_start_timestamp': 6.46,
                    'involvement_end_timestamp': 6.56,
                    'local_involvement_start_timestamp': 6.46,
                    'local_involvement_end_timestamp': 6.56,
                },
                {
                    'frame': 348,
                    'timestamp': 11.59,
                    'release_frame': 328,
                    'release_timestamp': 10.93,
                    'made': True,
                    'score_event_detected': True,
                    'owner': 'target',
                    'owner_confidence': 0.887,
                    'target_visible': True,
                    'attribution_highlight_role': 'score',
                    'attribution_highlight_confidence': 0.887,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.887,
                    'involvement_start_frame': 310,
                    'involvement_end_frame': 328,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.88,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.88,
                    'local_involvement_start_frame': 310,
                    'local_involvement_end_frame': 328,
                    'involvement_start_timestamp': 10.33,
                    'involvement_end_timestamp': 10.93,
                    'local_involvement_start_timestamp': 10.33,
                    'local_involvement_end_timestamp': 10.93,
                },
            ],
            {
                'enabled': True,
                'coverage': 1.0,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 1)
        self.assertEqual(result['review_candidates'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['stats']['possible_highlights'], 0)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 0)
        self.assertEqual(result['pipeline']['attribution']['reviewCandidates'], 0)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 0)

    def test_detect_shots_with_clips_skips_low_signal_review_candidate_when_confirmed_highlight_exists(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 120,
                    'timestamp': 4.0,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.88,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.88,
                    'involvement_start_frame': 110,
                    'involvement_end_frame': 120,
                    'involvement_start_timestamp': 3.67,
                    'involvement_end_timestamp': 4.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.82,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.82,
                    'local_involvement_start_frame': 108,
                    'local_involvement_end_frame': 120,
                    'local_involvement_start_timestamp': 3.60,
                    'local_involvement_end_timestamp': 4.0,
                    'score_event_detected': True,
                },
                {
                    'frame': 240,
                    'timestamp': 8.0,
                    'made': False,
                    'owner': 'target',
                    'owner_confidence': 0.67,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.61,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': 216,
                    'local_involvement_end_frame': 236,
                    'local_involvement_start_timestamp': 7.2,
                    'local_involvement_end_timestamp': 7.87,
                    'involvement_start_frame': 216,
                    'involvement_end_frame': 236,
                    'involvement_start_timestamp': 7.2,
                    'involvement_end_timestamp': 7.87,
                    'score_event_detected': False,
                },
            ],
            {
                'enabled': True,
                'coverage': 0.91,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 1)
        self.assertEqual(result['review_candidates'], [])
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['stats']['review_candidate_highlights'], 0)
        self.assertEqual(result['stats']['possible_highlights'], 0)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 0)
        self.assertEqual(result['diagnostics']['outcome'], 'confirmed_highlights')

    def test_detect_shots_with_clips_keeps_confirmed_and_weak_local_assist_make_as_possible(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 120,
                    'timestamp': 4.0,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.88,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.88,
                    'involvement_start_frame': 110,
                    'involvement_end_frame': 120,
                    'involvement_start_timestamp': 3.67,
                    'involvement_end_timestamp': 4.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.82,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.82,
                    'local_involvement_start_frame': 108,
                    'local_involvement_end_frame': 120,
                    'local_involvement_start_timestamp': 3.60,
                    'local_involvement_end_timestamp': 4.0,
                    'score_event_detected': True,
                },
                {
                    'frame': 336,
                    'timestamp': 11.2,
                    'made': True,
                    'owner': 'unknown',
                    'owner_confidence': 0.18,
                    'target_visible': False,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.32,
                    'local_highlight_role': 'assist',
                    'local_highlight_confidence': 0.52,
                    'local_involvement_start_frame': 276,
                    'local_involvement_end_frame': 304,
                    'local_involvement_start_timestamp': 9.2,
                    'local_involvement_end_timestamp': 10.13,
                    'score_event_detected': True,
                },
            ],
            {
                'enabled': True,
                'coverage': 0.88,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 2)
        self.assertEqual(len(result['review_candidates']), 0)
        self.assertEqual(len(result['selected_shots']), 2)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['selected_shots'][1]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][1]['candidate_reason'], 'local_assist')
        self.assertEqual(result['stats']['target_scores'], 1)
        self.assertEqual(result['stats']['target_assists'], 0)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 1)
        self.assertEqual(result['pipeline']['attribution']['possibleHighlights'], 1)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 1)

    def test_detect_shots_with_clips_promotes_local_assist_and_global_window_to_confirmed(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 336,
                    'timestamp': 11.2,
                    'made': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.52,
                    'attribution_highlight_confidence': 0.52,
                    'owner': 'unknown',
                    'owner_confidence': 0.18,
                    'target_visible': True,
                    'involvement_start_frame': 288,
                    'involvement_end_frame': 316,
                    'involvement_start_timestamp': 9.6,
                    'involvement_end_timestamp': 10.53,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.32,
                    'local_highlight_role': 'assist',
                    'local_highlight_confidence': 0.52,
                    'local_involvement_start_frame': 276,
                    'local_involvement_end_frame': 304,
                    'local_involvement_start_timestamp': 9.2,
                    'local_involvement_end_timestamp': 10.13,
                    'score_event_detected': True,
                },
            ],
            {
                'enabled': True,
                'coverage': 0.84,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 1)
        self.assertEqual(len(result['review_candidates']), 0)
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'assist')
        self.assertGreaterEqual(result['selected_shots'][0]['highlight_confidence'], 0.58)
        self.assertEqual(result['stats']['target_scores'], 0)
        self.assertEqual(result['stats']['target_assists'], 1)
        self.assertEqual(result['stats']['possible_highlights'], 0)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 0)
        self.assertEqual(result['pipeline']['attribution']['confirmedAssists'], 1)
        self.assertEqual(result['pipeline']['export']['assistClips'], 1)

    def test_detect_shots_with_clips_does_not_append_extra_attempt_fallbacks_when_related_highlights_exist(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 120,
                    'timestamp': 4.0,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.88,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.88,
                    'involvement_start_frame': 110,
                    'involvement_end_frame': 120,
                    'involvement_start_timestamp': 3.67,
                    'involvement_end_timestamp': 4.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.82,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.82,
                    'local_involvement_start_frame': 108,
                    'local_involvement_end_frame': 120,
                    'local_involvement_start_timestamp': 3.60,
                    'local_involvement_end_timestamp': 4.0,
                    'score_event_detected': True,
                },
                {
                    'frame': 300,
                    'timestamp': 10.0,
                    'made': True,
                    'owner': 'unknown',
                    'owner_confidence': 0.18,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.47,
                    'attribution_highlight_confidence': 0.47,
                    'involvement_start_frame': 252,
                    'involvement_end_frame': 282,
                    'involvement_start_timestamp': 8.4,
                    'involvement_end_timestamp': 9.4,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.0,
                    'local_highlight_role': 'none',
                    'local_highlight_confidence': 0.0,
                    'local_involvement_start_frame': None,
                    'local_involvement_end_frame': None,
                    'local_involvement_start_timestamp': None,
                    'local_involvement_end_timestamp': None,
                    'score_event_detected': True,
                },
                {
                    'frame': 420,
                    'release_frame': 406,
                    'timestamp': 14.0,
                    'made': False,
                    'owner': 'unknown',
                    'owner_confidence': 0.18,
                    'target_visible': False,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': False,
                    'local_owner_confidence': 0.90,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.75,
                    'local_involvement_start_frame': None,
                    'local_involvement_end_frame': None,
                    'local_involvement_start_timestamp': None,
                    'local_involvement_end_timestamp': None,
                    'score_event_detected': False,
                },
            ],
            {
                'enabled': True,
                'coverage': 0.83,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=4,
                after_seconds=1,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 2)
        self.assertEqual(len(result['review_candidates']), 0)
        self.assertEqual(len(result['selected_shots']), 2)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['selected_shots'][1]['highlight_role'], 'possible')
        self.assertEqual(result['selected_shots'][1]['candidate_reason'], 'global_assist_window')
        self.assertEqual(result['stats']['review_candidate_highlights'], 0)
        self.assertEqual(result['stats']['possible_highlights'], 1)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 1)
        self.assertEqual(result['diagnostics']['outcome'], 'confirmed_highlights')
        self.assertEqual(result['pipeline']['attribution']['reviewCandidates'], 0)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 1)

    def test_local_review_recovers_score_window_from_recent_ball_frames(self):
        tracker = self._build_local_review_tracker()
        frame_buffer = {}
        ball_positions = []

        for frame_index, ball_point in ((28, (96, 86)), (31, (99, 89)), (34, (102, 93))):
            frame_buffer[frame_index] = self._build_local_review_frame(ball_point=ball_point, width=240, height=240)
            ball_positions.append((ball_point, frame_index, 12, 12, 0.95))

        review = review_shot_with_local_window(
            ball_positions,
            frame_buffer,
            tracker,
            shot_release_frame=35,
        )

        self.assertEqual(review['owner'], 'target')
        self.assertEqual(review['highlight_role'], 'score')
        self.assertGreaterEqual(review['highlight_confidence'], 0.64)
        self.assertEqual(review['involvement_start_frame'], 28)
        self.assertEqual(review['involvement_end_frame'], 34)

    def test_local_review_recovers_score_when_target_scale_differs_from_selection_frame(self):
        tracker = self._build_local_review_tracker()
        frame_buffer = {}
        ball_positions = []
        smaller_target_bbox = (72, 54, 44, 104)

        for frame_index, ball_point in ((28, (95, 92)), (31, (98, 96)), (34, (101, 100))):
            frame_buffer[frame_index] = self._build_scaled_target_frame(
                smaller_target_bbox,
                ball_point=ball_point,
            )
            ball_positions.append((ball_point, frame_index, 12, 12, 0.95))

        review = review_shot_with_local_window(
            ball_positions,
            frame_buffer,
            tracker,
            shot_release_frame=35,
        )

        self.assertEqual(review['owner'], 'target')
        self.assertEqual(review['highlight_role'], 'score')
        self.assertGreaterEqual(review['highlight_confidence'], 0.64)
        self.assertEqual(review['involvement_start_frame'], 28)
        self.assertEqual(review['involvement_end_frame'], 34)

    def test_local_review_prefers_release_frame_tracker_box_as_expected_bbox(self):
        tracker = self._build_local_review_tracker()
        tracker.current_bbox = (92, 60, 46, 110)
        tracker.last_bbox = (88, 58, 48, 114)
        tracker.get_box_at_frame = lambda frame_index, max_gap=12: (74, 56, 42, 102)

        frame_buffer = {
            28: self._build_scaled_target_frame((74, 56, 42, 102), ball_point=(96, 92)),
        }
        ball_positions = [((96, 92), 28, 12, 12, 0.95)]
        captured_expected_bboxes = []

        def fake_find_local_target_match(frame, frame_index, ball_point, tracker_arg, expected_bbox):
            captured_expected_bboxes.append(expected_bbox)
            return {'score': 0.72, 'inside': True}

        with patch('player_tracker._find_local_target_match', side_effect=fake_find_local_target_match), \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._control_window_confidence', return_value=0.72):
            build_windows_mock.side_effect = [
                [{
                    'start_frame': 28,
                    'end_frame': 28,
                    'sample_count': 1,
                    'span': 0,
                    'best_score': 0.72,
                    'mean_score': 0.72,
                }],
                [],
            ]

            review = review_shot_with_local_window(
                ball_positions,
                frame_buffer,
                tracker,
                shot_release_frame=28,
            )

        self.assertEqual(captured_expected_bboxes, [(74, 56, 42, 102)])
        self.assertEqual(review['owner'], 'target')
        self.assertEqual(review['highlight_role'], 'score')

    def test_local_review_supports_assist_with_coherent_receiver_path(self):
        tracker = self._build_local_review_tracker()
        frame_buffer = {}
        ball_positions = []
        trajectory = (
            (10, (96, 86)),
            (13, (99, 89)),
            (16, (102, 93)),
            (22, (180, 104)),
            (26, (240, 108)),
            (30, (310, 112)),
            (34, (380, 116)),
            (38, (450, 120)),
            (40, (500, 124)),
        )

        for frame_index, ball_point in trajectory:
            frame_buffer[frame_index] = self._build_local_review_frame(ball_point=ball_point)
            ball_positions.append((ball_point, frame_index, 12, 12, 0.95))

        review = review_shot_with_local_window(
            ball_positions,
            frame_buffer,
            tracker,
            shot_release_frame=40,
        )

        self.assertEqual(review['highlight_role'], 'assist')
        self.assertEqual(review['involvement_start_frame'], 10)
        self.assertEqual(review['involvement_end_frame'], 16)

    def test_local_review_confirms_strong_aggregate_assist_when_continuity_is_slightly_low(self):
        tracker = self._build_local_review_tracker()
        frame_buffer = {
            frame_index: self._build_local_review_frame(ball_point=ball_point)
            for frame_index, ball_point in (
                (10, (96, 86)),
                (13, (99, 89)),
                (16, (102, 93)),
                (22, (180, 104)),
                (26, (240, 108)),
                (30, (310, 112)),
                (34, (380, 116)),
                (38, (450, 120)),
                (40, (500, 124)),
            )
        }
        ball_positions = [
            (ball_point, frame_index, 12, 12, 0.95)
            for frame_index, ball_point in (
                (10, (96, 86)),
                (13, (99, 89)),
                (16, (102, 93)),
                (22, (180, 104)),
                (26, (240, 108)),
                (30, (310, 112)),
                (34, (380, 116)),
                (38, (450, 120)),
                (40, (500, 124)),
            )
        ]

        with patch('player_tracker._find_local_target_match', return_value={'score': 0.72, 'inside': True}), \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._control_window_confidence', return_value=0.84), \
             patch('player_tracker._handoff_confidence', return_value=0.62), \
             patch('player_tracker._post_handoff_continuity_confidence', return_value=0.28), \
             patch('player_tracker._terminal_release_window_confidence', return_value=0.56), \
             patch('player_tracker._receiver_trajectory_confidence', return_value=0.52):
            build_windows_mock.side_effect = [
                [],
                [{
                    'start_frame': 10,
                    'end_frame': 16,
                    'sample_count': 3,
                    'span': 6,
                    'best_score': 0.74,
                    'mean_score': 0.72,
                }],
            ]

            review = review_shot_with_local_window(
                ball_positions,
                frame_buffer,
                tracker,
                shot_release_frame=40,
            )

        self.assertEqual(review['highlight_role'], 'assist')
        self.assertGreaterEqual(review['highlight_confidence'], 0.60)
        self.assertEqual(review['involvement_start_frame'], 10)
        self.assertEqual(review['involvement_end_frame'], 16)

    def test_local_review_confirms_high_anchor_assist_when_one_tail_signal_dips_below_tolerance(self):
        tracker = self._build_local_review_tracker()
        frame_buffer = {
            frame_index: self._build_local_review_frame(ball_point=ball_point)
            for frame_index, ball_point in (
                (10, (96, 86)),
                (13, (99, 89)),
                (16, (102, 93)),
                (22, (180, 104)),
                (26, (240, 108)),
                (30, (310, 112)),
                (34, (380, 116)),
                (38, (450, 120)),
                (40, (500, 124)),
            )
        }
        ball_positions = [
            (ball_point, frame_index, 12, 12, 0.95)
            for frame_index, ball_point in (
                (10, (96, 86)),
                (13, (99, 89)),
                (16, (102, 93)),
                (22, (180, 104)),
                (26, (240, 108)),
                (30, (310, 112)),
                (34, (380, 116)),
                (38, (450, 120)),
                (40, (500, 124)),
            )
        ]

        with patch('player_tracker._find_local_target_match', return_value={'score': 0.72, 'inside': True}), \
             patch('player_tracker._build_control_windows') as build_windows_mock, \
             patch('player_tracker._control_window_confidence', return_value=0.86), \
             patch('player_tracker._handoff_confidence', return_value=0.61), \
             patch('player_tracker._post_handoff_continuity_confidence', return_value=0.21), \
             patch('player_tracker._terminal_release_window_confidence', return_value=0.58), \
             patch('player_tracker._receiver_trajectory_confidence', return_value=0.52):
            build_windows_mock.side_effect = [
                [],
                [{
                    'start_frame': 10,
                    'end_frame': 16,
                    'sample_count': 3,
                    'span': 6,
                    'best_score': 0.74,
                    'mean_score': 0.72,
                }],
            ]

            review = review_shot_with_local_window(
                ball_positions,
                frame_buffer,
                tracker,
                shot_release_frame=40,
            )

        self.assertEqual(review['highlight_role'], 'assist')
        self.assertGreaterEqual(review['highlight_confidence'], 0.59)
        self.assertEqual(review['involvement_start_frame'], 10)
        self.assertEqual(review['involvement_end_frame'], 16)

    def test_tracker_add_reference_sample_registers_nearby_frame(self):
        tracker = self._build_local_review_tracker()

        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(frame, (69, 40), (129, 180), (20, 80, 220), -1)
        cv2.rectangle(frame, (87, 70), (111, 108), (255, 255, 255), -1)

        sample = tracker.add_reference_sample(frame, frame_index=12, anchor_bbox=(60, 40, 60, 140))

        self.assertIsNotNone(sample)
        self.assertIn(12, tracker.reference_sample_frames)
        self.assertGreaterEqual(len(tracker.reference_hists), 1)
        self.assertGreaterEqual(len(tracker.reference_templates), 1)

    def test_tracker_register_tracking_sample_keeps_stable_runtime_reference(self):
        tracker = self._build_local_review_tracker()
        tracker.current_bbox = (66, 42, 60, 140)
        tracker.last_bbox = tracker.current_bbox

        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(frame, (66, 42), (126, 182), (20, 80, 220), -1)
        cv2.rectangle(frame, (84, 72), (108, 110), (255, 255, 255), -1)

        sample = tracker.register_tracking_sample(
            frame,
            frame_index=18,
            bbox=(66, 42, 60, 140),
        )

        self.assertIsNotNone(sample)
        self.assertIn(18, tracker.reference_sample_frames)
        self.assertGreaterEqual(sample['histScore'], 0.34)
        self.assertGreaterEqual(sample['score'], 0.52)

    def test_tracker_register_tracking_sample_rejects_drifted_runtime_reference(self):
        tracker = self._build_local_review_tracker()
        tracker.current_bbox = (66, 42, 60, 140)
        tracker.last_bbox = tracker.current_bbox

        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.rectangle(frame, (66, 42), (126, 182), (40, 200, 60), -1)
        cv2.rectangle(frame, (82, 74), (110, 120), (0, 0, 0), -1)

        sample = tracker.register_tracking_sample(
            frame,
            frame_index=22,
            bbox=(66, 42, 60, 140),
        )

        self.assertIsNone(sample)
        self.assertNotIn(22, tracker.reference_sample_frames)

    def test_related_shots_skip_low_signal_possible_when_confirmed_highlight_exists(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 120,
                'timestamp': 4.0,
                'made': True,
                'highlight_role': 'score',
                'owner': 'target',
                'owner_confidence': 0.92,
                'target_visible': True,
            },
            {
                'frame': 300,
                'timestamp': 10.0,
                'made': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': True,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.5},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(len(related_shots), 1)
        self.assertEqual(related_shots[0]['highlight_role'], 'score')

    def test_related_shots_keep_low_signal_possible_when_no_confirmed_highlight_exists(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 300,
                'timestamp': 10.0,
                'made': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': True,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.5},
        )

        self.assertEqual(selection_summary['confirmed'], 0)
        self.assertEqual(selection_summary['possible'], 1)
        self.assertEqual(len(related_shots), 1)
        self.assertEqual(related_shots[0]['highlight_role'], 'possible')
        self.assertEqual(related_shots[0]['candidate_reason'], 'target_visible')

    def test_related_shots_promote_strong_local_review_to_confirmed(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 300,
                'timestamp': 10.0,
                'made': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': False,
                'local_target_visible': True,
                'local_owner_confidence': 0.74,
                'local_highlight_role': 'assist',
                'local_highlight_confidence': 0.81,
                'local_involvement_start_frame': 240,
                'local_involvement_end_frame': 268,
                'local_involvement_start_timestamp': 8.0,
                'local_involvement_end_timestamp': 8.93,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.18},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 8.0)

    def test_related_shots_confirm_local_assist_below_score_threshold(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 312,
                'timestamp': 10.4,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'owner': 'unknown',
                'owner_confidence': 0.18,
                'target_visible': False,
                'local_target_visible': True,
                'local_owner_confidence': 0.34,
                'local_highlight_role': 'assist',
                'local_highlight_confidence': 0.61,
                'local_involvement_start_frame': 246,
                'local_involvement_end_frame': 276,
                'local_involvement_start_timestamp': 8.2,
                'local_involvement_end_timestamp': 9.2,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.64},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertGreaterEqual(related_shots[0]['highlight_confidence'], 0.61)
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 8.2)

    def test_related_shots_promote_strong_partial_local_assist_to_confirmed(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 324,
                'timestamp': 10.8,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'owner': 'unknown',
                'owner_confidence': 0.16,
                'target_visible': False,
                'local_target_visible': True,
                'local_owner_confidence': 0.27,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.61,
                'local_involvement_start_frame': 252,
                'local_involvement_end_frame': 282,
                'local_involvement_start_timestamp': 8.4,
                'local_involvement_end_timestamp': 9.4,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.66},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertGreaterEqual(related_shots[0]['highlight_confidence'], 0.61)
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 8.4)

    def test_related_shots_promote_aligned_partial_assist_signals_to_confirmed(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 360,
                'timestamp': 12.0,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.55,
                'attribution_highlight_confidence': 0.55,
                'owner': 'unknown',
                'owner_confidence': 0.22,
                'target_visible': True,
                'involvement_start_frame': 300,
                'involvement_end_frame': 330,
                'involvement_start_timestamp': 10.0,
                'involvement_end_timestamp': 11.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.29,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.53,
                'local_involvement_start_frame': 306,
                'local_involvement_end_frame': 334,
                'local_involvement_start_timestamp': 10.2,
                'local_involvement_end_timestamp': 11.13,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.78},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertGreaterEqual(related_shots[0]['highlight_confidence'], 0.60)
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 10.0)

    def test_related_shots_promote_dual_partial_assist_signals_at_relaxed_boundary(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 360,
                'timestamp': 12.0,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.52,
                'attribution_highlight_confidence': 0.52,
                'owner': 'unknown',
                'owner_confidence': 0.21,
                'target_visible': True,
                'involvement_start_frame': 300,
                'involvement_end_frame': 330,
                'involvement_start_timestamp': 10.0,
                'involvement_end_timestamp': 11.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.28,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.52,
                'local_involvement_start_frame': 306,
                'local_involvement_end_frame': 334,
                'local_involvement_start_timestamp': 10.2,
                'local_involvement_end_timestamp': 11.13,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.76},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertGreaterEqual(related_shots[0]['highlight_confidence'], 0.58)
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 10.0)

    def test_related_shots_promote_local_assist_review_and_global_window_at_relaxed_boundary(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 360,
                'timestamp': 12.0,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.52,
                'attribution_highlight_confidence': 0.52,
                'owner': 'unknown',
                'owner_confidence': 0.21,
                'target_visible': True,
                'involvement_start_frame': 300,
                'involvement_end_frame': 330,
                'involvement_start_timestamp': 10.0,
                'involvement_end_timestamp': 11.0,
                'local_target_visible': False,
                'local_owner_confidence': 0.28,
                'local_highlight_role': 'assist',
                'local_highlight_confidence': 0.52,
                'local_involvement_start_frame': 306,
                'local_involvement_end_frame': 334,
                'local_involvement_start_timestamp': 10.2,
                'local_involvement_end_timestamp': 11.13,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.77},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'assist')
        self.assertGreaterEqual(related_shots[0]['highlight_confidence'], 0.58)
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 10.0)

    def test_related_shots_keep_partial_local_assist_evidence_as_possible(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 300,
                'timestamp': 10.0,
                'made': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': False,
                'local_target_visible': True,
                'local_owner_confidence': 0.41,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.46,
                'local_involvement_start_frame': 240,
                'local_involvement_end_frame': 268,
                'local_involvement_start_timestamp': 8.0,
                'local_involvement_end_timestamp': 8.93,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.52},
        )

        self.assertEqual(selection_summary['confirmed'], 0)
        self.assertEqual(selection_summary['possible'], 1)
        self.assertEqual(related_shots[0]['highlight_role'], 'possible')
        self.assertEqual(related_shots[0]['candidate_reason'], 'local_assist_window')
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 8.0)

    def test_related_shots_keep_local_assist_review_below_confirm_threshold_as_possible(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 336,
                'timestamp': 11.2,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'owner': 'unknown',
                'owner_confidence': 0.18,
                'target_visible': False,
                'local_target_visible': False,
                'local_owner_confidence': 0.32,
                'local_highlight_role': 'assist',
                'local_highlight_confidence': 0.52,
                'local_involvement_start_frame': 276,
                'local_involvement_end_frame': 304,
                'local_involvement_start_timestamp': 9.2,
                'local_involvement_end_timestamp': 10.13,
                'score_event_detected': True,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.81},
        )

        self.assertEqual(selection_summary['confirmed'], 0)
        self.assertEqual(selection_summary['possible'], 1)
        self.assertEqual(related_shots[0]['highlight_role'], 'possible')
        self.assertEqual(related_shots[0]['candidate_reason'], 'local_assist')
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 9.2)

    def test_related_shots_keep_partial_global_assist_evidence_as_possible(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 360,
                'timestamp': 12.0,
                'made': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.47,
                'owner': 'unknown',
                'owner_confidence': 0.18,
                'target_visible': True,
                'involvement_start_frame': 300,
                'involvement_end_frame': 328,
                'involvement_start_timestamp': 10.0,
                'involvement_end_timestamp': 10.93,
                'local_target_visible': False,
                'local_owner_confidence': 0.0,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.0,
                'local_involvement_start_frame': None,
                'local_involvement_end_frame': None,
                'local_involvement_start_timestamp': None,
                'local_involvement_end_timestamp': None,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.74},
        )

        self.assertEqual(selection_summary['confirmed'], 0)
        self.assertEqual(selection_summary['possible'], 1)
        self.assertEqual(related_shots[0]['highlight_role'], 'possible')
        self.assertEqual(related_shots[0]['candidate_reason'], 'global_assist_window')
        self.assertEqual(related_shots[0]['involvement_start_timestamp'], 10.0)

    def test_related_shots_promote_backfilled_made_score_with_strong_local_visibility(self):
        detector = object.__new__(BasketballShotDetector)

        made_shots = [
            {
                'frame': 348,
                'timestamp': 11.59,
                'release_frame': 328,
                'release_timestamp': 10.93,
                'made': True,
                'score_event_detected': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': True,
                'local_target_visible': True,
                'local_owner_confidence': 0.708,
                'local_highlight_role': 'none',
                'local_highlight_confidence': 0.675,
                'local_involvement_start_frame': None,
                'local_involvement_end_frame': None,
                'local_involvement_start_timestamp': None,
                'local_involvement_end_timestamp': None,
            },
        ]

        related_shots, selection_summary = detector._select_related_made_shots(
            made_shots,
            target_player_box={'selectionTime': 12.1},
            tracking_summary={'coverage': 1.0},
        )

        self.assertEqual(selection_summary['confirmed'], 1)
        self.assertEqual(selection_summary['possible'], 0)
        self.assertEqual(related_shots[0]['highlight_role'], 'score')
        self.assertEqual(related_shots[0]['owner'], 'target')
        self.assertGreaterEqual(related_shots[0]['owner_confidence'], 0.708)

    def test_related_shots_returns_empty_when_target_not_locked(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = lambda *args, **kwargs: ([
            {
                'frame': 120,
                'timestamp': 4.0,
                'made': True,
                'highlight_role': 'none',
                'owner': 'unknown',
                'owner_confidence': 0.0,
                'target_visible': False,
            },
        ], {'enabled': True, 'coverage': 0.12})

        with (
            patch.object(detector, '_discover_retry_target_boxes', return_value=[]),
            patch('shot_detector_video.cv2.VideoCapture') as video_capture,
        ):
            cap = video_capture.return_value
            cap.get.side_effect = [300, 30]
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=6,
                after_seconds=2,
                target_player_box={'selectionTime': 1.0},
            )

        self.assertEqual(result['stats']['possible_highlights'], 0)
        self.assertEqual(result['selection_summary']['mode'], 'no_target_highlights')
        self.assertEqual(result['selected_made_shots'], [])
        self.assertEqual(result['selected_shots'], [])
        self.assertEqual(result['diagnostics']['outcome'], 'global_makes_without_target')

    def test_attempt_dedup_blocks_same_shot_retrigger(self):
        detector = object.__new__(BasketballShotDetector)

        self.assertTrue(
            detector._is_duplicate_attempt(
                last_release_frame=100,
                last_down_frame=112,
                current_release_frame=104,
                current_down_frame=118,
            )
        )

    def test_attempt_dedup_keeps_distinct_quick_second_shot(self):
        detector = object.__new__(BasketballShotDetector)

        self.assertFalse(
            detector._is_duplicate_attempt(
                last_release_frame=100,
                last_down_frame=112,
                current_release_frame=124,
                current_down_frame=134,
            )
        )

    def test_select_target_review_candidates_does_not_truncate_above_four(self):
        detector = object.__new__(BasketballShotDetector)

        all_shots = []
        for index in range(6):
            all_shots.append({
                'frame': 120 + index * 60,
                'timestamp': 4.0 + index * 2.0,
                'made': False,
                'owner': 'target',
                'owner_confidence': 0.68,
                'target_visible': True,
                'attribution_highlight_role': 'assist',
                'attribution_highlight_confidence': 0.62,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'involvement_start_frame': 96 + index * 60,
                'involvement_end_frame': 108 + index * 60,
                'involvement_start_timestamp': 3.2 + index * 2.0,
                'involvement_end_timestamp': 3.6 + index * 2.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.58,
                'local_highlight_role': 'score',
                'local_highlight_confidence': 0.72,
                'local_involvement_start_frame': 100 + index * 60,
                'local_involvement_end_frame': 118 + index * 60,
                'local_involvement_start_timestamp': 3.33 + index * 2.0,
                'local_involvement_end_timestamp': 3.93 + index * 2.0,
                'score_event_detected': False,
            })

        review_candidates = detector._select_target_review_candidates(
            all_shots,
            target_player_box={'selectionTime': 1.0},
            tracking_summary={'coverage': 0.88},
        )

        self.assertEqual(len(review_candidates), 6)
        self.assertTrue(all(candidate['highlight_role'] == 'possible' for candidate in review_candidates))

    def test_merge_review_candidates_dedups_same_attempt(self):
        detector = object.__new__(BasketballShotDetector)

        selected_shots = [
            {
                'frame': 112,
                'release_frame': 100,
                'timestamp': 3.73,
                'highlight_role': 'score',
            },
        ]
        review_candidates = [
            {
                'frame': 118,
                'release_frame': 104,
                'timestamp': 3.93,
                'highlight_role': 'possible',
                'candidate_reason': 'attempt_target_release',
            },
        ]

        merged_shots, merged_review_candidates = detector._merge_review_candidates(
            selected_shots,
            review_candidates,
        )

        self.assertEqual(len(merged_shots), 1)
        self.assertEqual(len(merged_review_candidates), 0)

    def test_merge_review_candidates_keeps_distinct_quick_second_attempt(self):
        detector = object.__new__(BasketballShotDetector)

        selected_shots = [
            {
                'frame': 112,
                'release_frame': 100,
                'timestamp': 3.73,
                'highlight_role': 'score',
            },
        ]
        review_candidates = [
            {
                'frame': 134,
                'release_frame': 124,
                'timestamp': 4.47,
                'highlight_role': 'possible',
                'candidate_reason': 'attempt_target_release',
            },
        ]

        merged_shots, merged_review_candidates = detector._merge_review_candidates(
            selected_shots,
            review_candidates,
        )

        self.assertEqual(len(merged_shots), 2)
        self.assertEqual(len(merged_review_candidates), 1)
        self.assertEqual(merged_review_candidates[0]['frame'], 134)

    def test_merge_review_candidates_skips_redundant_preceding_attempt_review(self):
        detector = object.__new__(BasketballShotDetector)

        selected_shots = [
            {
                'frame': 348,
                'release_frame': 336,
                'timestamp': 11.59,
                'highlight_role': 'score',
            },
        ]
        review_candidates = [
            {
                'frame': 240,
                'release_frame': 228,
                'timestamp': 8.0,
                'highlight_role': 'possible',
                'candidate_reason': 'attempt_local_score_window',
                'candidate_source': 'attempt_review',
            },
        ]

        merged_shots, merged_review_candidates = detector._merge_review_candidates(
            selected_shots,
            review_candidates,
        )

        self.assertEqual(len(merged_shots), 1)
        self.assertEqual(len(merged_review_candidates), 0)

    def test_select_target_highlights_returns_separated_selection_stages(self):
        detector = object.__new__(BasketballShotDetector)

        all_shots = [
            {
                'frame': 120,
                'timestamp': 4.0,
                'made': True,
                'owner': 'target',
                'owner_confidence': 0.88,
                'target_visible': True,
                'highlight_role': 'score',
                'highlight_confidence': 0.88,
                'local_target_visible': True,
                'local_owner_confidence': 0.8,
                'local_highlight_role': 'score',
                'local_highlight_confidence': 0.8,
            },
            {
                'frame': 240,
                'timestamp': 8.0,
                'made': False,
                'owner': 'target',
                'owner_confidence': 0.67,
                'target_visible': True,
                'highlight_role': 'none',
                'highlight_confidence': 0.0,
                'local_target_visible': True,
                'local_owner_confidence': 0.61,
                'local_highlight_role': 'score',
                'local_highlight_confidence': 0.74,
            },
        ]

        selection_result = detector._select_target_highlights(
            all_shots,
            {'enabled': True, 'coverage': 0.91},
            {'selectionTime': 1.0},
        )

        self.assertEqual(len(selection_result['made_shots']), 1)
        self.assertEqual(len(selection_result['related_made_shots']), 1)
        self.assertEqual(len(selection_result['review_candidates']), 1)
        self.assertEqual(len(selection_result['merged_review_candidates']), 1)
        self.assertEqual(len(selection_result['selected_shots']), 2)
        self.assertEqual(selection_result['selection_summary']['mode'], 'mixed_with_review_candidates')

    def test_detect_shots_with_clips_suppresses_preceding_attempt_review_when_confirmed_highlight_exists(self):
        detector = object.__new__(BasketballShotDetector)
        detector.detect_shots = Mock(return_value=(
            [
                {
                    'frame': 240,
                    'timestamp': 8.0,
                    'made': False,
                    'owner': 'target',
                    'owner_confidence': 0.67,
                    'target_visible': True,
                    'highlight_role': 'none',
                    'highlight_confidence': 0.0,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.61,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.74,
                    'local_involvement_start_frame': 222,
                    'local_involvement_end_frame': 238,
                    'local_involvement_start_timestamp': 7.40,
                    'local_involvement_end_timestamp': 7.93,
                    'score_event_detected': False,
                },
                {
                    'frame': 348,
                    'timestamp': 11.59,
                    'made': True,
                    'owner': 'target',
                    'owner_confidence': 0.91,
                    'target_visible': True,
                    'highlight_role': 'score',
                    'highlight_confidence': 0.91,
                    'local_target_visible': True,
                    'local_owner_confidence': 0.84,
                    'local_highlight_role': 'score',
                    'local_highlight_confidence': 0.84,
                    'score_event_detected': True,
                },
            ],
            {
                'enabled': True,
                'coverage': 1.0,
            },
        ))

        with patch('shot_detector_video.cv2.VideoCapture', return_value=FakeVideoCapture()):
            result = detector.detect_shots_with_clips(
                'video.mp4',
                before_seconds=6,
                after_seconds=2,
                target_player_box={'x': 0, 'y': 0, 'width': 1, 'height': 1},
            )

        self.assertEqual(len(result['selected_made_shots']), 1)
        self.assertEqual(len(result['selected_shots']), 1)
        self.assertEqual(result['selected_shots'][0]['highlight_role'], 'score')
        self.assertEqual(result['review_candidates'], [])
        self.assertEqual(result['stats']['review_candidate_highlights'], 0)
        self.assertEqual(result['stats']['possible_highlights'], 0)
        self.assertEqual(result['selection_summary']['mode'], 'mixed')
        self.assertEqual(result['selection_summary']['confirmed'], 1)
        self.assertEqual(result['selection_summary']['possible'], 0)
        self.assertEqual(result['pipeline']['export']['possibleClips'], 0)


if __name__ == '__main__':
    unittest.main()
