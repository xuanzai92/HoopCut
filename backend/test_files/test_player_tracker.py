"""
独立测试目标球员跟踪与最小归因逻辑（不依赖模型推理）
"""
import os
import sys

import numpy as np

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from player_tracker import TargetPlayerTracker, classify_shot_involvement, classify_shot_owner


def run_tracker_smoke_test():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker = TargetPlayerTracker({
        'x': 100,
        'y': 120,
        'width': 140,
        'height': 300,
        'frameWidth': 1280,
        'frameHeight': 720,
    })

    tracker.initialize(frame, 0)
    tracker.update(frame, 1)
    tracker.update(frame, 2)

    ball_positions = [
        ((150, 220), 1, 12, 12, 0.9),
        ((165, 235), 2, 12, 12, 0.92),
        ((180, 250), 3, 12, 12, 0.95),
    ]

    owner, confidence, visible = classify_shot_owner(ball_positions, tracker, 3)
    summary = tracker.get_summary()

    print('tracker summary:', summary)
    print('owner:', owner)
    print('confidence:', confidence)
    print('visible:', visible)

    if owner != 'target':
        raise AssertionError('预期 smoke test 归因为 target')

    if confidence <= 0:
        raise AssertionError('预期 smoke test 置信度大于 0')


def run_assist_smoke_test():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker = TargetPlayerTracker({
        'x': 100,
        'y': 120,
        'width': 140,
        'height': 300,
        'frameWidth': 1280,
        'frameHeight': 720,
    })

    tracker.initialize(frame, 0)
    for frame_index in range(1, 41):
        tracker.update(frame, frame_index)

    ball_positions = [
        ((155, 225), 4, 12, 12, 0.91),
        ((165, 235), 5, 12, 12, 0.93),
        ((175, 245), 6, 12, 12, 0.95),
        ((620, 210), 24, 12, 12, 0.92),
        ((640, 200), 28, 12, 12, 0.94),
        ((660, 190), 32, 12, 12, 0.95),
    ]

    attribution = classify_shot_involvement(ball_positions, tracker, 32)

    print('assist attribution:', attribution)

    if attribution['highlight_role'] != 'assist':
      raise AssertionError('预期 assist smoke test 归因为 assist')

    if attribution['owner'] != 'unknown':
      raise AssertionError('预期 assist smoke test 的出手人不是 target')


if __name__ == '__main__':
    run_tracker_smoke_test()
    run_assist_smoke_test()
