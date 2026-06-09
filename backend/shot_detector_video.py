import os
from ultralytics import YOLO
import cv2
import math
import numpy as np
import tempfile
from player_tracker import (
    TargetPlayerTracker,
    classify_shot_involvement,
    draw_target_bbox,
    review_shot_with_local_window,
)
from utils import (
    score, detect_down, detect_up, in_hoop_region,
    find_recent_down_frame, find_recent_score_event, find_recent_up_frame,
    clean_hoop_pos, clean_ball_pos, get_device
)
from typing import List, Dict, Tuple, Optional

class BasketballShotDetector:
    """
    批量处理篮球视频，检测所有进球时刻
    """
    LOCAL_REVIEW_CONFIRM_THRESHOLD = 0.72
    LOCAL_ASSIST_CONFIRM_THRESHOLD = 0.58
    LOCAL_REVIEW_POSSIBLE_THRESHOLD = 0.45
    LOCAL_REVIEW_VISIBLE_THRESHOLD = 0.35
    LOCAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD = 0.42
    GLOBAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD = 0.42
    PARTIAL_ASSIST_PROMOTION_THRESHOLD = 0.60
    DUAL_SIGNAL_ASSIST_PROMOTION_BOOST = 0.06
    DUAL_SIGNAL_ASSIST_PROMOTION_THRESHOLD = 0.58
    BACKFILLED_MADE_SCORE_THRESHOLD = 0.68
    ATTEMPT_REVIEW_PRIMARY_THRESHOLD = 0.58
    ATTEMPT_REVIEW_FALLBACK_THRESHOLD = 0.48
    HIGH_SIGNAL_REVIEW_REASONS = {
        'attempt_local_score_window',
        'local_assist',
        'local_assist_window',
        'global_assist_window',
    }
    HIGH_SIGNAL_MIXED_POSSIBLE_REASONS = {
        'local_score',
        'local_assist',
        'local_assist_window',
        'global_assist_window',
    }
    ATTEMPT_RELEASE_DEDUP_FRAMES = 8
    ATTEMPT_DOWN_DEDUP_FRAMES = 12
    ATTEMPT_REVIEW_REDUNDANT_PRECEDING_CONFIRM_GAP_SECONDS = 4.0
    REFERENCE_PRIME_OFFSETS_SECONDS = (0.25, 0.5, 0.85, 1.2, 1.6)
    REFERENCE_PRIME_MIN_SUCCESSFUL_SAMPLES = 4
    REFERENCE_PRIME_DENSE_WINDOW_SECONDS = 0.9
    REFERENCE_PRIME_DENSE_STEP_SECONDS = 0.12
    PREALIGN_MIN_SELECTION_TIME_SECONDS = 1.0
    PREALIGN_MIN_EARLIER_SHIFT_SECONDS = 0.35
    PREALIGN_MIN_CANDIDATE_SCORE = 0.72
    LIVE_REFERENCE_REFRESH_INTERVAL_FRAMES = 18
    LIVE_REFERENCE_REFRESH_MIN_CONFIDENCE = 0.62
    LIVE_REFERENCE_REFRESH_STATUSES = {'tracking', 'revalidated', 'reacquired'}
    AUTO_RETRY_TRIGGER_OUTCOMES = {
        'review_candidates',
        'target_attempt_fallback',
        'global_makes_without_target',
        'no_makes_detected',
        'no_attempts_detected',
    }
    AUTO_RETRY_MAX_CANDIDATES = 3
    AUTO_RETRY_MIN_TRACKING_COVERAGE = 0.55
    AUTO_RETRY_BACKWARD_OFFSETS_SECONDS = (2.4, 3.2, 4.6, 6.0)
    AUTO_RETRY_EARLIER_FRAME_BONUS_PER_SECOND = 0.012
    AUTO_RETRY_EARLIER_FRAME_BONUS_CAP = 0.06
    DEFAULT_INVOLVEMENT_LEAD_SECONDS = 1.0
    ASSIST_INVOLVEMENT_LEAD_SECONDS = 2.0
    TARGET_RELATED_REVIEW_LEAD_SECONDS = 1.5

    def __init__(self, model_path='best.pt', confidence_threshold=0.25):
        """
        初始化检测器
        
        Args:
            model_path: YOLO模型文件路径
            confidence_threshold: 检测置信度阈值
        """
        self.model = YOLO(model_path)
        self.class_names = ['Basketball', 'Basketball Hoop']
        self.device = get_device()
        self.confidence_threshold = confidence_threshold
        
        print(f"使用设备: {self.device}")
        print(f"模型加载完成: {model_path}")

    @staticmethod
    def _build_possible_highlight(shot: Dict, reason: str) -> Dict:
        possible_shot = dict(shot)
        possible_shot['highlight_role'] = 'possible'
        possible_shot['highlight_confidence'] = round(
            max(
                float(shot.get('highlight_confidence') or 0.0),
                float(shot.get('attribution_highlight_confidence') or 0.0),
                float(shot.get('owner_confidence') or 0.0),
                0.25,
            ),
            3,
        )
        possible_shot['candidate_reason'] = reason
        return possible_shot

    def _build_attempt_review_candidate(self, shot: Dict, reason: str, confidence: float) -> Dict:
        candidate_shot = self._build_possible_highlight(
            self._inherit_local_involvement(shot),
            reason=reason,
        )
        candidate_shot['made'] = False
        candidate_shot['highlight_confidence'] = round(
            max(
                float(candidate_shot.get('highlight_confidence') or 0.0),
                float(confidence),
                0.35,
            ),
            3,
        )
        candidate_shot['clip_export'] = True
        candidate_shot['candidate_source'] = 'attempt_review'
        return candidate_shot

    def _build_target_attempt_fallback_candidate(self, shot: Dict, reason: str, confidence: float) -> Dict:
        candidate_shot = self._build_possible_highlight(
            self._inherit_local_involvement(shot),
            reason=reason,
        )
        candidate_shot['made'] = bool(shot.get('made'))
        candidate_shot['highlight_confidence'] = round(
            max(
                float(candidate_shot.get('highlight_confidence') or 0.0),
                float(shot.get('local_highlight_confidence') or 0.0),
                float(shot.get('local_owner_confidence') or 0.0),
                float(confidence),
                0.28,
            ),
            3,
        )
        candidate_shot['clip_export'] = True
        candidate_shot['candidate_source'] = 'target_attempt_fallback'
        return candidate_shot

    def _has_partial_local_assist_evidence(self, shot: Dict) -> bool:
        if str(shot.get('local_highlight_role') or 'none') != 'none':
            return False

        if self._has_conflicting_target_release_context(shot):
            return False

        local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        local_target_visible = bool(shot.get('local_target_visible'))
        has_involvement_window = (
            shot.get('local_involvement_start_frame') is not None
            and shot.get('local_involvement_end_frame') is not None
        )

        return (
            local_target_visible
            and has_involvement_window
            and local_confidence >= self.LOCAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD
        )

    def _has_conflicting_target_release_context(self, shot: Dict) -> bool:
        local_role = str(shot.get('local_highlight_role') or 'none')
        local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        owner_confidence = float(shot.get('owner_confidence') or 0.0)
        local_owner_confidence = float(shot.get('local_owner_confidence') or 0.0)

        return (
            (
                local_role == 'score'
                and local_confidence >= self.LOCAL_REVIEW_POSSIBLE_THRESHOLD
            )
            or (
                shot.get('owner') == 'target'
                and owner_confidence >= 0.52
            )
            or local_owner_confidence >= 0.52
        )

    def _has_local_assist_review_evidence(self, shot: Dict) -> bool:
        if str(shot.get('local_highlight_role') or 'none') != 'assist':
            return False

        local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        has_involvement_window = (
            shot.get('local_involvement_start_frame') is not None
            and shot.get('local_involvement_end_frame') is not None
        )

        return (
            has_involvement_window
            and local_confidence >= self.LOCAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD
        )

    def _has_partial_global_assist_evidence(self, shot: Dict) -> bool:
        attribution_role = str(
            shot.get('attribution_highlight_role')
            or shot.get('highlight_role')
            or 'none'
        )
        if attribution_role == 'score':
            return False

        if self._has_conflicting_target_release_context(shot):
            return False

        global_confidence = float(
            shot.get('attribution_highlight_confidence')
            or shot.get('highlight_confidence')
            or 0.0
        )
        target_visible = bool(shot.get('target_visible'))
        has_involvement_window = (
            shot.get('involvement_start_frame') is not None
            and shot.get('involvement_end_frame') is not None
        )

        return (
            target_visible
            and has_involvement_window
            and global_confidence >= self.GLOBAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD
        )

    def _has_strong_target_score_evidence(self, shot: Dict) -> bool:
        role = str(shot.get('highlight_role') or 'none')
        if role == 'score' or shot.get('owner') == 'target':
            return True

        owner_confidence = float(shot.get('owner_confidence') or 0.0)
        if owner_confidence >= 0.58:
            return True

        local_role = str(shot.get('local_highlight_role') or 'none')
        local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        if local_role == 'score' and local_confidence >= self.LOCAL_REVIEW_POSSIBLE_THRESHOLD:
            return True

        local_owner_confidence = float(shot.get('local_owner_confidence') or 0.0)
        return local_owner_confidence >= self.BACKFILLED_MADE_SCORE_THRESHOLD

    def _resolve_confirmable_assist_confidence(self, shot: Dict) -> Optional[float]:
        if not bool(shot.get('made')):
            return None

        local_role = str(shot.get('local_highlight_role') or 'none')
        local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        if local_role == 'assist' and local_confidence >= self.LOCAL_ASSIST_CONFIRM_THRESHOLD:
            return round(local_confidence, 3)

        if self._has_strong_target_score_evidence(shot):
            return None

        has_local_assist_review = self._has_local_assist_review_evidence(shot)
        has_partial_local_assist = self._has_partial_local_assist_evidence(shot)
        has_partial_global_assist = self._has_partial_global_assist_evidence(shot)
        if not has_local_assist_review and not has_partial_local_assist and not has_partial_global_assist:
            return None

        global_confidence = float(
            shot.get('attribution_highlight_confidence')
            or shot.get('highlight_confidence')
            or 0.0
        )

        if has_local_assist_review and has_partial_global_assist:
            promoted_confidence = min(
                1.0,
                max(local_confidence, global_confidence) + self.DUAL_SIGNAL_ASSIST_PROMOTION_BOOST,
            )
            if promoted_confidence >= self.DUAL_SIGNAL_ASSIST_PROMOTION_THRESHOLD:
                return round(promoted_confidence, 3)

        promoted_confidence = max(
            local_confidence if has_partial_local_assist else 0.0,
            global_confidence if has_partial_global_assist else 0.0,
        )
        if has_partial_local_assist and has_partial_global_assist:
            promoted_confidence = min(
                1.0,
                promoted_confidence + self.DUAL_SIGNAL_ASSIST_PROMOTION_BOOST,
            )
            if promoted_confidence >= self.DUAL_SIGNAL_ASSIST_PROMOTION_THRESHOLD:
                return round(promoted_confidence, 3)

        if promoted_confidence < self.PARTIAL_ASSIST_PROMOTION_THRESHOLD:
            return None

        return round(promoted_confidence, 3)

    def _score_target_review_candidate(self, shot: Dict) -> Tuple[float, str]:
        owner = shot.get('owner')
        owner_confidence = float(shot.get('owner_confidence') or 0.0)
        target_visible = bool(shot.get('target_visible'))
        local_target_visible = bool(shot.get('local_target_visible'))
        local_owner_confidence = float(shot.get('local_owner_confidence') or 0.0)
        local_role = str(shot.get('local_highlight_role') or 'none')
        local_highlight_confidence = float(shot.get('local_highlight_confidence') or 0.0)
        attribution_role = str(
            shot.get('attribution_highlight_role')
            or shot.get('highlight_role')
            or 'none'
        )
        attribution_highlight_confidence = float(
            shot.get('attribution_highlight_confidence')
            or shot.get('highlight_confidence')
            or 0.0
        )
        score_event_detected = bool(shot.get('score_event_detected'))
        has_partial_local_assist = self._has_partial_local_assist_evidence(shot)
        has_partial_global_assist = self._has_partial_global_assist_evidence(shot)

        candidate_score = 0.0
        if owner == 'target':
            candidate_score += 0.24 + owner_confidence * 0.28
        else:
            candidate_score += owner_confidence * 0.12

        candidate_score += local_owner_confidence * 0.18

        if local_role == 'score':
            candidate_score += 0.18 + local_highlight_confidence * 0.22
        elif local_role == 'assist':
            candidate_score += 0.08 + local_highlight_confidence * 0.16
        elif has_partial_local_assist:
            candidate_score += 0.12 + local_highlight_confidence * 0.16

        if has_partial_global_assist:
            candidate_score += 0.26 + attribution_highlight_confidence * 0.46
            if attribution_role == 'assist':
                candidate_score += 0.06

        if target_visible:
            candidate_score += 0.08
        if local_target_visible:
            candidate_score += 0.10
        if score_event_detected:
            candidate_score += 0.05
        if shot.get('involvement_start_frame') is not None:
            candidate_score += 0.04
        if shot.get('local_involvement_start_frame') is not None:
            candidate_score += 0.05

        reason = 'attempt_target_context'
        if local_role == 'score' and local_highlight_confidence >= 0.52:
            reason = 'attempt_local_score_window'
        elif (
            local_role == 'assist'
            and local_highlight_confidence >= self.LOCAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD
        ):
            reason = 'local_assist'
        elif has_partial_local_assist:
            reason = 'local_assist_window'
        elif has_partial_global_assist:
            reason = 'global_assist_window'
        elif owner == 'target' and owner_confidence >= 0.52:
            reason = 'attempt_target_release'
        elif local_target_visible and local_owner_confidence >= 0.40:
            reason = 'attempt_local_target_visible'
        elif target_visible:
            reason = 'attempt_target_visible'

        return round(min(candidate_score, 1.0), 3), reason

    def _resolve_target_review_threshold(self, tracking_summary: Dict) -> float:
        coverage = float(tracking_summary.get('coverage') or 0.0)
        return (
            self.ATTEMPT_REVIEW_PRIMARY_THRESHOLD
            if coverage >= 0.55
            else self.ATTEMPT_REVIEW_FALLBACK_THRESHOLD
        )

    @classmethod
    def _is_high_signal_review_reason(cls, reason: str) -> bool:
        return reason in cls.HIGH_SIGNAL_REVIEW_REASONS

    @classmethod
    def _is_high_signal_mixed_possible_reason(cls, reason: str) -> bool:
        return reason in cls.HIGH_SIGNAL_MIXED_POSSIBLE_REASONS

    def _select_target_review_candidates(
        self,
        all_shots: List[Dict],
        target_player_box: Optional[Dict],
        tracking_summary: Dict,
    ) -> List[Dict]:
        if not target_player_box:
            return []

        threshold = self._resolve_target_review_threshold(tracking_summary)
        review_candidates: List[Dict] = []

        for shot in all_shots:
            if shot.get('made'):
                continue

            if not (
                shot.get('owner') == 'target'
                or shot.get('target_visible')
                or shot.get('local_target_visible')
            ):
                continue

            candidate_score, reason = self._score_target_review_candidate(shot)
            if candidate_score < threshold:
                continue
            if not self._is_high_signal_review_reason(reason):
                continue

            review_candidates.append(
                self._build_attempt_review_candidate(shot, reason, candidate_score)
            )

        review_candidates.sort(
            key=lambda shot: (
                float(shot.get('highlight_confidence') or 0.0),
                float(shot.get('local_owner_confidence') or 0.0),
                float(shot.get('owner_confidence') or 0.0),
                float(shot.get('timestamp') or 0.0),
            ),
            reverse=True,
        )

        return review_candidates

    def _select_target_attempt_fallbacks(
        self,
        all_shots: List[Dict],
        target_player_box: Optional[Dict],
        minimum_confidence: float = 0.0,
    ) -> List[Dict]:
        if not target_player_box:
            return []

        fallback_candidates: List[Dict] = []
        for shot in all_shots:
            if not (
                shot.get('owner') == 'target'
                or shot.get('target_visible')
                or shot.get('local_target_visible')
                or float(shot.get('owner_confidence') or 0.0) >= 0.25
                or float(shot.get('local_owner_confidence') or 0.0) >= 0.25
                or shot.get('involvement_start_frame') is not None
                or shot.get('local_involvement_start_frame') is not None
            ):
                continue

            candidate_score, reason = self._score_target_review_candidate(shot)
            if candidate_score < minimum_confidence:
                continue
            fallback_candidates.append(
                self._build_target_attempt_fallback_candidate(shot, reason, candidate_score)
            )

        fallback_candidates.sort(
            key=lambda shot: (
                float(shot.get('highlight_confidence') or 0.0),
                float(shot.get('timestamp') or 0.0),
            ),
            reverse=True,
        )
        return fallback_candidates

    @staticmethod
    def _sort_output_shots(shots: List[Dict]) -> List[Dict]:
        return sorted(
            shots,
            key=lambda shot: (
                float(shot.get('timestamp') or 0.0),
                int(shot.get('frame') or 0),
            ),
        )

    def _is_duplicate_attempt(
        self,
        last_release_frame: int,
        last_down_frame: int,
        current_release_frame: int,
        current_down_frame: int,
    ) -> bool:
        if last_release_frame < 0 or last_down_frame < 0:
            return False

        return (
            abs(current_release_frame - last_release_frame) <= self.ATTEMPT_RELEASE_DEDUP_FRAMES
            and abs(current_down_frame - last_down_frame) <= self.ATTEMPT_DOWN_DEDUP_FRAMES
        )

    def _shots_refer_to_same_attempt(self, existing_shot: Dict, candidate_shot: Dict) -> bool:
        existing_release_frame = int(existing_shot.get('release_frame') or -1)
        existing_down_frame = int(existing_shot.get('frame') or -1)
        candidate_release_frame = int(candidate_shot.get('release_frame') or -1)
        candidate_down_frame = int(candidate_shot.get('frame') or -1)

        if (
            existing_release_frame >= 0
            and existing_down_frame >= 0
            and candidate_release_frame >= 0
            and candidate_down_frame >= 0
        ):
            return self._is_duplicate_attempt(
                last_release_frame=existing_release_frame,
                last_down_frame=existing_down_frame,
                current_release_frame=candidate_release_frame,
                current_down_frame=candidate_down_frame,
            )

        if existing_down_frame < 0 or candidate_down_frame < 0:
            return False

        return abs(existing_down_frame - candidate_down_frame) <= self.ATTEMPT_DOWN_DEDUP_FRAMES

    def _merge_additional_candidates(
        self,
        selected_shots: List[Dict],
        candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        if not candidates:
            return self._sort_output_shots(selected_shots), []

        merged_shots = list(selected_shots)
        merged_candidates: List[Dict] = []

        for candidate in candidates:
            is_duplicate = any(
                self._shots_refer_to_same_attempt(existing_shot, candidate)
                for existing_shot in merged_shots
            )
            if is_duplicate:
                continue
            merged_shots.append(candidate)
            merged_candidates.append(candidate)

        return self._sort_output_shots(merged_shots), merged_candidates

    def _merge_review_candidates(
        self,
        selected_shots: List[Dict],
        review_candidates: List[Dict],
    ) -> Tuple[List[Dict], List[Dict]]:
        if not review_candidates:
            return self._sort_output_shots(selected_shots), []

        filtered_review_candidates = [
            candidate
            for candidate in review_candidates
            if not self._is_redundant_preceding_attempt_review_candidate(selected_shots, candidate)
        ]

        return self._merge_additional_candidates(selected_shots, filtered_review_candidates)

    def _is_redundant_preceding_attempt_review_candidate(
        self,
        selected_shots: List[Dict],
        candidate: Dict,
    ) -> bool:
        candidate_source = str(candidate.get('candidate_source') or '')
        candidate_reason = str(candidate.get('candidate_reason') or '')
        if candidate_source != 'attempt_review' or not candidate_reason.startswith('attempt_'):
            return False

        candidate_timestamp = float(candidate.get('timestamp') or 0.0)
        for existing_shot in selected_shots:
            role = str(existing_shot.get('highlight_role') or '')
            if role not in {'score', 'assist'}:
                continue

            existing_timestamp = float(existing_shot.get('timestamp') or 0.0)
            if candidate_timestamp >= existing_timestamp:
                continue

            if (
                existing_timestamp - candidate_timestamp
                > self.ATTEMPT_REVIEW_REDUNDANT_PRECEDING_CONFIRM_GAP_SECONDS
            ):
                continue

            return True

        return False

    @staticmethod
    def _build_processing_diagnostics(
        total_attempts: int,
        total_makes: int,
        selected_shots: List[Dict],
        review_candidates: List[Dict],
        tracking_summary: Dict,
        target_player_box: Optional[Dict],
        selection_summary: Dict,
    ) -> Dict:
        coverage = float(tracking_summary.get('coverage') or 0.0)
        selected_clip_count = len(selected_shots)
        review_candidate_count = len(review_candidates)
        confirmed_count = int(selection_summary.get('confirmed', 0) or 0)
        possible_count = int(selection_summary.get('possible', 0) or 0)

        diagnostics = {
            'outcome': 'confirmed_highlights',
            'summary': '已导出与目标球员相关的片段。',
            'reasons': [],
            'recommendedActions': [],
            'counts': {
                'attempts': total_attempts,
                'madeShots': total_makes,
                'selectedClips': selected_clip_count,
                'reviewCandidates': review_candidate_count,
                'possibleHighlights': possible_count,
            },
            'trackingCoverage': round(coverage, 3),
        }

        if not target_player_box:
            if total_makes == 0:
                diagnostics['outcome'] = 'no_makes_detected'
                diagnostics['summary'] = '当前没有检测到可导出的进球片段。'
                diagnostics['reasons'] = ['视频里未识别出进球事件']
                diagnostics['recommendedActions'] = ['先确认样例视频里是否确实包含清晰进球']
            return diagnostics

        if selected_clip_count > 0 and review_candidate_count > 0 and confirmed_count == 0:
            diagnostics['outcome'] = 'review_candidates'
            diagnostics['summary'] = (
                f'当前没有确认到目标球员进球或助攻，已额外导出 {review_candidate_count} 个系统补充回合，避免直接返回空结果。'
            )
            diagnostics['reasons'] = ['进球规则没有确认命中，但目标球员相关出手特征存在']
            if coverage < 0.55:
                diagnostics['reasons'].append('目标跟踪覆盖率偏低，归因置信度不足')
            diagnostics['recommendedActions'] = [
                '先快速检查系统补充片段，确认是否存在漏检进球',
                '尽量把框选起点放在目标球员更清晰、身体完整的一帧',
            ]
            return diagnostics

        if selected_clip_count > 0 and review_candidate_count > 0:
            diagnostics['outcome'] = 'confirmed_with_review_candidates'
            diagnostics['summary'] = (
                f'已导出 {confirmed_count} 个已确认片段，并额外保留 {review_candidate_count} 个系统补充回合。建议先验收已确认片段，如怀疑漏剪，再检查系统补充片段。'
            )
            diagnostics['reasons'] = ['部分目标球员相关回合仍未通过最终进球确认，暂作为系统补充片段保留']
            diagnostics['recommendedActions'] = ['先验收已确认片段，如怀疑漏剪，再检查系统补充片段。']
            return diagnostics

        if selected_clip_count > 0 and confirmed_count == 0 and possible_count > 0:
            diagnostics['outcome'] = 'target_attempt_fallback'
            diagnostics['summary'] = (
                f'当前没有确认到目标球员进球或助攻，已先导出 {possible_count} 个目标相关片段，避免直接返回空结果。'
            )
            diagnostics['reasons'] = ['当前只保留了目标球员相关的系统补充回合，用于优先减少漏剪']
            if coverage < 0.55:
                diagnostics['reasons'].append('目标跟踪覆盖率偏低，归因还不够稳定')
            diagnostics['recommendedActions'] = [
                '先快速检查这些系统补充片段，确认是否包含漏检进球或助攻',
                '尽量把框选起点放在目标球员更清晰、身体完整的一帧',
            ]
            return diagnostics

        if selected_clip_count > 0:
            diagnostics['summary'] = (
                '已导出与目标球员相关的进球、助攻或系统补充片段。'
                if possible_count > 0
                else '已导出与目标球员相关的进球和助攻片段。'
            )
            if possible_count > 0:
                diagnostics['reasons'] = ['部分片段属于系统补充候选，用于减少漏剪']
            if coverage < 0.45:
                diagnostics['recommendedActions'] = ['可重新选择更清晰的出镜帧，提升目标锁定稳定性']
            return diagnostics

        if total_makes > 0:
            diagnostics['outcome'] = 'global_makes_without_target'
            diagnostics['summary'] = '检测到了全场进球，但没有稳定归因到目标球员。'
            diagnostics['reasons'] = ['目标球员与进球回合之间的关联证据不足']
            diagnostics['recommendedActions'] = [
                '重新选择更早且更清晰的目标球员起始帧后重跑',
                '优先选择目标球员控球或准备出手前的画面作为框选起点',
            ]
            return diagnostics

        if total_attempts > 0:
            diagnostics['outcome'] = 'no_makes_detected'
            diagnostics['summary'] = '检测到了投篮回合，但当前规则没有确认进球。'
            diagnostics['reasons'] = ['篮筐穿越或进球确认规则没有通过']
            diagnostics['recommendedActions'] = [
                '优先查看标注视频，确认篮球和篮筐检测是否稳定',
                '如结果明显漏剪，优先重新选择更早且更清晰的目标球员起始帧后重跑',
            ]
            return diagnostics

        diagnostics['outcome'] = 'no_attempts_detected'
        diagnostics['summary'] = '当前没有检测到可用的投篮回合。'
        diagnostics['reasons'] = ['篮球或篮筐检测不足以形成投篮事件']
        diagnostics['recommendedActions'] = [
            '先确认视频机位里篮筐和篮球是否长期可见',
            '如遮挡较多或机位较远，优先重新选择目标球员更早且更清晰的起始帧后重跑',
        ]
        return diagnostics

    @staticmethod
    def _count_highlight_roles(shots: List[Dict]) -> Dict[str, int]:
        counts = {
            'score': 0,
            'assist': 0,
            'possible': 0,
        }
        for shot in shots:
            role = str(shot.get('highlight_role') or 'none')
            if role in counts:
                counts[role] += 1
        return counts

    def _build_pipeline_summary(
        self,
        all_shots: List[Dict],
        made_shots: List[Dict],
        related_made_shots: List[Dict],
        selected_shots: List[Dict],
        review_candidates: List[Dict],
        clips: List[Dict],
        tracking_summary: Dict,
        target_player_box: Optional[Dict],
        selection_summary: Dict,
        before_seconds: float,
        after_seconds: float,
    ) -> Dict:
        confirmed_role_counts = self._count_highlight_roles(related_made_shots)
        exported_role_counts = self._count_highlight_roles(selected_shots)
        tracking_start_time = (
            round(float(target_player_box.get('selectionTime') or 0.0), 3)
            if target_player_box
            else None
        )

        return {
            'scan': {
                'mode': 'full_video_single_pass',
                'fullVideoScanned': True,
                'trackerEnabled': bool(target_player_box),
                'trackingStartTime': tracking_start_time,
                'trackingStartFrame': (
                    int(target_player_box.get('selectionFrame'))
                    if target_player_box and isinstance(target_player_box.get('selectionFrame'), (int, float))
                    else tracking_summary.get('startFrame')
                ),
                'totalShotEvents': len(all_shots),
                'madeShotEvents': len(made_shots),
                'targetVisibleEvents': len([
                    shot for shot in all_shots
                    if shot.get('target_visible') or shot.get('local_target_visible')
                ]),
            },
            'attribution': {
                'selectionMode': selection_summary.get('mode'),
                'confirmedHighlights': int(selection_summary.get('confirmed', 0) or 0),
                'possibleHighlights': int(selection_summary.get('possible', 0) or 0),
                'confirmedScores': confirmed_role_counts['score'],
                'confirmedAssists': confirmed_role_counts['assist'],
                'reviewCandidates': len(review_candidates),
                'trackingCoverage': round(float(tracking_summary.get('coverage') or 0.0), 3),
            },
            'export': {
                'selectedClipCount': len(clips),
                'selectedHighlights': len(selected_shots),
                'clipWindowBeforeSeconds': round(float(before_seconds), 3),
                'clipWindowAfterSeconds': round(float(after_seconds), 3),
                'scoreClips': exported_role_counts['score'],
                'assistClips': exported_role_counts['assist'],
                'possibleClips': exported_role_counts['possible'],
            },
        }

    @staticmethod
    def _selection_mode_rank(mode: Optional[str]) -> int:
        ranks = {
            'mixed': 5,
            'mixed_with_review_candidates': 4,
            'review_candidates_fallback': 3,
            'target_attempt_fallback': 2,
            'no_target_highlights': 1,
        }
        return ranks.get(str(mode or ''), 0)

    def _score_detection_output(self, output: Dict) -> Tuple[int, int, int, int, int, int]:
        selection_summary = output.get('selection_summary', {})
        stats = output.get('stats', {})
        tracking = output.get('tracking', {})
        confirmed = int(selection_summary.get('confirmed', 0) or 0)
        target_highlights = int(stats.get('target_highlights', 0) or 0)
        related_highlights = int(stats.get('related_highlights', 0) or 0)
        coverage_score = int(round(float(tracking.get('coverage') or 0.0) * 1000))
        possible_highlights = int(stats.get('possible_highlights', 0) or 0)
        return (
            confirmed,
            target_highlights,
            related_highlights,
            coverage_score,
            self._selection_mode_rank(selection_summary.get('mode')),
            possible_highlights,
        )

    def _should_auto_retry_target_detection(self, output: Dict, target_player_box: Optional[Dict]) -> bool:
        if not target_player_box:
            return False

        diagnostics = output.get('diagnostics', {})
        outcome = str(diagnostics.get('outcome') or '')
        coverage = float(output.get('tracking', {}).get('coverage') or 0.0)
        possible_highlights = int(output.get('stats', {}).get('possible_highlights', 0) or 0)
        confirmed = int(output.get('selection_summary', {}).get('confirmed', 0) or 0)

        if possible_highlights > 0:
            return True

        if coverage < self.AUTO_RETRY_MIN_TRACKING_COVERAGE:
            return True

        return confirmed == 0 and outcome in self.AUTO_RETRY_TRIGGER_OUTCOMES

    def _should_stop_auto_retry_after_stable_improvement(self, output: Dict) -> bool:
        selection_summary = output.get('selection_summary', {})
        stats = output.get('stats', {})
        tracking = output.get('tracking', {})
        diagnostics = output.get('diagnostics', {})

        confirmed = int(selection_summary.get('confirmed', 0) or 0)
        possible = int(selection_summary.get('possible', 0) or 0)
        target_highlights = int(stats.get('target_highlights', 0) or 0)
        related_highlights = int(stats.get('related_highlights', 0) or 0)
        review_candidate_highlights = int(stats.get('review_candidate_highlights', 0) or 0)
        coverage = float(tracking.get('coverage') or 0.0)
        outcome = str(diagnostics.get('outcome') or '')

        return (
            confirmed > 0
            and possible == 0
            and review_candidate_highlights == 0
            and target_highlights == confirmed
            and related_highlights == target_highlights
            and coverage >= self.AUTO_RETRY_MIN_TRACKING_COVERAGE
            and outcome == 'confirmed_highlights'
        )

    def _build_detection_output(
        self,
        *,
        all_shots: List[Dict],
        tracking_summary: Dict,
        video_path: str,
        before_seconds: float,
        after_seconds: float,
        target_player_box: Optional[Dict],
        annotate: bool,
        annotated_output_path: Optional[str],
    ) -> Dict:
        effective_annotated_output_path = None
        if (
            annotate
            and annotated_output_path
            and os.path.exists(annotated_output_path)
            and os.path.getsize(annotated_output_path) > 0
        ):
            effective_annotated_output_path = annotated_output_path

        selection_result = self._select_target_highlights(
            all_shots,
            tracking_summary,
            target_player_box,
        )
        made_shots = selection_result['made_shots']
        related_made_shots = selection_result['related_made_shots']
        merged_review_candidates = selection_result['merged_review_candidates']
        review_candidates = selection_result['review_candidates']
        selected_shots = selection_result['selected_shots']
        selection_summary = selection_result['selection_summary']

        duration = self._get_video_duration(video_path)
        clips = self._build_clip_windows(
            selected_shots,
            duration,
            before_seconds,
            after_seconds,
        )

        total_attempts = len(all_shots)
        total_makes = len(made_shots)
        accuracy = (total_makes / total_attempts * 100) if total_attempts > 0 else 0
        target_attempts = len([shot for shot in all_shots if shot.get('owner') == 'target'])
        target_makes = len([shot for shot in made_shots if shot.get('owner') == 'target'])
        target_scores = len([shot for shot in made_shots if shot.get('highlight_role') == 'score'])
        target_assists = len([shot for shot in made_shots if shot.get('highlight_role') == 'assist'])
        if target_player_box:
            target_scores = len([
                shot for shot in related_made_shots
                if shot.get('highlight_role') == 'score'
            ])
            target_assists = len([
                shot for shot in related_made_shots
                if shot.get('highlight_role') == 'assist'
            ])
            target_makes = max(target_makes, target_scores)
            target_attempts = max(target_attempts, target_scores)
        possible_highlights = len([shot for shot in selected_shots if shot.get('highlight_role') == 'possible'])
        related_highlights = len(selected_shots)
        target_highlights = target_scores + target_assists
        diagnostics = self._build_processing_diagnostics(
            total_attempts=total_attempts,
            total_makes=total_makes,
            selected_shots=selected_shots,
            review_candidates=merged_review_candidates,
            tracking_summary=tracking_summary,
            target_player_box=target_player_box,
            selection_summary=selection_summary,
        )
        pipeline_summary = self._build_pipeline_summary(
            all_shots=all_shots,
            made_shots=made_shots,
            related_made_shots=related_made_shots,
            selected_shots=selected_shots,
            review_candidates=merged_review_candidates,
            clips=clips,
            tracking_summary=tracking_summary,
            target_player_box=target_player_box,
            selection_summary=selection_summary,
            before_seconds=before_seconds,
            after_seconds=after_seconds,
        )

        return {
            'shots': all_shots,
            'made_shots': made_shots,
            'selected_made_shots': related_made_shots,
            'selected_shots': selected_shots,
            'review_candidates': merged_review_candidates,
            'clips': clips,
            'target_player_box': target_player_box,
            'stats': {
                'total_attempts': total_attempts,
                'total_makes': total_makes,
                'accuracy': round(accuracy, 2),
                'target_attempts': target_attempts,
                'target_makes': target_makes,
                'target_scores': target_scores,
                'target_assists': target_assists,
                'target_highlights': target_highlights,
                'possible_highlights': possible_highlights,
                'related_highlights': related_highlights,
                'review_candidate_highlights': len(merged_review_candidates),
            },
            'tracking': tracking_summary,
            'annotated_video': effective_annotated_output_path if annotate else None,
            'selection_summary': selection_summary,
            'diagnostics': diagnostics,
            'pipeline': pipeline_summary,
        }

    def _select_target_highlights(
        self,
        all_shots: List[Dict],
        tracking_summary: Dict,
        target_player_box: Optional[Dict],
    ) -> Dict:
        made_shots = [shot for shot in all_shots if shot['made']]
        related_made_shots, selection_summary = self._select_related_made_shots(
            made_shots,
            target_player_box,
            tracking_summary,
        )
        review_candidates = self._select_target_review_candidates(
            all_shots,
            target_player_box,
            tracking_summary,
        )
        selected_shots, merged_review_candidates = self._merge_review_candidates(
            related_made_shots,
            review_candidates,
        )

        if not related_made_shots and merged_review_candidates:
            selection_summary = {
                'mode': 'review_candidates_fallback',
                'confirmed': 0,
                'possible': len(merged_review_candidates),
            }
        elif merged_review_candidates:
            selection_summary = {
                'mode': 'mixed_with_review_candidates',
                'confirmed': int(selection_summary.get('confirmed', 0) or 0),
                'possible': int(selection_summary.get('possible', 0) or 0) + len(merged_review_candidates),
            }
        else:
            selected_shots = self._sort_output_shots(related_made_shots)

        if not selected_shots:
            fallback_attempts = self._select_target_attempt_fallbacks(
                all_shots,
                target_player_box,
            )
            if fallback_attempts:
                selected_shots = self._sort_output_shots(fallback_attempts)
                selection_summary = {
                    'mode': 'target_attempt_fallback',
                    'confirmed': 0,
                    'possible': len(fallback_attempts),
                }

        return {
            'made_shots': made_shots,
            'related_made_shots': related_made_shots,
            'review_candidates': review_candidates,
            'merged_review_candidates': merged_review_candidates,
            'selected_shots': selected_shots,
            'selection_summary': selection_summary,
        }

    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0:
            raise ValueError("无法读取视频帧率")

        return frame_count / fps

    @staticmethod
    def _resolve_tracking_start(
        target_player_box: Optional[Dict],
        fps: float,
        total_frames: int,
    ) -> Tuple[int, float]:
        if not target_player_box:
            return 0, 0.0

        start_time = max(float(target_player_box.get('selectionTime', 0.0) or 0.0), 0.0)
        if fps > 0:
            start_frame = int(round(start_time * fps))
        elif isinstance(target_player_box.get('selectionFrame'), (int, float)):
            start_frame = int(round(target_player_box.get('selectionFrame')))
        else:
            start_frame = 0

        if total_frames > 0:
            start_frame = min(max(start_frame, 0), total_frames - 1)
            if fps > 0:
                start_time = start_frame / fps

        return start_frame, start_time

    @staticmethod
    def _build_target_player_box_from_bbox(
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        selection_time: float,
        selection_frame: int,
    ) -> Dict:
        x, y, width, height = bbox
        return {
            'x': int(round(x)),
            'y': int(round(y)),
            'width': int(round(width)),
            'height': int(round(height)),
            'frameWidth': int(round(frame_width)),
            'frameHeight': int(round(frame_height)),
            'selectionTime': round(float(selection_time), 3),
            'selectionFrame': int(round(selection_frame)),
        }

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
    def _build_clip_windows(
        cls,
        selected_shots: List[Dict],
        duration: float,
        before_seconds: float,
        after_seconds: float,
    ) -> List[Dict]:
        clips = []
        for shot in selected_shots:
            start_time = max(0, shot['timestamp'] - before_seconds)
            involvement_start = shot.get('involvement_start_timestamp')
            if isinstance(involvement_start, (int, float)):
                involvement_lead_seconds = cls._resolve_involvement_lead_seconds(shot)
                start_time = min(start_time, max(0, float(involvement_start) - involvement_lead_seconds))
            end_time = min(duration, shot['timestamp'] + after_seconds)
            clips.append({
                'start': start_time,
                'end': end_time,
                'shot_frame': shot['frame'],
                'shot_timestamp': shot['timestamp'],
                'highlight_role': shot.get('highlight_role', 'score'),
                'candidate_reason': shot.get('candidate_reason'),
                'candidate_source': shot.get('candidate_source'),
                'highlight_confidence': shot.get('highlight_confidence'),
            })
        return clips

    @staticmethod
    def _inherit_local_involvement(shot: Dict) -> Dict:
        normalized_shot = dict(shot)
        if normalized_shot.get('involvement_start_frame') is None and normalized_shot.get('local_involvement_start_frame') is not None:
            normalized_shot['involvement_start_frame'] = normalized_shot.get('local_involvement_start_frame')
            normalized_shot['involvement_end_frame'] = normalized_shot.get('local_involvement_end_frame')
            normalized_shot['involvement_start_timestamp'] = normalized_shot.get('local_involvement_start_timestamp')
            normalized_shot['involvement_end_timestamp'] = normalized_shot.get('local_involvement_end_timestamp')
        elif (
            normalized_shot.get('local_involvement_start_timestamp') is not None
            and normalized_shot.get('involvement_start_timestamp') is not None
            and float(normalized_shot['local_involvement_start_timestamp']) < float(normalized_shot['involvement_start_timestamp'])
        ):
            normalized_shot['involvement_start_frame'] = normalized_shot.get('local_involvement_start_frame')
            normalized_shot['involvement_end_frame'] = normalized_shot.get('local_involvement_end_frame')
            normalized_shot['involvement_start_timestamp'] = normalized_shot.get('local_involvement_start_timestamp')
            normalized_shot['involvement_end_timestamp'] = normalized_shot.get('local_involvement_end_timestamp')

        return normalized_shot

    def _promote_local_review(self, shot: Dict, role: str, confidence: float) -> Dict:
        promoted_shot = self._inherit_local_involvement(shot)
        promoted_shot['highlight_role'] = role
        promoted_shot['highlight_confidence'] = round(
            max(float(promoted_shot.get('highlight_confidence') or 0.0), float(confidence)),
            3,
        )
        promoted_shot['target_visible'] = bool(
            promoted_shot.get('target_visible') or promoted_shot.get('local_target_visible')
        )
        promoted_shot['owner_confidence'] = round(
            max(
                float(promoted_shot.get('owner_confidence') or 0.0),
                float(promoted_shot.get('local_owner_confidence') or 0.0),
            ),
            3,
        )
        if role == 'score':
            promoted_shot['owner'] = 'target'
        return promoted_shot

    def _prime_target_reference_window(
        self,
        cap,
        tracker: TargetPlayerTracker,
        center_frame: int,
        fps: float,
        total_frames: int,
    ) -> int:
        if fps <= 0 or total_frames <= 0:
            return 0

        current_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        sample_frames: List[int] = []
        attempted_frames = set()

        for offset_seconds in self.REFERENCE_PRIME_OFFSETS_SECONDS:
            offset_frames = max(int(round(offset_seconds * fps)), 1)
            for direction in (-1, 1):
                sample_frame = center_frame + direction * offset_frames
                if sample_frame < 0 or sample_frame >= total_frames or sample_frame == center_frame:
                    continue
                if sample_frame not in sample_frames:
                    sample_frames.append(sample_frame)

        sample_frames.sort(key=lambda frame_index: (abs(frame_index - center_frame), frame_index))

        successful_samples = 0
        anchor_bbox = tracker.initial_bbox or tracker.last_bbox or tracker.current_bbox

        def attempt_sample_frame(sample_frame: int) -> bool:
            nonlocal successful_samples, anchor_bbox
            if sample_frame in attempted_frames:
                return False
            attempted_frames.add(sample_frame)

            cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
            sample_ok, sample_frame_image = cap.read()
            if not sample_ok:
                return False

            added_sample = tracker.add_reference_sample(
                sample_frame_image,
                sample_frame,
                anchor_bbox=anchor_bbox,
            )
            if added_sample is None:
                return False

            successful_samples += 1
            anchor_bbox = added_sample.get('bbox') or anchor_bbox
            return True

        try:
            for sample_frame in sample_frames:
                attempt_sample_frame(sample_frame)

            if successful_samples < self.REFERENCE_PRIME_MIN_SUCCESSFUL_SAMPLES:
                dense_step_frames = max(int(round(self.REFERENCE_PRIME_DENSE_STEP_SECONDS * fps)), 1)
                dense_window_frames = max(
                    int(round(self.REFERENCE_PRIME_DENSE_WINDOW_SECONDS * fps)),
                    dense_step_frames,
                )
                dense_frames: List[int] = []

                for offset_frames in range(dense_step_frames, dense_window_frames + 1, dense_step_frames):
                    for direction in (-1, 1):
                        sample_frame = center_frame + direction * offset_frames
                        if sample_frame < 0 or sample_frame >= total_frames or sample_frame == center_frame:
                            continue
                        if sample_frame not in dense_frames:
                            dense_frames.append(sample_frame)

                dense_frames.sort(key=lambda frame_index: (abs(frame_index - center_frame), frame_index))

                for sample_frame in dense_frames:
                    attempt_sample_frame(sample_frame)
                    if successful_samples >= self.REFERENCE_PRIME_MIN_SUCCESSFUL_SAMPLES:
                        break
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_position)

        return successful_samples

    def _discover_retry_target_boxes(
        self,
        video_path: str,
        target_player_box: Optional[Dict],
    ) -> List[Dict]:
        if not target_player_box:
            return []

        cap = cv2.VideoCapture(video_path)
        is_opened = getattr(cap, 'isOpened', None)
        if not callable(is_opened) or not is_opened():
            return []

        if not all(hasattr(cap, method_name) for method_name in ('get', 'set', 'read', 'release')):
            return []

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or target_player_box.get('frameWidth') or 0)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or target_player_box.get('frameHeight') or 0)
            center_frame, center_time = self._resolve_tracking_start(
                target_player_box,
                fps,
                total_frames,
            )

            tracker = TargetPlayerTracker(
                target_player_box,
                start_frame=center_frame,
                start_time=center_time,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, center_frame)
            center_ok, center_image = cap.read()
            if not center_ok:
                return []

            tracker.initialize(center_image, center_frame)
            self._prime_target_reference_window(
                cap,
                tracker,
                center_frame,
                fps,
                total_frames,
            )

            sample_frames = self._build_retry_sample_frames(
                center_frame=center_frame,
                fps=fps,
                total_frames=total_frames,
            )

            discovered_candidates: List[Dict] = []
            seen_candidate_keys = set()
            anchor_bbox = tracker.initial_bbox or tracker.last_bbox or tracker.current_bbox

            for sample_frame in sample_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame)
                sample_ok, sample_frame_image = cap.read()
                if not sample_ok:
                    continue

                added_sample = tracker.add_reference_sample(
                    sample_frame_image,
                    sample_frame,
                    anchor_bbox=anchor_bbox,
                )
                if added_sample is None:
                    continue

                bbox = added_sample.get('bbox')
                if bbox is None:
                    continue

                selection_time = sample_frame / fps if fps > 0 else center_time
                candidate_box = self._build_target_player_box_from_bbox(
                    bbox=bbox,
                    frame_width=frame_width or sample_frame_image.shape[1],
                    frame_height=frame_height or sample_frame_image.shape[0],
                    selection_time=selection_time,
                    selection_frame=sample_frame,
                )
                candidate_key = (
                    candidate_box['selectionFrame'],
                    candidate_box['x'],
                    candidate_box['y'],
                    candidate_box['width'],
                    candidate_box['height'],
                )
                if candidate_key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(candidate_key)

                discovered_candidates.append({
                    'target_player_box': candidate_box,
                    'score': float(added_sample.get('score') or 0.0),
                })
                anchor_bbox = bbox

            discovered_candidates.sort(
                key=lambda candidate: self._rank_retry_candidate(
                    candidate=candidate,
                    center_frame=center_frame,
                    fps=fps,
                ),
                reverse=True,
            )
            return discovered_candidates[:self.AUTO_RETRY_MAX_CANDIDATES]
        finally:
            cap.release()

    def _select_prealigned_target_player_box_from_candidates(
        self,
        target_player_box: Optional[Dict],
        retry_candidates: List[Dict],
    ) -> Optional[Dict]:
        if not target_player_box:
            return target_player_box

        selection_time = float(target_player_box.get('selectionTime') or 0.0)
        if selection_time < self.PREALIGN_MIN_SELECTION_TIME_SECONDS:
            return target_player_box

        if not retry_candidates:
            return target_player_box

        for candidate in retry_candidates:
            candidate_box = candidate.get('target_player_box')
            if not isinstance(candidate_box, dict):
                continue

            candidate_time = float(candidate_box.get('selectionTime') or selection_time)
            if candidate_time >= selection_time:
                continue

            if selection_time - candidate_time < self.PREALIGN_MIN_EARLIER_SHIFT_SECONDS:
                continue

            if float(candidate.get('score') or 0.0) < self.PREALIGN_MIN_CANDIDATE_SCORE:
                continue

            return candidate_box

        return target_player_box

    def _select_prealigned_target_player_box(
        self,
        video_path: str,
        target_player_box: Optional[Dict],
    ) -> Optional[Dict]:
        retry_candidates = self._discover_retry_target_boxes(video_path, target_player_box)
        return self._select_prealigned_target_player_box_from_candidates(
            target_player_box,
            retry_candidates,
        )

    @classmethod
    def _build_retry_sample_frames(
        cls,
        *,
        center_frame: int,
        fps: float,
        total_frames: int,
    ) -> List[int]:
        sample_frames: List[int] = []
        seen_frames = set()

        def add_frame(frame_index: int) -> None:
            if frame_index < 0 or frame_index >= total_frames or frame_index == center_frame:
                return
            if frame_index in seen_frames:
                return
            seen_frames.add(frame_index)
            sample_frames.append(frame_index)

        if fps > 0:
            for offset_seconds in cls.REFERENCE_PRIME_OFFSETS_SECONDS:
                offset_frames = max(int(round(offset_seconds * fps)), 1)
                add_frame(center_frame - offset_frames)
                add_frame(center_frame + offset_frames)

            for offset_seconds in cls.AUTO_RETRY_BACKWARD_OFFSETS_SECONDS:
                offset_frames = max(int(round(offset_seconds * fps)), 1)
                add_frame(center_frame - offset_frames)

        sample_frames.sort(key=lambda frame_index: (abs(frame_index - center_frame), frame_index))
        return sample_frames

    @classmethod
    def _rank_retry_candidate(
        cls,
        *,
        candidate: Dict,
        center_frame: int,
        fps: float,
    ) -> Tuple[float, int, float]:
        raw_score = float(candidate.get('score') or 0.0)
        selection_frame = int(candidate.get('target_player_box', {}).get('selectionFrame') or center_frame)
        earlier_frame_gain = max(center_frame - selection_frame, 0)
        earlier_bonus = 0.0
        if fps > 0 and earlier_frame_gain > 0:
            earlier_bonus = min(
                (earlier_frame_gain / fps) * cls.AUTO_RETRY_EARLIER_FRAME_BONUS_PER_SECOND,
                cls.AUTO_RETRY_EARLIER_FRAME_BONUS_CAP,
            )

        return (
            round(raw_score + earlier_bonus, 4),
            earlier_frame_gain,
            -abs(selection_frame - center_frame),
        )

    def _select_related_made_shots(
        self,
        made_shots: List[Dict],
        target_player_box: Optional[Dict],
        tracking_summary: Dict,
    ) -> Tuple[List[Dict], Dict]:
        if not target_player_box:
            return made_shots, {
                'mode': 'all_made',
                'confirmed': len(made_shots),
                'possible': 0,
            }

        confirmed_shots: List[Dict] = []
        possible_shots: List[Dict] = []
        seen_frames = set()
        selection_time = float(target_player_box.get('selectionTime') or 0.0)

        for shot in made_shots:
            shot_key = (shot.get('frame'), shot.get('timestamp'))
            role = shot.get('highlight_role')
            local_role = shot.get('local_highlight_role')
            local_confidence = float(shot.get('local_highlight_confidence') or 0.0)
            local_owner_confidence = float(shot.get('local_owner_confidence') or 0.0)
            local_target_visible = bool(shot.get('local_target_visible'))
            shot_timestamp = float(shot.get('timestamp') or 0.0)
            is_backfilled_made_context = shot_timestamp + 0.25 < selection_time

            if role in {'score', 'assist'}:
                confirmed_shots.append(self._inherit_local_involvement(shot))
                seen_frames.add(shot_key)
                continue

            if (
                local_role == 'score'
                and local_confidence >= self.LOCAL_REVIEW_CONFIRM_THRESHOLD
            ):
                confirmed_shots.append(
                    self._promote_local_review(shot, role=str(local_role), confidence=local_confidence)
                )
                seen_frames.add(shot_key)
                continue

            confirmable_assist_confidence = self._resolve_confirmable_assist_confidence(shot)
            if confirmable_assist_confidence is not None:
                confirmed_shots.append(
                    self._promote_local_review(
                        shot,
                        role='assist',
                        confidence=confirmable_assist_confidence,
                    )
                )
                seen_frames.add(shot_key)
                continue

            if (
                is_backfilled_made_context
                and bool(shot.get('score_event_detected'))
                and local_role == 'none'
                and local_target_visible
                and local_owner_confidence >= self.BACKFILLED_MADE_SCORE_THRESHOLD
            ):
                confirmed_shots.append(
                    self._promote_local_review(
                        shot,
                        role='score',
                        confidence=local_owner_confidence,
                    )
                )
                seen_frames.add(shot_key)
                continue

            if (
                local_role == 'assist'
                and local_confidence >= self.LOCAL_ASSIST_REVIEW_POSSIBLE_THRESHOLD
                and shot.get('local_involvement_start_frame') is not None
                and shot.get('local_involvement_end_frame') is not None
            ):
                possible_shots.append(
                    self._build_possible_highlight(
                        self._promote_local_review(
                            shot,
                            role='assist',
                            confidence=local_confidence,
                        ),
                        reason='local_assist',
                    )
                )
                seen_frames.add(shot_key)
                continue

            if self._has_partial_local_assist_evidence(shot):
                possible_shots.append(
                    self._build_possible_highlight(
                        self._promote_local_review(
                            shot,
                            role='assist',
                            confidence=local_confidence,
                        ),
                        reason='local_assist_window',
                    )
                )
                seen_frames.add(shot_key)
                continue

            if self._has_partial_global_assist_evidence(shot):
                possible_shots.append(
                    self._build_possible_highlight(
                        self._inherit_local_involvement(shot),
                        reason='global_assist_window',
                    )
                )
                seen_frames.add(shot_key)
                continue

            if (
                shot.get('owner') == 'target'
                or shot.get('target_visible')
                or float(shot.get('owner_confidence') or 0.0) >= 0.35
                or (
                    local_role in {'score', 'assist'}
                    and local_confidence >= self.LOCAL_REVIEW_POSSIBLE_THRESHOLD
                )
                or local_owner_confidence >= self.LOCAL_REVIEW_VISIBLE_THRESHOLD
                or local_target_visible
            ):
                possible_reason = 'target_visible'
                possible_source_shot = self._inherit_local_involvement(shot)
                if local_role in {'score', 'assist'} and local_confidence >= self.LOCAL_REVIEW_POSSIBLE_THRESHOLD:
                    possible_reason = f'local_{local_role}'
                    possible_source_shot = self._promote_local_review(
                        possible_source_shot,
                        role=str(local_role),
                        confidence=local_confidence,
                    )
                elif local_owner_confidence >= self.LOCAL_REVIEW_VISIBLE_THRESHOLD or local_target_visible:
                    possible_reason = 'local_target_visible'

                possible_shots.append(
                    self._build_possible_highlight(possible_source_shot, reason=possible_reason)
                )
                seen_frames.add(shot_key)

        coverage = float(tracking_summary.get('coverage') or 0.0)

        if coverage < 0.35:
            for shot in made_shots:
                shot_key = (shot.get('frame'), shot.get('timestamp'))
                if shot_key in seen_frames:
                    continue
                if (
                    shot.get('target_visible')
                    or float(shot.get('owner_confidence') or 0.0) >= 0.15
                    or bool(shot.get('local_target_visible'))
                    or float(shot.get('local_owner_confidence') or 0.0) >= 0.15
                ):
                    possible_shots.append(
                        self._build_possible_highlight(
                            self._inherit_local_involvement(shot),
                            reason='low_tracking_coverage',
                        )
                    )
                    seen_frames.add(shot_key)

        if confirmed_shots:
            possible_shots = [
                shot for shot in possible_shots
                if self._is_high_signal_mixed_possible_reason(
                    str(shot.get('candidate_reason') or '')
                )
            ]

        related_shots = confirmed_shots + possible_shots
        if not related_shots and made_shots:
            return [], {
                'mode': 'no_target_highlights',
                'confirmed': 0,
                'possible': 0,
            }
        return related_shots, {
            'mode': 'mixed',
            'confirmed': len(confirmed_shots),
            'possible': len(possible_shots),
        }
    
    def detect_shots(
        self,
        video_path: str,
        progress_callback=None,
        annotate: bool = False,
        annotated_output_path: Optional[str] = None,
        target_player_box: Optional[Dict] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        检测视频中的所有进球
        
        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调函数 callback(current_frame, total_frames)
        
        Returns:
            (进球列表, 目标人物跟踪摘要)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息 - FPS: {fps}, 尺寸: {width}x{height}, 总帧数: {total_frames}")

        # 标注视频写出器
        writer = None
        effective_annotated_output_path = None
        if annotate:
            try:
                if not annotated_output_path:
                    base = os.path.splitext(os.path.basename(video_path))[0]
                    annotated_output_path = os.path.join(tempfile.gettempdir(), f"{base}_annotated.mp4")

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(annotated_output_path, fourcc, fps if fps > 0 else 30, (width, height))
                if not writer or not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    writer = cv2.VideoWriter(annotated_output_path, fourcc, fps if fps > 0 else 30, (width, height))

                if writer and writer.isOpened():
                    effective_annotated_output_path = annotated_output_path
                    print(f"标注视频输出: {annotated_output_path}")
                else:
                    print("⚠️ 无法初始化视频写出器，跳过标注输出")
                    writer = None
            except Exception as e:
                print(f"⚠️ 初始化标注视频写出器失败: {e}")
                writer = None
        
        # 初始化追踪变量
        ball_pos = []
        attribution_ball_pos = []
        recent_ball_frames: Dict[int, np.ndarray] = {}
        hoop_pos = []
        frame_count = 0
        target_tracker: Optional[TargetPlayerTracker] = None
        tracker_error: Optional[str] = None
        target_tracker_start_frame = 0
        target_tracker_start_time = 0.0
        primed_reference_samples = 0
        live_reference_samples = 0
        last_live_reference_frame = -self.LIVE_REFERENCE_REFRESH_INTERVAL_FRAMES
        
        # 投篮检测变量
        up = False
        down = False
        up_frame = 0
        down_frame = 0
        last_attempt_frame = -1
        last_release_frame = -1
        scoring_event = None
        
        # 结果存储
        shot_results = []
        makes = 0
        attempts = 0
        target_attempts = 0
        target_makes = 0

        if target_player_box:
            try:
                target_tracker_start_frame, target_tracker_start_time = self._resolve_tracking_start(
                    target_player_box,
                    fps,
                    total_frames,
                )

                target_tracker = TargetPlayerTracker(
                    target_player_box,
                    start_frame=target_tracker_start_frame,
                    start_time=target_tracker_start_time,
                )
                primed_position = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_tracker_start_frame)
                primed_ok, primed_frame = cap.read()
                cap.set(cv2.CAP_PROP_POS_FRAMES, primed_position)
                if primed_ok:
                    target_tracker.initialize(primed_frame, target_tracker_start_frame)
                    primed_reference_samples = self._prime_target_reference_window(
                        cap,
                        target_tracker,
                        target_tracker_start_frame,
                        fps,
                        total_frames,
                    )
                    last_live_reference_frame = target_tracker_start_frame
                    if primed_reference_samples > 0:
                        print(f"目标球员额外参考帧: {primed_reference_samples} 个")
                else:
                    print("⚠️ 无法读取目标人物选择帧，将只使用运行时跟踪初始化")
                if target_tracker_start_frame > 0:
                    print(
                        f"目标球员跟踪将在第 {target_tracker_start_frame} 帧 "
                        f"({target_tracker_start_time:.2f}s) 开始初始化"
                    )
            except Exception as error:
                tracker_error = str(error)
                target_tracker = None
                print(f"⚠️ 初始化目标人物跟踪失败: {tracker_error}")
        
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            tracker_record = None
            if target_tracker:
                if frame_count >= target_tracker.start_frame:
                    tracker_record = target_tracker.update(frame, frame_count)
                    if tracker_record and tracker_record.get('visible'):
                        tracker_status = str(tracker_record.get('status') or '')
                        tracker_confidence = float(tracker_record.get('confidence') or 0.0)
                        if (
                            tracker_status in self.LIVE_REFERENCE_REFRESH_STATUSES
                            and tracker_confidence >= self.LIVE_REFERENCE_REFRESH_MIN_CONFIDENCE
                            and frame_count - last_live_reference_frame >= self.LIVE_REFERENCE_REFRESH_INTERVAL_FRAMES
                        ):
                            refreshed_reference_sample = target_tracker.register_tracking_sample(
                                frame,
                                frame_count,
                                bbox=tracker_record.get('bbox'),
                            )
                            if refreshed_reference_sample is not None:
                                live_reference_samples += 1
                                last_live_reference_frame = frame_count

            # 运行YOLO检测
            results = self.model(frame, stream=True, device=self.device, verbose=False)

            # 当前帧的篮球边框集合（用于绘制红框）
            ball_boxes_in_frame: List[Tuple[int, int, int, int, float]] = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # 边界框
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    
                    # 置信度
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    
                    # 类别
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]
                    
                    center = (int(x1 + w / 2), int(y1 + h / 2))
                    
                    # 检测篮球
                    if (conf > self.confidence_threshold or 
                        (in_hoop_region(center, hoop_pos) and conf > 0.15)) and \
                        current_class == "Basketball":
                        detection = (center, frame_count, w, h, conf)
                        ball_pos.append(detection)
                        # 记录当前帧的篮球边框
                        ball_boxes_in_frame.append((x1, y1, x2, y2, conf))
                    
                    # 检测篮筐
                    if conf > 0.3 and current_class == "Basketball Hoop":
                        hoop_pos.append((center, frame_count, w, h, conf))
            
            # 清理位置数据
            ball_pos = clean_ball_pos(ball_pos, frame_count)
            current_frame_ball_samples = [position for position in ball_pos if position[1] == frame_count]
            if current_frame_ball_samples:
                attribution_ball_pos.extend(current_frame_ball_samples)
                attribution_ball_pos = [
                    position for position in attribution_ball_pos
                    if frame_count - position[1] <= 150
                ]
                encoded_ok, encoded_frame = cv2.imencode(
                    '.jpg',
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                recent_ball_frames[frame_count] = encoded_frame if encoded_ok else frame.copy()
                expired_ball_frames = [
                    buffered_frame
                    for buffered_frame in recent_ball_frames
                    if frame_count - buffered_frame > 150
                ]
                for buffered_frame in expired_ball_frames:
                    recent_ball_frames.pop(buffered_frame, None)
            if len(hoop_pos) > 1:
                hoop_pos = clean_hoop_pos(hoop_pos)
            
            # 投篮检测逻辑
            if len(hoop_pos) > 0 and len(ball_pos) > 0:
                scoring_event = find_recent_score_event(ball_pos, hoop_pos)
                if scoring_event:
                    up = True
                    down = True
                    up_frame = int(scoring_event['up_frame'])
                    down_frame = int(scoring_event['down_frame'])

                # 检测球在上方区域
                if not up:
                    detected_up_frame = find_recent_up_frame(ball_pos, hoop_pos)
                    up = detected_up_frame is not None or detect_up(ball_pos, hoop_pos)
                    if detected_up_frame is not None:
                        up_frame = int(detected_up_frame)
                
                # 检测球在下方区域
                if up and not down:
                    detected_down_frame = find_recent_down_frame(ball_pos, hoop_pos, after_frame=up_frame)
                    down = detected_down_frame is not None or detect_down(ball_pos, hoop_pos)
                    if detected_down_frame is not None:
                        down_frame = int(detected_down_frame)
                
                # 判断是否完成一次投篮
                if scoring_event or frame_count % 10 == 0:
                    if up and down and up_frame < down_frame:
                        is_duplicate_attempt = self._is_duplicate_attempt(
                            last_release_frame=last_release_frame,
                            last_down_frame=last_attempt_frame,
                            current_release_frame=up_frame,
                            current_down_frame=down_frame,
                        )

                        if not is_duplicate_attempt:
                            attempts += 1
                            last_attempt_frame = down_frame
                            last_release_frame = up_frame

                            # 判断是否进球
                            is_made = bool(scoring_event) or score(ball_pos, hoop_pos)
                            attribution = classify_shot_involvement(
                                attribution_ball_pos,
                                target_tracker,
                                up_frame,
                            )
                            local_review = (
                                review_shot_with_local_window(
                                    attribution_ball_pos,
                                    recent_ball_frames,
                                    target_tracker,
                                    up_frame,
                                )
                                if target_tracker
                                else None
                            )
                            owner = attribution['owner']
                            owner_confidence = attribution['owner_confidence']
                            target_visible = attribution['target_visible']
                            highlight_role = attribution['highlight_role']
                            highlight_confidence = attribution['highlight_confidence']
                            local_target_visible = bool(local_review.get('target_visible')) if local_review else False
                            local_owner_confidence = (
                                float(local_review.get('owner_confidence') or 0.0)
                                if local_review
                                else 0.0
                            )
                            local_highlight_role = (
                                str(local_review.get('highlight_role') or 'none')
                                if local_review
                                else 'none'
                            )
                            local_highlight_confidence = (
                                float(local_review.get('highlight_confidence') or 0.0)
                                if local_review
                                else 0.0
                            )
                            target_visible = bool(target_visible or local_target_visible)

                            if is_made:
                                makes += 1
                            if owner == 'target':
                                target_attempts += 1
                                if is_made:
                                    target_makes += 1

                            # 记录这次投篮
                            shot_results.append({
                                'frame': down_frame,
                                'timestamp': round(down_frame / fps, 2),
                                'release_frame': up_frame,
                                'release_timestamp': (
                                    round(up_frame / fps, 2)
                                    if fps > 0
                                    else None
                                ),
                                'made': is_made,
                                'score_event_detected': bool(scoring_event),
                                'owner': owner,
                                'owner_confidence': owner_confidence,
                                'target_visible': target_visible,
                                'attribution_highlight_role': highlight_role,
                                'attribution_highlight_confidence': round(highlight_confidence, 3),
                                'highlight_role': highlight_role if is_made else 'none',
                                'highlight_confidence': highlight_confidence if is_made else 0.0,
                                'involvement_start_frame': attribution.get('involvement_start_frame'),
                                'involvement_end_frame': attribution.get('involvement_end_frame'),
                                'local_target_visible': local_target_visible,
                                'local_owner_confidence': round(local_owner_confidence, 3),
                                'local_highlight_role': local_highlight_role,
                                'local_highlight_confidence': round(local_highlight_confidence, 3),
                                'local_involvement_start_frame': (
                                    local_review.get('involvement_start_frame')
                                    if local_review
                                    else None
                                ),
                                'local_involvement_end_frame': (
                                    local_review.get('involvement_end_frame')
                                    if local_review
                                    else None
                                ),
                                'involvement_start_timestamp': (
                                    round(attribution['involvement_start_frame'] / fps, 2)
                                    if attribution.get('involvement_start_frame') is not None and fps > 0
                                    else None
                                ),
                                'involvement_end_timestamp': (
                                    round(attribution['involvement_end_frame'] / fps, 2)
                                        if attribution.get('involvement_end_frame') is not None and fps > 0
                                        else None
                                ),
                                'local_involvement_start_timestamp': (
                                    round(local_review['involvement_start_frame'] / fps, 2)
                                    if (
                                        local_review
                                        and local_review.get('involvement_start_frame') is not None
                                        and fps > 0
                                    )
                                    else None
                                ),
                                'local_involvement_end_timestamp': (
                                    round(local_review['involvement_end_frame'] / fps, 2)
                                    if (
                                        local_review
                                        and local_review.get('involvement_end_frame') is not None
                                        and fps > 0
                                    )
                                    else None
                                ),
                            })

                            print(f"检测到投篮 #{attempts} - "
                                  f"帧: {down_frame}, "
                                  f"时间: {down_frame/fps:.2f}s, "
                                  f"{'进球' if is_made else '未进'}"
                                  f"{'，归因给目标球员' if owner == 'target' else ''}"
                                  f"{'，归因为目标助攻' if is_made and highlight_role == 'assist' else ''}")
                        
                        # 重置检测标志
                        up = False
                        down = False
                        scoring_event = None
            
            # 在当前帧上绘制标注（蓝色轨迹点 + 红色篮球框）
            if annotate and writer:
                try:
                    # 蓝色点标记篮球轨迹
                    for bp in ball_pos:
                        cv2.circle(frame, bp[0], 3, (255, 0, 0), -1)  # 蓝色点 (BGR)

                    # 红色框框选篮球（仅当前帧检测到的篮球）
                    for (bx1, by1, bx2, by2, bconf) in ball_boxes_in_frame:
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

                    if tracker_record and tracker_record.get('bbox') is not None:
                        draw_target_bbox(
                            frame,
                            tracker_record.get('bbox'),
                            visible=bool(tracker_record.get('visible')),
                            status=str(tracker_record.get('status', 'tracking')),
                            confidence=float(tracker_record.get('confidence', 0.0)),
                        )

                    writer.write(frame)
                except Exception as e:
                    # 如果某帧写入失败，不影响检测流程
                    pass

            frame_count += 1
            
            # 进度回调
            if progress_callback and frame_count % 30 == 0:
                progress_callback(frame_count, total_frames)
        
        cap.release()
        if writer:
            writer.release()
        
        # 打印统计信息
        accuracy = (makes / attempts * 100) if attempts > 0 else 0
        print(f"\n检测完成:")
        print(f"  总投篮次数: {attempts}")
        print(f"  进球次数: {makes}")
        print(f"  命中率: {accuracy:.2f}%")
        print(f"  检测到的进球时刻: {len([s for s in shot_results if s['made']])}")
        if target_tracker:
            tracker_summary = target_tracker.get_summary()
            tracker_summary['primedReferenceSamples'] = primed_reference_samples
            tracker_summary['runtimeReferenceSamples'] = live_reference_samples
            print(f"  目标球员覆盖率: {tracker_summary['coverage'] * 100:.1f}%")
            print(f"  目标球员投篮: {target_attempts} 次, 命中: {target_makes} 次")
            print(
                f"  跟踪状态: {tracker_summary.get('latestStatus', 'unknown')}, "
                f"重获次数: {tracker_summary.get('reacquiredCount', 0)}, "
                f"阻止误切换: {tracker_summary.get('guardedSwitches', 0)}"
            )
        elif tracker_error:
            print(f"  目标球员跟踪未启用: {tracker_error}")

        tracking_result = tracker_summary if target_tracker else {
                'enabled': False,
                'error': tracker_error,
                'activeFrames': 0,
                'totalFrames': frame_count,
                'coverage': 0.0,
                'missingFrames': 0,
                'lostFrames': 0,
                'reacquiredCount': 0,
                'guardedSwitches': 0,
                'latestStatus': 'disabled',
                'startFrame': target_tracker_start_frame,
                'startTime': round(target_tracker_start_time, 3),
                'primedReferenceSamples': primed_reference_samples,
                'runtimeReferenceSamples': live_reference_samples,
        }

        return shot_results, tracking_result
    
    def detect_shots_with_clips(
        self,
        video_path: str,
        before_seconds=8,
        after_seconds=2,
        progress_callback=None,
        annotate: bool = False,
        annotated_output_path: Optional[str] = None,
        target_player_box: Optional[Dict] = None,
    ) -> Dict:
        """
        检测进球并返回每个进球的剪辑时间段
        
        Args:
            video_path: 视频文件路径
            before_seconds: 进球前保留的秒数
            after_seconds: 进球后保留的秒数
            progress_callback: 进度回调函数 callback(current_frame, total_frames)
        
        Returns:
            {
            'shots': 所有投篮列表,
            'made_shots': 只包含进球的列表,
            'selected_made_shots': 归因到目标球员的个人高光列表（进球+助攻）,
            'clips': 剪辑时间段列表,
            'stats': 统计信息,
            'tracking': 目标球员跟踪摘要,
            }
        """
        # 检测所有投篮（可选标注并生成标注视频）
        if annotate and not annotated_output_path:
            base = os.path.splitext(os.path.basename(video_path))[0]
            annotated_output_path = os.path.join(tempfile.gettempdir(), f"{base}_annotated.mp4")

        def run_detection_once(
            run_target_player_box: Optional[Dict],
            run_annotate: bool,
            run_annotated_output_path: Optional[str],
        ) -> Dict:
            run_all_shots, run_tracking_summary = self.detect_shots(
                video_path,
                progress_callback=progress_callback,
                annotate=run_annotate,
                annotated_output_path=run_annotated_output_path,
                target_player_box=run_target_player_box,
            )
            return self._build_detection_output(
                all_shots=run_all_shots,
                tracking_summary=run_tracking_summary,
                video_path=video_path,
                before_seconds=before_seconds,
                after_seconds=after_seconds,
                target_player_box=run_target_player_box,
                annotate=run_annotate,
                annotated_output_path=run_annotated_output_path,
            )

        prealign_retry_candidates = self._discover_retry_target_boxes(
            video_path,
            target_player_box,
        ) if target_player_box else []
        initial_target_player_box = self._select_prealigned_target_player_box_from_candidates(
            target_player_box,
            prealign_retry_candidates,
        )
        if (
            target_player_box
            and initial_target_player_box
            and float(initial_target_player_box.get('selectionTime') or 0.0)
            < float(target_player_box.get('selectionTime') or 0.0)
        ):
            print(
                '目标球员起始帧已自动前移: '
                f"{float(target_player_box.get('selectionTime') or 0.0):.2f}s -> "
                f"{float(initial_target_player_box.get('selectionTime') or 0.0):.2f}s"
            )

        initial_output = run_detection_once(
            initial_target_player_box,
            annotate,
            annotated_output_path,
        )
        final_output = initial_output
        auto_retry_metadata = {
            'attempted': 0,
            'used': False,
            'initialSelectionTime': (
                round(float(target_player_box.get('selectionTime') or 0.0), 3)
                if target_player_box
                else None
            ),
            'finalSelectionTime': (
                round(float(initial_target_player_box.get('selectionTime') or 0.0), 3)
                if initial_target_player_box
                else None
            ),
        }

        retry_outputs: List[Tuple[Optional[str], Dict, Dict]] = []
        if self._should_auto_retry_target_detection(initial_output, initial_target_player_box):
            can_reuse_prealign_candidates = (
                target_player_box == initial_target_player_box
                and bool(prealign_retry_candidates)
            )
            retry_candidates = (
                prealign_retry_candidates
                if can_reuse_prealign_candidates
                else self._discover_retry_target_boxes(video_path, initial_target_player_box)
            )
            best_score = self._score_detection_output(initial_output)
            best_key = 'initial'

            for retry_index, retry_candidate in enumerate(retry_candidates, start=1):
                retry_target_player_box = retry_candidate['target_player_box']
                retry_annotated_output_path = None
                if annotate and annotated_output_path:
                    annotated_base, annotated_ext = os.path.splitext(annotated_output_path)
                    retry_annotated_output_path = (
                        f"{annotated_base}_retry_{retry_index}{annotated_ext or '.mp4'}"
                    )

                retry_output = run_detection_once(
                    retry_target_player_box,
                    annotate,
                    retry_annotated_output_path,
                )
                retry_outputs.append((retry_annotated_output_path, retry_target_player_box, retry_output))
                auto_retry_metadata['attempted'] += 1
                previous_best_score = best_score
                retry_score = self._score_detection_output(retry_output)
                if retry_score > best_score:
                    best_score = retry_score
                    best_key = f'retry_{retry_index}'
                    final_output = retry_output
                    auto_retry_metadata['used'] = True
                    auto_retry_metadata['finalSelectionTime'] = round(
                        float(retry_target_player_box.get('selectionTime') or 0.0),
                        3,
                    )
                    if (
                        retry_score > previous_best_score
                        and self._should_stop_auto_retry_after_stable_improvement(retry_output)
                    ):
                        break

            if annotate:
                kept_annotated_path = final_output.get('annotated_video')
                candidate_paths = [annotated_output_path]
                candidate_paths.extend(path for path, _, _ in retry_outputs if path)
                for candidate_path in candidate_paths:
                    if not candidate_path or candidate_path == kept_annotated_path:
                        continue
                    if os.path.exists(candidate_path):
                        try:
                            os.remove(candidate_path)
                        except OSError:
                            pass

            auto_retry_metadata['selectedRun'] = best_key

        final_output['auto_retry'] = auto_retry_metadata
        return final_output


# 测试代码
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    detector = BasketballShotDetector(model_path=os.path.join(base_dir, 'best.pt'))
    test_video = os.path.join(base_dir, 'test_files', 'video_test_1.mp4')
    
    # 检测并输出剪辑信息
    print("\n" + "=" * 50)
    print("检测进球并生成剪辑信息")
    print("=" * 50)
    result = detector.detect_shots_with_clips(test_video, before_seconds=8, after_seconds=2)
    
    print(f"\n统计:")
    print(f"  总投篮: {result['stats']['total_attempts']}")
    print(f"  进球数: {result['stats']['total_makes']}")
    print(f"  命中率: {result['stats']['accuracy']}%")
    print(f"\n需要剪辑的片段数: {len(result['clips'])}")
    
    for i, clip in enumerate(result['clips'], 1):
        print(f"  片段 {i}: {clip['start']:.2f}s - {clip['end']:.2f}s")
