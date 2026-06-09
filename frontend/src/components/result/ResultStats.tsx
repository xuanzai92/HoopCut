import React from 'react';
import { Card, Chip, ProgressBar } from '@heroui/react';
import { Medal, Radar, Sparkles, TimerReset } from 'lucide-react';
import type { ProcessingResult } from '@/types';
import { formatDuration, formatFileSize, formatTimestamp } from '@/utils';

interface ResultStatsProps {
  result: ProcessingResult;
  className?: string;
}

const getSelectionModeLabel = (mode?: string) => {
  switch (mode) {
    case 'mixed_with_review_candidates':
      return '已确认片段 + 高级排错片段';
    case 'review_candidates_fallback':
      return '高级排错片段';
    case 'target_attempt_fallback':
      return '目标相关高级排错片段';
    case 'no_target_highlights':
      return '暂未稳定锁定目标片段';
    case 'all_made_fallback':
      return '历史结果（未区分目标）';
    case 'mixed':
      return '已确认片段';
    default:
      return '自动整理结果';
  }
};

export const ResultStats: React.FC<ResultStatsProps> = ({ result, className = '' }) => {
  const targetTimestamps = result.timestamps ?? [];
  const debugTimestamps = result.debugTimestamps ?? [];
  const targetScores =
    result.targetScores ??
    result.targetShots ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'score' || timestamp.owner === 'target').length;
  const targetAssists =
    result.targetAssists ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'assist').length;
  const possibleHighlights =
    result.possibleHighlights ??
    debugTimestamps.filter((timestamp) => timestamp.highlight_role === 'possible').length;
  const targetHighlights = result.relatedHighlights ?? targetTimestamps.length;
  const confirmedHighlights = result.selectionSummary?.confirmed ?? Math.max(targetScores + targetAssists, 0);
  const trackingCoverage = (result.tracking?.coverage ?? 0) * 100;
  const hasTrackingRisk = Boolean(result.tracking?.enabled) && trackingCoverage < 55;
  const outputFileSize = result.fileSize ?? result.file_size ?? 0;
  const selectionMode = result.selectionSummary?.mode ?? result.pipeline?.attribution?.selectionMode;
  const effectiveTargetPlayerBox = result.effectiveTargetPlayerBox ?? null;
  const userSelectionTime = result.targetPlayerBox?.selectionTime;
  const effectiveSelectionTime = effectiveTargetPlayerBox?.selectionTime;
  const systemShiftedTrackingStart = (
    typeof userSelectionTime === 'number'
    && typeof effectiveSelectionTime === 'number'
    && Math.abs(userSelectionTime - effectiveSelectionTime) >= 0.01
  );
  const diagnosticsOutcome = result.diagnostics?.outcome;
  const reviewSummary = (() => {
    if (diagnosticsOutcome === 'global_makes_without_target') {
      return '当前检测到了全场进球，但没有稳定锁定到目标球员。建议先看标注视频，再重新框选并重跑。';
    }
    if (hasTrackingRisk) {
      return `当前跟踪覆盖率只有 ${trackingCoverage.toFixed(1)}%，建议先看标注视频，再决定是否直接验收片段。`;
    }
    if (possibleHighlights > 0 && confirmedHighlights === 0) {
      return `当前只有 ${possibleHighlights} 个高级排错片段，说明系统还没有稳定确认到目标球员的进球或助攻。`;
    }
    return '当前主交付是已确认片段 ZIP，拼接视频、标注视频和系统补充片段都只用于辅助复核。';
  })();

  const metrics = [
    { label: '主交付片段', value: targetHighlights, icon: Sparkles },
    { label: '已确认', value: confirmedHighlights, icon: Medal },
    { label: '高级排错片段', value: possibleHighlights, icon: TimerReset },
    { label: '目标球员助攻', value: targetAssists, icon: Radar },
  ];

  const acceptanceSteps = [
    possibleHighlights > 0
      ? `先验收已确认片段；只有怀疑漏剪时，再展开 ${possibleHighlights} 个系统补充片段`
      : '可以优先下载全部片段 ZIP 直接验收',
    hasTrackingRisk
      ? `当前跟踪覆盖率只有 ${trackingCoverage.toFixed(1)}%，建议同时打开标注视频核对人物跟踪`
      : '当前目标跟踪整体稳定，可以先从已确认片段开始看',
    result.annotatedVideo
      ? '标注视频已生成，可用于排查跟踪是否跑偏'
      : '当前没有额外标注视频，直接以片段结果为主验收',
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-5">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-slate-400">交付</div>
            <h3 className="mt-2 text-xl font-semibold text-slate-950">交付摘要</h3>
          </div>

          <div className="flex flex-wrap gap-2">
            <Chip color="success" variant="soft">主交付：片段 ZIP</Chip>
            <Chip color={possibleHighlights > 0 ? 'accent' : 'default'} variant="soft">
              {getSelectionModeLabel(selectionMode)}
            </Chip>
            {hasTrackingRisk ? (
              <Chip color="warning" variant="soft">跟踪需重点复核</Chip>
            ) : null}
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            {metrics.map(({ label, value, icon: Icon }) => (
              <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-slate-500">{label}</div>
                  <Icon size={16} className="text-orange-500" />
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-tight text-slate-950">{value}</div>
              </div>
            ))}
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
            <div className="mb-2 flex items-center justify-between text-sm text-slate-500">
              <span>人物跟踪覆盖率</span>
              <span className="font-semibold text-slate-900">{trackingCoverage.toFixed(1)}%</span>
            </div>
            <ProgressBar
              value={trackingCoverage}
              color={
                !result.tracking?.enabled
                  ? 'default'
                  : hasTrackingRisk
                    ? 'warning'
                    : 'success'
              }
            />
          </div>

          {result.message ? (
            <div className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm text-slate-600">
              {result.message}
            </div>
          ) : null}
          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-700">
            {reviewSummary}
          </div>
        </div>
      </Card>

      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-4">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-slate-400">验收</div>
            <h3 className="mt-2 text-xl font-semibold text-slate-950">验收重点</h3>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="text-sm text-slate-600">主交付结果：<span className="font-medium text-slate-900">已确认片段 ZIP 包</span></div>
            <div className="text-sm text-slate-600">主交付片段数量：<span className="font-medium text-slate-900">{targetHighlights}</span></div>
            <div className="text-sm text-slate-600">附加拼接视频：<span className="font-medium text-slate-900">{result.highlightVideo ? '已生成' : '未生成'}</span></div>
            <div className="text-sm text-slate-600">标注视频：<span className="font-medium text-slate-900">{result.annotatedVideo ? '已生成' : '未生成'}</span></div>
            {result.annotatedVideoReason ? (
              <div className="text-sm text-slate-600">
                标注视频保留原因：
                <span className="font-medium text-slate-900">
                  {result.annotatedVideoReason === 'tracking_low_coverage'
                    ? '跟踪覆盖率偏低'
                    : result.annotatedVideoReason === 'highlight_review'
                      ? '存在系统补充片段'
                      : result.annotatedVideoReason === 'risk_review'
                        ? '当前结果需要重点复核'
                        : result.annotatedVideoReason === 'debug'
                          ? '调试模式'
                          : '自动保留'}
                </span>
              </div>
            ) : null}
            {outputFileSize > 0 ? <div className="text-sm text-slate-600">输出文件大小：<span className="font-medium text-slate-900">{formatFileSize(outputFileSize)}</span></div> : null}
            {result.completed_at ? <div className="text-sm text-slate-600">完成时间：<span className="font-medium text-slate-900">{formatTimestamp(result.completed_at)}</span></div> : null}
            {result.targetPlayerBox ? <div className="text-sm text-slate-600">你框选的起始选区：<span className="font-medium text-slate-900">{result.targetPlayerBox.width} x {result.targetPlayerBox.height}</span></div> : null}
            {typeof result.targetPlayerBox?.selectionTime === 'number' ? (
              <div className="text-sm text-slate-600">
                你选择的起始帧：
                <span className="font-medium text-slate-900">
                  {formatDuration(result.targetPlayerBox.selectionTime)} ({result.targetPlayerBox.selectionTime.toFixed(2)}s)
                </span>
              </div>
            ) : null}
            {systemShiftedTrackingStart ? (
              <div className="text-sm text-slate-600">
                系统实际追踪起点：
                <span className="font-medium text-slate-900">
                  {formatDuration(effectiveSelectionTime)} ({effectiveSelectionTime.toFixed(2)}s)
                </span>
                <span className="ml-2 text-slate-500">
                  为了补抓更早回合，系统自动把追踪起点前移了一些。
                </span>
              </div>
            ) : null}
            <div className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm text-slate-700 sm:col-span-2">
              主交付 ZIP 只包含 `score / assist`。高级排错片段会单独下载，`manifest.json` 里也保留了每个片段的角色、原因、来源和高光置信度，方便你排错。
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
            <div className="text-sm font-medium text-slate-900">建议的验收顺序</div>
            <div className="mt-3 space-y-2">
              {acceptanceSteps.map((step) => (
                <div key={step} className="text-sm leading-6 text-slate-600">
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default ResultStats;
