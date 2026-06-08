import React from 'react';
import { Card, Chip, ProgressBar } from '@heroui/react';
import { Crosshair, Goal, Medal, Radar, Sparkles, TimerReset } from 'lucide-react';
import type { ProcessingResult, ShotTimestamp } from '@/types';
import { formatDuration, formatFileSize, formatTimestamp } from '@/utils';

interface ResultStatsProps {
  result: ProcessingResult;
  className?: string;
}

const getHighlightMeta = (timestamp: ShotTimestamp) => {
  if (timestamp.highlight_role === 'assist') {
    return { color: 'warning' as const, label: '你的助攻' };
  }

  if (timestamp.highlight_role === 'score' || timestamp.owner === 'target') {
    return { color: 'success' as const, label: '你的进球' };
  }

  return { color: 'default' as const, label: '全场进球' };
};

export const ResultStats: React.FC<ResultStatsProps> = ({ result, className = '' }) => {
  const totalShots = result.totalShots ?? 0;
  const madeShots = result.madeShots ?? 0;
  const targetTimestamps = result.timestamps ?? [];
  const targetScores =
    result.targetScores ??
    result.targetShots ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'score' || timestamp.owner === 'target').length;
  const targetAssists =
    result.targetAssists ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'assist').length;
  const targetHighlights = result.targetHighlights ?? targetTimestamps.length;
  const accuracy = result.accuracy ?? 0;
  const trackingCoverage = (result.tracking?.coverage ?? 0) * 100;
  const outputFileSize = result.fileSize ?? result.file_size ?? 0;

  const metrics = [
    { label: '总投篮', value: totalShots, icon: Crosshair },
    { label: '全场进球', value: madeShots, icon: Goal },
    { label: '你的高光', value: targetHighlights, icon: Sparkles },
    { label: '你的进球', value: targetScores, icon: Medal },
    { label: '你的助攻', value: targetAssists, icon: Radar },
    { label: '命中率', value: `${accuracy.toFixed(1)}%`, icon: TimerReset },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-5">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Overview</div>
            <h3 className="mt-2 text-xl font-semibold text-slate-950">个人结果概览</h3>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
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
            <ProgressBar value={trackingCoverage} color={result.tracking?.enabled ? 'success' : 'default'} />
          </div>

          {result.message ? (
            <div className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm text-slate-600">
              {result.message}
            </div>
          ) : null}
        </div>
      </Card>

      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-4">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Output</div>
            <h3 className="mt-2 text-xl font-semibold text-slate-950">输出信息</h3>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="text-sm text-slate-600">是否生成集锦：<span className="font-medium text-slate-900">{result.highlightVideo ? '已生成' : '未生成'}</span></div>
            <div className="text-sm text-slate-600">标注视频：<span className="font-medium text-slate-900">{result.annotatedVideo ? '已生成' : '未生成'}</span></div>
            {outputFileSize > 0 ? <div className="text-sm text-slate-600">输出文件大小：<span className="font-medium text-slate-900">{formatFileSize(outputFileSize)}</span></div> : null}
            {result.completed_at ? <div className="text-sm text-slate-600">完成时间：<span className="font-medium text-slate-900">{formatTimestamp(result.completed_at)}</span></div> : null}
            {result.targetPlayerBox ? <div className="text-sm text-slate-600">初始选区：<span className="font-medium text-slate-900">{result.targetPlayerBox.width} x {result.targetPlayerBox.height}</span></div> : null}
            {typeof result.targetPlayerBox?.selectionTime === 'number' ? (
              <div className="text-sm text-slate-600">
                跟踪起点：
                <span className="font-medium text-slate-900">
                  {formatDuration(result.targetPlayerBox.selectionTime)} ({result.targetPlayerBox.selectionTime.toFixed(2)}s)
                </span>
              </div>
            ) : null}
          </div>
        </div>
      </Card>

      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-4">
          <div>
            <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Moments</div>
            <h3 className="mt-2 text-xl font-semibold text-slate-950">你的高光时刻</h3>
          </div>

          {targetTimestamps.length > 0 ? (
            <div className="space-y-3">
              {targetTimestamps.slice(0, 5).map((timestamp, index) => {
                const meta = getHighlightMeta(timestamp);
                return (
                  <div key={`${timestamp.frame}-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div>
                        <div className="font-medium text-slate-900">镜头 {index + 1}</div>
                        <div className="mt-1 text-sm text-slate-500">
                          {timestamp.timestamp.toFixed(2)}s · 帧 {timestamp.frame}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Chip color={meta.color} variant="soft">{meta.label}</Chip>
                        {typeof timestamp.highlight_confidence === 'number' && timestamp.highlight_confidence > 0 ? (
                          <Chip color="accent" variant="soft">
                            高光置信度 {(timestamp.highlight_confidence * 100).toFixed(0)}%
                          </Chip>
                        ) : null}
                      </div>
                    </div>
                  </div>
                );
              })}
              {targetTimestamps.length > 5 ? (
                <div className="text-sm text-slate-500">还有 {targetTimestamps.length - 5} 个个人高光时刻</div>
              ) : null}
            </div>
          ) : (
            <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 px-4 py-8 text-center text-sm text-slate-500">
              当前没有归因到你的高光时刻
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default ResultStats;
