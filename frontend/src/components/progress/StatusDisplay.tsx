import React from 'react';
import { Alert, Button, Card, Chip } from '@heroui/react';
import { CheckCircle2, Eye, FileVideo, RefreshCcw, UserRound } from 'lucide-react';
import type { Task } from '@/types';
import { formatTimestamp } from '@/utils';

interface StatusDisplayProps {
  task: Task;
  onRetry?: () => void;
  onViewResult?: () => void;
  onReuseSource?: () => void;
  retryLoading?: boolean;
  className?: string;
}

const getStatusChip = (status: Task['status']) => {
  if (status === 'completed') return <Chip color="success" variant="soft">已完成</Chip>;
  if (status === 'failed') return <Chip color="danger" variant="soft">失败</Chip>;
  if (status === 'pending') return <Chip color="default" variant="soft">等待中</Chip>;
  return <Chip color="warning" variant="soft">处理中</Chip>;
};

const getDeliveryModeLabel = ({
  confirmedHighlights,
  possibleHighlights,
  diagnosticsOutcome,
}: {
  confirmedHighlights: number;
  possibleHighlights: number;
  diagnosticsOutcome?: string;
}) => {
  if (confirmedHighlights > 0 && possibleHighlights > 0) {
    return '已确认片段 + 高级排错';
  }
  if (confirmedHighlights > 0) {
    return '已确认片段';
  }
  if (possibleHighlights > 0) {
    return '高级排错片段';
  }
  if (diagnosticsOutcome === 'global_makes_without_target') {
    return '暂未稳定锁定到目标片段';
  }
  return '等待人工复核';
};

export const StatusDisplay: React.FC<StatusDisplayProps> = ({
  task,
  onRetry,
  onViewResult,
  onReuseSource,
  retryLoading = false,
  className = '',
}) => {
  const result = task.result;
  const diagnosticsOutcome = result?.diagnostics?.outcome;
  const trackingCoveragePercent = (result?.tracking?.coverage ?? 0) * 100;
  const hasTrackingRisk = Boolean(result?.tracking?.enabled) && trackingCoveragePercent < 55;
  const hasCompletionTimestamp = Boolean(
    task.updated_at && !Number.isNaN(new Date(task.updated_at).getTime()),
  );
  const formattedCompletionTimestamp = hasCompletionTimestamp && task.updated_at
    ? formatTimestamp(task.updated_at)
    : null;
  const confirmedHighlights = result?.selectionSummary?.confirmed ?? (
    (result?.targetScores ?? 0) + (result?.targetAssists ?? 0)
  );
  const debugHighlights = result?.possibleHighlights ?? result?.debugClips?.length ?? 0;
  const deliveryModeLabel = getDeliveryModeLabel({
    confirmedHighlights,
    possibleHighlights: debugHighlights,
    diagnosticsOutcome,
  });
  const completionSummary = (() => {
    if (diagnosticsOutcome === 'global_makes_without_target') {
      return '本地处理已完成，但当前没有稳定锁定到目标球员相关片段。先去结果页看高级排错区；如果你怀疑人物跟丢，再看标注视频，然后重新框选并重跑。';
    }
    if (hasTrackingRisk) {
      return '本地处理已完成，但当前人物跟踪还不够稳定。先验收已确认片段；只有怀疑漏剪时，再打开结果页里的高级排错区和标注视频。';
    }
    if (debugHighlights > 0) {
      return '本地处理已完成。结果页会先交付已确认片段，系统补充片段只会放进高级排错区，不再混入主交付。';
    }
    if (result?.highlightVideo) {
      return '本地处理已完成。先进入结果页验收已确认片段，附加拼接视频只作为补充查看。';
    }
    return '本地处理已完成。当前没有可下载的已确认片段，但你仍然可以进入结果页查看识别结果。';
  })();
  const statItems = [
    { label: '主交付片段', value: result?.relatedHighlights ?? result?.timestamps?.length ?? 0, icon: FileVideo },
    { label: '已确认片段', value: confirmedHighlights, icon: CheckCircle2 },
    { label: '高级排错片段', value: debugHighlights, icon: UserRound },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-sm uppercase tracking-[0.2em] text-slate-400">任务</div>
                <h3 className="mt-2 text-xl font-semibold text-slate-950">当前任务</h3>
              </div>
              {getStatusChip(task.status)}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">当前阶段</div>
                <div className="mt-2 text-sm text-slate-700">{task.stage || '等待开始'}</div>
              </div>
              {task.status === 'completed' && hasCompletionTimestamp ? (
                <div>
                  <div className="text-xs uppercase tracking-[0.16em] text-slate-400">完成时间</div>
                  <div className="mt-2 text-sm text-slate-700">{formattedCompletionTimestamp}</div>
                </div>
              ) : null}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-700">
              这页只用来确认自动处理是否完成。真正的验收动作放在结果页：先下载已确认片段；只有怀疑漏剪时，再看高级排错区里的系统补充片段，不满意就重新框选并重跑。
            </div>
          </div>
        </Card>

        <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="space-y-5">
            <div>
              <div className="text-sm uppercase tracking-[0.2em] text-slate-400">交付</div>
              <h3 className="mt-2 text-xl font-semibold text-slate-950">交付概览</h3>
            </div>

            {result ? (
              <div className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm text-slate-700">
                当前交付：{deliveryModeLabel}
              </div>
            ) : null}

            {task.status === 'completed' && diagnosticsOutcome === 'global_makes_without_target' ? (
              <Alert status="warning">
                <div className="text-sm leading-6 text-current/85">
                  当前检测到了全场进球，但还没有稳定锁定到目标球员。建议先看结果页里的诊断和标注视频，再重新框选并重跑。
                </div>
              </Alert>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-2">
              {statItems.map(({ label, value, icon: Icon }) => (
                <div key={label} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="flex items-center justify-between">
                    <div className="text-sm text-slate-500">{label}</div>
                    <Icon size={16} className="text-orange-500" />
                  </div>
                  <div className="mt-3 text-2xl font-semibold tracking-tight text-slate-950">{value}</div>
                </div>
              ))}
            </div>

            {task.status === 'failed' && task.error_message ? (
              <Alert status="danger">
                <div className="flex flex-col gap-2">
                  <div className="font-medium text-current">处理失败</div>
                  <div className="text-sm text-current/80">{task.error_message}</div>
                  <div className="flex flex-wrap gap-2">
                    {onRetry ? (
                      <Button variant="danger-soft" isDisabled={retryLoading} onClick={onRetry}>
                        <span className="inline-flex items-center gap-2">
                          <RefreshCcw size={14} />
                          {retryLoading ? '刷新中...' : '刷新状态'}
                        </span>
                      </Button>
                    ) : null}
                    {onReuseSource ? (
                      <Button variant="secondary" onClick={onReuseSource}>
                        <span className="inline-flex items-center gap-2">
                          <RefreshCcw size={14} />
                          重新框选并重跑
                        </span>
                      </Button>
                    ) : null}
                  </div>
                </div>
              </Alert>
            ) : null}
          </div>
        </Card>
      </div>

      {task.status === 'completed' ? (
        <Card className="border border-slate-900/5 bg-slate-950 p-6 text-white shadow-[0_30px_90px_rgba(15,23,42,0.26)]">
          <div className="space-y-4">
            <div className="max-w-3xl text-sm leading-6 text-slate-300">
              {completionSummary}
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              {onViewResult ? (
                <Button variant="primary" onClick={onViewResult} className="rounded-full">
                  <span className="inline-flex items-center gap-2">
                    <Eye size={16} />
                    去结果页验收片段
                  </span>
                </Button>
              ) : null}
              {onReuseSource ? (
                <Button variant="secondary" onClick={onReuseSource} className="rounded-full">
                  <span className="inline-flex items-center gap-2">
                    <RefreshCcw size={16} />
                    重新框选并重跑
                  </span>
                </Button>
              ) : null}
            </div>
          </div>
        </Card>
      ) : null}
    </div>
  );
};
