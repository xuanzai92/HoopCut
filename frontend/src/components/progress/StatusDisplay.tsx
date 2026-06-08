import React from 'react';
import { Alert, Button, Card, Chip } from '@heroui/react';
import { CheckCircle2, Clock3, Download, Eye, FileVideo, RefreshCcw, UserRound } from 'lucide-react';
import type { Task } from '@/types';
import { formatDuration, formatFileSize, formatTimestamp } from '@/utils';

interface StatusDisplayProps {
  task: Task;
  onRetry?: () => void;
  onViewResult?: () => void;
  onDownloadResult?: () => void;
  retryLoading?: boolean;
  className?: string;
}

const getStatusChip = (status: Task['status']) => {
  if (status === 'completed') return <Chip color="success" variant="soft">已完成</Chip>;
  if (status === 'failed') return <Chip color="danger" variant="soft">失败</Chip>;
  if (status === 'pending') return <Chip color="default" variant="soft">等待中</Chip>;
  return <Chip color="warning" variant="soft">处理中</Chip>;
};

export const StatusDisplay: React.FC<StatusDisplayProps> = ({
  task,
  onRetry,
  onViewResult,
  onDownloadResult,
  retryLoading = false,
  className = '',
}) => {
  const result = task.result;
  const statItems = [
    { label: '总投篮次数', value: result?.totalShots ?? 0, icon: FileVideo },
    { label: '成功投篮', value: result?.madeShots ?? 0, icon: CheckCircle2 },
    { label: '个人高光', value: result?.targetHighlights ?? result?.timestamps?.length ?? 0, icon: UserRound },
    { label: '跟踪覆盖率', value: `${((result?.tracking?.coverage ?? 0) * 100).toFixed(1)}%`, icon: Clock3 },
  ];

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="space-y-5">
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Task Overview</div>
                <h3 className="mt-2 text-xl font-semibold text-slate-950">任务信息</h3>
              </div>
              {getStatusChip(task.status)}
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">任务 ID</div>
                <div className="mt-2 break-all font-mono text-sm text-slate-700">{task.id}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">创建时间</div>
                <div className="mt-2 text-sm text-slate-700">{formatTimestamp(task.created_at)}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">当前阶段</div>
                <div className="mt-2 text-sm text-slate-700">{task.stage || '等待开始'}</div>
              </div>
              <div>
                <div className="text-xs uppercase tracking-[0.16em] text-slate-400">最近更新时间</div>
                <div className="mt-2 text-sm text-slate-700">{formatTimestamp(task.updated_at)}</div>
              </div>
              {task.video_file ? (
                <>
                  <div>
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">文件名</div>
                    <div className="mt-2 text-sm text-slate-700">{task.video_file.name}</div>
                  </div>
                  <div>
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">文件大小</div>
                    <div className="mt-2 text-sm text-slate-700">{formatFileSize(task.video_file.size)}</div>
                  </div>
                  {task.video_file.duration ? (
                    <div>
                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">视频时长</div>
                      <div className="mt-2 text-sm text-slate-700">{formatDuration(task.video_file.duration)}</div>
                    </div>
                  ) : null}
                </>
              ) : null}
            </div>

            {task.config ? (
              <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                <div className="text-sm font-medium text-slate-900">处理配置</div>
                <div className="mt-3 grid grid-cols-2 gap-3 text-sm text-slate-600">
                  <div>进球前保留：{task.config.beforeSeconds}s</div>
                  <div>进球后保留：{task.config.afterSeconds}s</div>
                </div>
              </div>
            ) : null}
          </div>
        </Card>

        <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="space-y-5">
            <div>
              <div className="text-sm uppercase tracking-[0.2em] text-slate-400">Stats</div>
              <h3 className="mt-2 text-xl font-semibold text-slate-950">处理统计</h3>
            </div>

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
                  {onRetry ? (
                    <div>
                      <Button variant="danger-soft" isDisabled={retryLoading} onClick={onRetry}>
                        <span className="inline-flex items-center gap-2">
                          <RefreshCcw size={14} />
                          {retryLoading ? '刷新中...' : '重试'}
                        </span>
                      </Button>
                    </div>
                  ) : null}
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
              {task.result?.highlightVideo
                ? '本地处理已完成。建议先查看结果页确认你的进球和助攻归因，再决定是否直接下载成片。'
                : '本地处理已完成。当前没有可下载的个人高光视频，但你仍然可以进入结果页查看统计和识别结果。'}
            </div>
            <div className="flex flex-col gap-3 sm:flex-row">
              {onViewResult ? (
                <Button variant="primary" onClick={onViewResult} className="rounded-full">
                  <span className="inline-flex items-center gap-2">
                    <Eye size={16} />
                    查看处理结果
                  </span>
                </Button>
              ) : null}
              {task.result?.highlightVideo && onDownloadResult ? (
                <Button variant="secondary" onClick={onDownloadResult} className="rounded-full">
                  <span className="inline-flex items-center gap-2">
                    <Download size={16} />
                    下载高光视频
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
