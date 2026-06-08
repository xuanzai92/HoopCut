import React from 'react';
import { Card, Chip, ProgressBar } from '@heroui/react';
import {
  CircleCheckBig,
  FolderCog,
  Radar,
  Search,
  Sparkles,
  WandSparkles,
} from 'lucide-react';
import type { ProcessingStage, TaskStatus } from '@/types';

interface ProgressIndicatorProps {
  status: TaskStatus;
  stage: ProcessingStage;
  progress: number;
  message?: string;
  currentStep?: string;
  totalSteps?: number;
  estimatedTime?: number;
  className?: string;
}

const stages: Array<{
  key: ProcessingStage;
  title: string;
  description: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
}> = [
  { key: 'uploading', title: '准备任务', description: '校验参数并初始化本地处理流程。', icon: FolderCog },
  { key: 'analyzing', title: '分析画面', description: '逐帧分析视频内容和目标球员出镜位置。', icon: Radar },
  { key: 'detecting', title: '检测投篮', description: '识别出手、轨迹和进球结果。', icon: Search },
  { key: 'generating', title: '剪辑高光', description: '拼接属于你的进球和助攻镜头。', icon: Sparkles },
  { key: 'finalizing', title: '整理结果', description: '写入统计信息并导出最终文件。', icon: WandSparkles },
  { key: 'completed', title: '处理完成', description: '结果已经可查看或下载。', icon: CircleCheckBig },
];

const formatEstimatedTime = (seconds: number) => {
  if (seconds < 60) return `约 ${Math.ceil(seconds)} 秒`;
  if (seconds < 3600) return `约 ${Math.ceil(seconds / 60)} 分钟`;
  return `约 ${Math.ceil(seconds / 3600)} 小时`;
};

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  status,
  stage,
  progress,
  message,
  currentStep,
  totalSteps,
  estimatedTime,
  className = '',
}) => {
  const currentIndex = Math.max(
    stages.findIndex((item) => item.key === stage),
    0,
  );

  return (
    <div className={`space-y-6 ${className}`}>
      <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
        <div className="space-y-6">
          <div className="space-y-2">
            <Chip variant="soft" color={status === 'failed' ? 'danger' : status === 'completed' ? 'success' : 'warning'}>
              {status === 'failed' ? '处理失败' : status === 'completed' ? '处理完成' : '处理中'}
            </Chip>
            <h2 className="text-3xl font-semibold tracking-tight text-slate-950">
              {stages[currentIndex]?.title ?? '处理中'}
            </h2>
            <p className="max-w-3xl text-sm leading-6 text-slate-500">
              {message || stages[currentIndex]?.description}
            </p>
          </div>

          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px] lg:items-end">
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-500">
                <span>整体进度</span>
                <span className="font-semibold text-slate-900">{Math.round(progress)}%</span>
              </div>
              <ProgressBar value={Math.max(0, Math.min(progress, 100))} color={status === 'failed' ? 'danger' : status === 'completed' ? 'success' : 'warning'} />
              {currentStep ? (
                <div className="rounded-2xl border border-orange-100 bg-orange-50/80 px-4 py-3 text-sm text-slate-600">
                  {currentStep}
                </div>
              ) : null}
            </div>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
              <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                <div className="text-xs uppercase tracking-[0.18em] text-slate-400">阶段</div>
                <div className="mt-2 text-lg font-semibold text-slate-900">{currentIndex + 1}{totalSteps ? ` / ${totalSteps}` : ''}</div>
              </div>
              {estimatedTime && estimatedTime > 0 ? (
                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="text-xs uppercase tracking-[0.18em] text-slate-400">预计剩余</div>
                  <div className="mt-2 text-lg font-semibold text-slate-900">{formatEstimatedTime(estimatedTime)}</div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {stages.map((item, index) => {
          const Icon = item.icon;
          const active = index === currentIndex;
          const done = index < currentIndex || status === 'completed';
          const failed = status === 'failed' && index === currentIndex;

          return (
            <Card
              key={item.key}
              className={`border p-5 shadow-[0_16px_50px_rgba(15,23,42,0.06)] transition ${
                active
                  ? 'border-orange-200 bg-orange-50/85'
                  : done
                    ? 'border-emerald-200 bg-emerald-50/70'
                    : failed
                      ? 'border-rose-200 bg-rose-50/80'
                      : 'border-white/40 bg-white/76'
              }`}
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-950 text-white">
                    <Icon size={18} />
                  </div>
                  <Chip
                    variant="soft"
                    color={failed ? 'danger' : done ? 'success' : active ? 'warning' : 'default'}
                  >
                    {failed ? '异常' : done ? '完成' : active ? '当前' : '待处理'}
                  </Chip>
                </div>
                <div className="text-lg font-semibold text-slate-900">{item.title}</div>
                <p className="text-sm leading-6 text-slate-500">{item.description}</p>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};
