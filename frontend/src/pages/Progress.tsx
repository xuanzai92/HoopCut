import React, { useEffect, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button, Card, Chip, EmptyState, Toast } from '@heroui/react';
import { ArrowLeft, Eye, RefreshCcw } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { ProgressIndicator } from '@/components/progress/ProgressIndicator';
import { StatusDisplay } from '@/components/progress/StatusDisplay';
import { ErrorAlert, LoadingState } from '@/components/common';
import { useErrorHandler, useLoading } from '@/hooks';
import { useAppStore } from '@/store/app';
import { useTaskProgress, QUERY_KEYS } from '@/services/queries';
import { ApiService } from '@/services/api';
import type { ProcessingStage, TaskStatus } from '@/types';
import { HttpRequestError } from '@/services/http';

const normalizeTaskStatus = (status: string): TaskStatus => {
  if (status === 'completed' || status === 'failed' || status === 'pending') {
    return status;
  }
  return 'processing';
};

const resolveProgressStage = (status: string, stageText?: string): ProcessingStage => {
  if (status === 'detecting') {
    return stageText?.includes('分析') ? 'analyzing' : 'detecting';
  }
  if (status === 'attributing') return 'attributing';
  if (status === 'generating') return 'generating';
  if (status === 'completed') return 'completed';
  if (status === 'failed') return 'finalizing';
  return 'uploading';
};

const isNotFoundError = (error: Error) => {
  const status = (error as HttpRequestError).status;
  return status === 404 || error.message.includes('资源不存在') || error.message.includes('404');
};

export const Progress: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const { addNotification } = useAppStore();
  const queryClient = useQueryClient();
  const lastStatusRef = useRef<string | null>(null);
  const { error, hasError, clearError, withErrorHandling } = useErrorHandler();
  const { loading: retryLoading, withLoading } = useLoading();

  const {
    data: progress,
    isLoading,
    error: queryError,
    refetch,
  } = useTaskProgress(fileId ?? '', {
    enabled: Boolean(fileId),
    refetchInterval: false,
  });

  const processingMode = progress?.processingMode ?? progress?.result?.processingMode ?? 'auto';
  const reuseActionLabel = processingMode === 'manual' ? '重新选时间点' : '重新框选并重跑';
  const heroDescription = processingMode === 'manual'
    ? '当前页面只负责展示手动时间点导出是否已经完成。处理结束后，直接去结果页验收片段；如果起止不合适，再回首页调整时间点或前后保留时间。'
    : '当前页面只负责展示自动处理是否已经完成。处理结束后，直接去结果页检查已确认的进球、助攻片段；只有怀疑漏剪时，再看高级排错区。';

  useEffect(() => {
    if (!fileId) return;

    void refetch();

    ApiService.connectWebSocket((data) => {
      if (data.taskId !== fileId) return;

      queryClient.setQueryData(QUERY_KEYS.PROGRESS(fileId), data.data);

      if (data.data.status !== lastStatusRef.current) {
        if (data.data.status === 'completed') {
          const completedMode = data.data.processingMode ?? data.data.result?.processingMode ?? 'auto';
          addNotification({
            type: 'success',
            title: '处理完成',
            message: completedMode === 'manual'
              ? '手动时间点片段已经导出完成，可以查看结果了。'
              : '本地处理已完成，可以查看结果了。',
          });
        } else if (data.data.status === 'failed') {
          addNotification({
            type: 'error',
            title: '处理失败',
            message: data.data.error || data.data.stage || '视频处理过程中出现错误',
          });
        }

        lastStatusRef.current = data.data.status;
      }
    });

    return () => {
      ApiService.disconnectWebSocket();
    };
  }, [addNotification, fileId, queryClient, refetch]);

  useEffect(() => {
    if (progress?.status) {
      lastStatusRef.current = progress.status;
    }
  }, [progress]);

  const handleRetry = withLoading(
    withErrorHandling(async () => {
      if (!fileId) {
        throw new Error('文件ID不存在');
      }
      await refetch();
      Toast.toast.success('已刷新任务状态');
    }),
  );

  const handleRefresh = withErrorHandling(async () => {
    await refetch();
    Toast.toast.success('已刷新任务状态');
  });

  const handleReuseSource = () => {
    if (!fileId) {
      return;
    }
    navigate(`/?reuseTaskId=${fileId}`);
  };

  const normalizedStatus = useMemo(
    () => (progress ? normalizeTaskStatus(progress.status) : 'processing'),
    [progress],
  );

  const normalizedStage = useMemo(
    () => (progress ? resolveProgressStage(progress.status, progress.stage) : 'uploading'),
    [progress],
  );

  if (queryError) {
    const is404 = isNotFoundError(queryError);
    return (
      <div className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)] px-4">
        <Card className="w-full max-w-xl border border-white/40 bg-white/82 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
          <EmptyState>
            <div className="space-y-4 text-center">
              <h1 className="text-3xl font-semibold tracking-tight text-slate-950">
                {is404 ? '任务不存在' : '加载失败'}
              </h1>
              <p className="text-sm leading-6 text-slate-500">
                {is404
                  ? '找不到指定的处理任务，可能已被删除或任务 ID 错误。'
                  : '无法获取任务信息，请检查后端服务或稍后重试。'}
              </p>
              <div className="flex flex-col justify-center gap-3 sm:flex-row">
                <Button variant="secondary" onClick={() => navigate('/')}>返回首页</Button>
                {!is404 ? (
                  <Button variant="primary" onClick={() => void refetch()}>重新加载</Button>
                ) : null}
              </div>
            </div>
          </EmptyState>
        </Card>
      </div>
    );
  }

  if (isLoading) {
    return <LoadingState type="page" tip="加载任务信息中..." size="large" />;
  }

  if (!progress || !fileId) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)] px-4">
        <Card className="w-full max-w-xl border border-white/40 bg-white/82 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
          <div className="space-y-4 text-center">
            <h1 className="text-3xl font-semibold tracking-tight text-slate-950">任务不存在</h1>
            <p className="text-sm leading-6 text-slate-500">当前没有找到可展示的任务信息。</p>
            <Button variant="primary" onClick={() => navigate('/')}>返回首页</Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen overflow-x-hidden bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)]">
      {hasError ? (
        <div className="fixed left-1/2 top-4 z-50 w-full max-w-md -translate-x-1/2 px-4">
          <ErrorAlert
            title="操作失败"
            message={error?.message || '发生未知错误'}
            type="error"
            showIcon
            closable
            onClose={clearError}
          />
        </div>
      ) : null}

      <div className="relative isolate">
        <div className="absolute left-1/2 top-0 -z-10 h-[420px] w-[420px] -translate-x-1/2 rounded-full bg-orange-300/20 blur-3xl" />
        <div className="absolute right-[-100px] top-[120px] -z-10 h-[320px] w-[320px] rounded-full bg-indigo-300/18 blur-3xl" />

        <div className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6 lg:px-8 lg:pt-12">
          <div className="mb-8 flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
            <div className="space-y-4">
              <Chip variant="soft" color={progress.status === 'completed' ? 'success' : progress.status === 'failed' ? 'danger' : 'warning'}>
                {progress.status === 'completed' ? '处理完成' : progress.status === 'failed' ? '处理失败' : '处理中'}
              </Chip>
              <div className="space-y-3">
                <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl">
                  处理进度
                </h1>
                <p className="max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                  {heroDescription}
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button variant="secondary" onClick={() => navigate('/')}>
                <span className="inline-flex items-center gap-2">
                  <ArrowLeft size={16} />
                  返回首页
                </span>
              </Button>
              {progress.status === 'completed' ? (
                <Button variant="primary" onClick={() => navigate(`/result/${fileId}`)}>
                  <span className="inline-flex items-center gap-2">
                    <Eye size={16} />
                    查看结果
                  </span>
                </Button>
              ) : null}
              {progress.status === 'completed' || progress.status === 'failed' ? (
                <Button variant="secondary" onClick={handleReuseSource}>
                  <span className="inline-flex items-center gap-2">
                    <RefreshCcw size={16} />
                    {reuseActionLabel}
                  </span>
                </Button>
              ) : null}
              <Button variant="ghost" onClick={() => void handleRefresh()}>
                <span className="inline-flex items-center gap-2">
                  <RefreshCcw size={16} />
                  刷新
                </span>
              </Button>
            </div>
          </div>

          <div className="space-y-6">
            <ProgressIndicator
              status={normalizedStatus}
              stage={normalizedStage}
              progress={progress.progress}
              message={progress.stage}
              processingMode={processingMode}
              totalSteps={processingMode === 'manual' ? 4 : 7}
            />

            <StatusDisplay
              task={{
                id: fileId,
                status: normalizedStatus,
                processingMode,
                stage: progress.stage,
                progress: progress.progress,
                message: progress.stage || '',
                result: progress.result,
                created_at: progress.createdAt,
                updated_at: progress.updatedAt,
                error_message: progress.error,
              }}
              onRetry={() => void handleRetry()}
              onViewResult={progress.status === 'completed' ? () => navigate(`/result/${fileId}`) : undefined}
              onReuseSource={progress.status === 'completed' || progress.status === 'failed' ? handleReuseSource : undefined}
              retryLoading={retryLoading}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Progress;
