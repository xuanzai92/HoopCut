import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button, Card, Chip, EmptyState, Toast } from '@heroui/react';
import { ArrowLeft, CheckSquare, ChevronLeft, ChevronRight, Download, Eye, RefreshCcw } from 'lucide-react';
import { VideoPlayer } from '@/components/result/VideoPlayer';
import { ErrorAlert, LoadingState } from '@/components/common';
import { useErrorHandler, useLoading } from '@/hooks';
import { ApiService } from '@/services/api';
import { useTaskResult } from '@/services';
import { HttpRequestError } from '@/services/http';
import { useAppStore } from '@/store/app';
import type { HighlightClip } from '@/types';
import { formatDuration } from '@/utils';

const isNotFoundError = (error: Error) => {
  const status = (error as HttpRequestError).status;
  return status === 404 || error.message.includes('资源不存在') || error.message.includes('404');
};

const EMPTY_CLIPS: HighlightClip[] = [];

const dedupeMessages = (messages: Array<string | null | undefined>) => {
  const seen = new Set<string>();
  return messages.filter((message): message is string => {
    const normalized = message?.trim();
    if (!normalized || seen.has(normalized)) {
      return false;
    }
    seen.add(normalized);
    return true;
  });
};

const normalizeRecommendedAction = (
  action: string,
  options: {
    hasConfirmedClips: boolean;
    hasReviewClips: boolean;
  },
) => {
  const normalized = action.trim();
  if (!normalized) {
    return null;
  }

  if (
    options.hasConfirmedClips
    && options.hasReviewClips
    && normalized.includes('优先检查系统补充片段')
  ) {
    return '如果你怀疑漏剪，再检查系统补充片段，确认是否还存在漏判进球或助攻。';
  }

  return normalized;
};

const getClipDisplay = (clip: HighlightClip) => {
  if (clip.highlightRole === 'manual') {
    return { label: '手动片段', color: 'warning' as const };
  }
  if (clip.highlightRole === 'assist') {
    return { label: '目标球员助攻', color: 'warning' as const };
  }
  if (clip.highlightRole === 'possible') {
    return { label: '系统补充片段', color: 'accent' as const };
  }
  if (clip.highlightRole === 'score') {
    return { label: '目标球员进球', color: 'success' as const };
  }
  return { label: '相关片段', color: 'default' as const };
};

const getCandidateReasonLabel = (reason?: string) => {
  switch (reason) {
    case 'attempt_local_score_window':
      return '疑似目标球员完成出手，但确认还不够稳定';
    case 'attempt_target_release':
      return '目标球员出手关联较强';
    case 'attempt_local_target_visible':
      return '目标球员在关键出手窗口内清晰可见';
    case 'attempt_target_visible':
      return '目标球员与这个回合关联较强';
    case 'attempt_target_context':
      return '目标球员与这个回合存在基础关联';
    case 'local_score':
      return '局部复核补出的进球候选';
    case 'local_assist':
      return '局部复核补出的助攻候选';
    case 'local_assist_window':
      return '局部复核看到了助攻链路，但最终确认还不够稳定';
    case 'global_assist_window':
      return '整段归因看到了助攻链路，但最终确认还不够稳定';
    case 'local_target_visible':
      return '局部复核看到目标球员参与了这个回合';
    case 'low_tracking_coverage':
      return '人物跟踪不够稳定，系统先保留这个回合避免漏剪';
    case 'all_made_fallback':
      return '历史结果里没有区分目标，先保留了全场进球';
    case 'target_visible':
      return '目标球员在这个进球回合中可见';
    default:
      return null;
  }
};

const getCandidateSourceLabel = (source?: string | null) => {
  switch (source) {
    case 'attempt_review':
      return '系统补充片段';
    case 'target_attempt_fallback':
      return '目标相关补充片段';
    default:
      return null;
  }
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

const CLIP_GROUPS: Array<{
  key: HighlightClip['highlightRole'];
  title: string;
  description: string;
  emptyText: string;
  chipColor: 'success' | 'warning' | 'accent' | 'default';
}> = [
  {
    key: 'score',
    title: '目标球员进球',
    description: '已经确认归因到目标球员的进球片段。',
    emptyText: '当前没有确认的目标球员进球片段',
    chipColor: 'success',
  },
  {
    key: 'assist',
    title: '目标球员助攻',
    description: '已经确认归因到目标球员的助攻片段。',
    emptyText: '当前没有确认的目标球员助攻片段',
    chipColor: 'warning',
  },
  {
    key: 'possible',
    title: '系统补充片段',
    description: '为了尽量找全相关镜头而自动补出的片段，建议你最后快速过一遍。',
    emptyText: '当前没有系统补充片段',
    chipColor: 'accent',
  },
];

export const Result: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const taskId = fileId ?? '';
  const navigate = useNavigate();
  const { addNotification } = useAppStore();
  const [activeClipFilename, setActiveClipFilename] = useState<string>('');
  const [showSupplementalClips, setShowSupplementalClips] = useState(false);
  const [showAuxiliaryReview, setShowAuxiliaryReview] = useState(false);
  const { error, hasError, clearError, withErrorHandling } = useErrorHandler();
  const { loading: actionLoading, withLoading } = useLoading();

  const {
    data: result,
    isLoading,
    error: queryError,
    refetch,
  } = useTaskResult(taskId, {
    enabled: Boolean(taskId),
  });
  const confirmedClips = useMemo(() => result?.clips ?? EMPTY_CLIPS, [result?.clips]);
  const reviewClips = useMemo(() => result?.debugClips ?? EMPTY_CLIPS, [result?.debugClips]);
  const availableClips = useMemo(
    () => [...confirmedClips, ...reviewClips],
    [confirmedClips, reviewClips],
  );

  useEffect(() => {
    if (availableClips.length === 0) {
      setActiveClipFilename('');
      return;
    }

    setActiveClipFilename((current) => (
      availableClips.some((clip) => clip.filename === current)
        ? current
        : (confirmedClips[0] ?? availableClips[0]).filename
    ));
  }, [availableClips, confirmedClips]);

  useEffect(() => {
    if (reviewClips.length === 0) {
      setShowSupplementalClips(false);
      return;
    }

    if (confirmedClips.length === 0) {
      setShowSupplementalClips(true);
    }
  }, [confirmedClips.length, reviewClips.length]);

  const downloadVideoFile = (filename: string, downloadName: string) => {
    const url = ApiService.getDownloadUrl(filename);
    const link = document.createElement('a');
    link.href = url;
    link.download = downloadName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleDownload = withLoading(
    withErrorHandling(async () => {
      if (!result?.highlightVideo) {
        throw new Error('当前没有额外生成已确认片段拼接视频');
      }

      downloadVideoFile(result.highlightVideo, `basketball_highlight_${result.highlightVideo}`);

      addNotification({
        type: 'success',
        title: '下载开始',
        message: result?.processingMode === 'manual'
          ? '手动片段拼接视频下载已开始'
          : '已确认片段拼接视频下载已开始',
      });
    }),
  );

  const handleDownloadAnnotated = withErrorHandling(async () => {
    if (!result?.annotatedVideo) {
      throw new Error('当前没有可下载的跟踪标注视频');
    }

    downloadVideoFile(result.annotatedVideo, `basketball_tracking_${result.annotatedVideo}`);
    Toast.toast.success('已开始下载跟踪标注视频');
  });

  const handleDownloadClip = withErrorHandling(async (clip: HighlightClip) => {
    downloadVideoFile(clip.filename, `basketball_clip_${clip.index}.mp4`);
    Toast.toast.success(`已开始下载片段 ${clip.index}`);
  });

  const downloadClipArchive = async (
    request: {
      filenames?: string[];
      scope?: 'confirmed' | 'debug' | 'all';
    },
    clipCount: number,
    options: {
      emptyMessage: string;
      successTitle: string;
      successMessage: string;
    },
  ) => {
    if (clipCount === 0) {
      throw new Error(options.emptyMessage);
    }

    await ApiService.downloadSelectedClips(taskId, request);

    addNotification({
      type: 'success',
      title: options.successTitle,
      message: options.successMessage,
    });
  };

  const handleDownloadConfirmedClips = withLoading(
    withErrorHandling(async () => {
      await downloadClipArchive(
        { scope: 'confirmed' },
        confirmedClips.length,
        {
          emptyMessage: result?.processingMode === 'manual'
            ? '当前没有可打包下载的手动片段'
            : '当前没有已确认的进球或助攻片段',
          successTitle: result?.processingMode === 'manual'
            ? '手动片段下载完成'
            : '已确认片段下载完成',
          successMessage: result?.processingMode === 'manual'
            ? '已打包下载你手动选择时间点导出的片段'
            : '已打包下载已确认的进球和助攻片段',
        },
      );
    }),
  );

  const handleDownloadReviewClips = withLoading(
    withErrorHandling(async () => {
      await downloadClipArchive(
        { scope: 'debug' },
        reviewClips.length,
        {
          emptyMessage: '当前没有系统补充片段',
          successTitle: '高级排错片段下载完成',
          successMessage: '已打包下载高级排错片段，方便你检查系统是否漏剪',
        },
      );
    }),
  );

  const handleRefresh = withErrorHandling(async () => {
    await refetch();
    Toast.toast.success('已刷新处理结果');
  });

  const handleReuseSource = () => {
    navigate(`/?reuseTaskId=${taskId}`);
  };

  const handlePreviewClip = (clip: HighlightClip) => {
    setActiveClipFilename(clip.filename);
  };

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
                {is404 ? '找不到指定的处理任务，可能任务 ID 已失效。' : '无法获取处理结果，请稍后重试。'}
              </p>
              <div className="flex flex-col justify-center gap-3 sm:flex-row">
                <Button variant="secondary" onClick={() => navigate('/')}>返回首页</Button>
                {!is404 ? (
                  <Button variant="primary" onClick={() => void handleRefresh()}>重新加载</Button>
                ) : null}
              </div>
            </div>
          </EmptyState>
        </Card>
      </div>
    );
  }

  if (isLoading) {
    return <LoadingState type="page" tip="加载结果中..." size="large" />;
  }

  if (!result) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)] px-4">
        <Card className="w-full max-w-xl border border-white/40 bg-white/82 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur">
          <div className="space-y-4 text-center">
            <h1 className="text-3xl font-semibold tracking-tight text-slate-950">结果不可用</h1>
            <p className="text-sm leading-6 text-slate-500">当前任务还没有可展示的处理结果。</p>
            <div className="flex flex-col justify-center gap-3 sm:flex-row">
              <Button variant="secondary" onClick={() => navigate('/')}>返回首页</Button>
              <Button variant="primary" onClick={() => navigate(`/progress/${taskId}`)}>查看进度</Button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  const isManualMode = result.processingMode === 'manual';

  if (isManualMode) {
    const manualMoments = result.manualMoments ?? [];
    const manualClipCount = confirmedClips.length;
    const clipWindowBeforeSeconds = result.pipeline?.export?.clipWindowBeforeSeconds ?? 0;
    const clipWindowAfterSeconds = result.pipeline?.export?.clipWindowAfterSeconds ?? 0;
    const activeClip = confirmedClips.find((clip) => clip.filename === activeClipFilename) ?? confirmedClips[0] ?? null;
    const activeClipIndex = activeClip ? confirmedClips.findIndex((clip) => clip.filename === activeClip.filename) : -1;
    const canPreviewPreviousClip = activeClipIndex > 0;
    const canPreviewNextClip = activeClipIndex >= 0 && activeClipIndex < confirmedClips.length - 1;
    const previewAdjacentClip = (offset: number) => {
      if (activeClipIndex < 0) {
        return;
      }

      const nextClip = confirmedClips[activeClipIndex + offset];
      if (nextClip) {
        setActiveClipFilename(nextClip.filename);
      }
    };

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
                <div className="flex flex-wrap gap-2">
                  <Chip variant="soft" color="success">手动剪片结果</Chip>
                  <Chip variant="soft" color="warning">已导出 {manualClipCount} 个片段</Chip>
                </div>
                <div className="space-y-3">
                  <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl">
                    手动剪片结果
                  </h1>
                  <p className="max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                    这里不再展示人物归因、跟踪和高级排错。你只需要验收自己选择时间点导出的片段，边界不合适就回首页调整。
                  </p>
                </div>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row">
                <Button variant="ghost" onClick={() => navigate('/')}>
                  <span className="inline-flex items-center gap-2">
                    <ArrowLeft size={16} />
                    返回首页
                  </span>
                </Button>
                <Button variant="ghost" onClick={() => void handleRefresh()}>
                  <span className="inline-flex items-center gap-2">
                    <RefreshCcw size={16} />
                    刷新
                  </span>
                </Button>
                <Button variant="secondary" onClick={handleReuseSource}>
                  <span className="inline-flex items-center gap-2">
                    <RefreshCcw size={16} />
                    继续调整时间点
                  </span>
                </Button>
                <Button variant="primary" isDisabled={actionLoading || manualClipCount === 0} onClick={() => void handleDownloadConfirmedClips()}>
                  <span className="inline-flex items-center gap-2">
                    <Download size={16} />
                    {actionLoading ? '下载中...' : `下载片段 ZIP (${manualClipCount})`}
                  </span>
                </Button>
              </div>
            </div>

            <Card className="mb-6 border border-white/40 bg-white/82 p-5 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
              <div className="grid gap-4 lg:grid-cols-[minmax(0,1.4fr)_repeat(4,minmax(0,1fr))]">
                <div className="space-y-2">
                  <div className="text-sm uppercase tracking-[0.18em] text-slate-400">说明</div>
                  <h2 className="text-2xl font-semibold tracking-tight text-slate-950">这次结果完全来自你手动选择的时间点</h2>
                  <p className="text-sm leading-6 text-slate-600">
                    {result.diagnostics?.summary || result.message || '系统没有再做自动找球或人物归因。'}
                  </p>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="text-sm text-slate-500">主交付片段</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{manualClipCount}</div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="text-sm text-slate-500">已选时间点</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{manualMoments.length}</div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="text-sm text-slate-500">前置保留</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{clipWindowBeforeSeconds.toFixed(1)}s</div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                  <div className="text-sm text-slate-500">后置保留</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{clipWindowAfterSeconds.toFixed(1)}s</div>
                </div>
              </div>
            </Card>

            <div className="space-y-6">
              {activeClip ? (
                <Card className="border border-white/40 bg-white/82 p-4 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
                  <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-slate-900">
                        <CheckSquare size={18} className="text-orange-500" />
                        <h2 className="text-xl font-semibold tracking-tight">导出片段预览</h2>
                      </div>
                      <p className="text-sm leading-6 text-slate-500">
                        先逐个看片段边界是否合适。只要发现起止不对，就回首页调整时间点或前后保留时间。
                      </p>
                    </div>
                    {result.highlightVideo ? (
                      <Button variant="secondary" isDisabled={actionLoading} onClick={() => void handleDownload()}>
                        <span className="inline-flex items-center gap-2">
                          <Download size={16} />
                          下载拼接视频
                        </span>
                      </Button>
                    ) : null}
                  </div>

                  <div className="mb-6 space-y-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                      <div className="space-y-2">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-lg font-semibold text-slate-900">
                            当前预览：片段 {activeClip.index}
                          </h3>
                          <Chip color="warning" variant="soft">手动片段</Chip>
                        </div>
                        <p className="text-sm leading-6 text-slate-500">
                          {activeClip.start.toFixed(2)}s - {activeClip.end.toFixed(2)}s · {activeClip.duration.toFixed(2)}s
                        </p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Button variant="ghost" isDisabled={!canPreviewPreviousClip} onClick={() => previewAdjacentClip(-1)}>
                          <span className="inline-flex items-center gap-2">
                            <ChevronLeft size={15} />
                            上一段
                          </span>
                        </Button>
                        <Button variant="ghost" isDisabled={!canPreviewNextClip} onClick={() => previewAdjacentClip(1)}>
                          <span className="inline-flex items-center gap-2">
                            下一段
                            <ChevronRight size={15} />
                          </span>
                        </Button>
                        <Button variant="secondary" onClick={() => void handleDownloadClip(activeClip)}>
                          <span className="inline-flex items-center gap-2">
                            <Download size={15} />
                            下载当前片段
                          </span>
                        </Button>
                      </div>
                    </div>

                    <VideoPlayer
                      src={ApiService.getStreamUrl(activeClip.filename)}
                      title={`片段 ${activeClip.index} 预览`}
                      onDownload={() => void handleDownloadClip(activeClip)}
                    />
                  </div>

                  <div className="space-y-3">
                    {confirmedClips.map((clip) => {
                      const isActiveClip = activeClip.filename === clip.filename;
                      return (
                        <div
                          key={clip.filename}
                          className={`rounded-2xl border p-4 transition ${
                            isActiveClip ? 'border-orange-200 bg-orange-50/70' : 'border-slate-200 bg-slate-50/80'
                          }`}
                        >
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <div className="font-medium text-slate-900">片段 {clip.index}</div>
                              <div className="mt-1 text-sm text-slate-500">
                                {clip.start.toFixed(2)}s - {clip.end.toFixed(2)}s · {clip.duration.toFixed(2)}s
                              </div>
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                              <Chip color="warning" variant="soft">手动片段</Chip>
                              <Button
                                variant={isActiveClip ? 'secondary' : 'ghost'}
                                onClick={() => handlePreviewClip(clip)}
                              >
                                <span className="inline-flex items-center gap-2">
                                  <Eye size={15} />
                                  {isActiveClip ? '正在预览' : '预览片段'}
                                </span>
                              </Button>
                              <Button variant="ghost" onClick={() => void handleDownloadClip(clip)}>
                                <span className="inline-flex items-center gap-2">
                                  <Download size={15} />
                                  单独下载
                                </span>
                              </Button>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </Card>
              ) : (
                <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                  <div className="space-y-3">
                    <h2 className="text-xl font-semibold tracking-tight text-slate-950">当前没有可预览片段</h2>
                    <p className="text-sm leading-6 text-slate-600">
                      这次导出没有生成片段。先回首页确认时间点是否正确，或者适当加大前后保留时间。
                    </p>
                  </div>
                </Card>
              )}

              <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                <div className="space-y-4">
                  <div className="text-sm uppercase tracking-[0.18em] text-slate-400">下一步</div>
                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm leading-6 text-slate-700">
                    {result.diagnostics?.recommendedActions?.[0] || '如果片段起止不合适，回首页调整前后保留时间或增删时间点后再重跑。'}
                  </div>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
  const relatedHighlights = result.relatedHighlights ?? targetTimestamps.length;
  const confirmedHighlights = result.selectionSummary?.confirmed ?? Math.max(targetScores + targetAssists, 0);
  const diagnostics = result.diagnostics;
  const pipelineAttribution = result.pipeline?.attribution;
  const diagnosticsOutcome = diagnostics?.outcome;
  const userTargetPlayerBox = result.targetPlayerBox ?? null;
  const effectiveTargetPlayerBox = result.effectiveTargetPlayerBox ?? null;
  const userSelectionTime = userTargetPlayerBox?.selectionTime;
  const effectiveSelectionTime = effectiveTargetPlayerBox?.selectionTime;
  const systemShiftedTrackingStart = (
    typeof userSelectionTime === 'number'
    && typeof effectiveSelectionTime === 'number'
    && Math.abs(userSelectionTime - effectiveSelectionTime) >= 0.01
  );
  const trackingCoveragePercent = (((pipelineAttribution?.trackingCoverage ?? result.tracking?.coverage) ?? 0) * 100);
  const hasTrackingRisk = Boolean(result.tracking?.enabled) && trackingCoveragePercent < 55;
  const groupedClips = CLIP_GROUPS.map((group) => ({
    ...group,
    clips: availableClips.filter((clip) => clip.highlightRole === group.key),
  })).filter((group) => group.clips.length > 0);
  const confirmedGroups = groupedClips.filter((group) => group.key !== 'possible');
  const reviewGroup = groupedClips.find((group) => group.key === 'possible') ?? null;
  const activeClip = availableClips.find((clip) => clip.filename === activeClipFilename) ?? availableClips[0] ?? null;
  const activeClipIndex = activeClip ? availableClips.findIndex((clip) => clip.filename === activeClip.filename) : -1;
  const activeClipDisplay = activeClip ? getClipDisplay(activeClip) : null;
  const canPreviewPreviousClip = activeClipIndex > 0;
  const canPreviewNextClip = activeClipIndex >= 0 && activeClipIndex < availableClips.length - 1;
  const previewAdjacentClip = (offset: number) => {
    if (activeClipIndex < 0) {
      return;
    }

    const nextClip = availableClips[activeClipIndex + offset];
    if (nextClip) {
      setActiveClipFilename(nextClip.filename);
    }
  };
  const acceptanceState = (() => {
    if (diagnosticsOutcome === 'global_makes_without_target') {
      return {
        tone: 'warning' as const,
        title: '当前没有稳定锁定到目标球员',
        description: '系统检测到了全场进球，但目前没有足够证据把这些回合稳定归因到你框选的人。建议先重新选择更早、更清晰的截图再重跑；只有在怀疑跟踪跑偏时，再看标注视频。',
      };
    }
    if (hasTrackingRisk) {
      return {
        tone: 'warning' as const,
        title: '当前人物跟踪还不够稳定',
        description: '先验收已确认片段；如果你怀疑漏剪或人物跟丢，再展开高级排错区查看系统补充片段和标注视频。',
      };
    }
    if (possibleHighlights > 0 && confirmedHighlights === 0) {
      return {
        tone: 'warning' as const,
        title: '当前还没有拿到可直接交付的确认片段',
        description: `系统暂时只保留了 ${possibleHighlights} 个高级排错回合，方便你检查是否漏判了目标球员的进球或助攻。`,
      };
    }
    if (possibleHighlights > 0) {
      return {
        tone: 'accent' as const,
        title: `先下载 ${confirmedHighlights} 个已确认片段`,
        description: `当前已经确认了 ${confirmedHighlights} 个片段；另外有 ${possibleHighlights} 个系统补充回合被移到了高级排错区，只有在你想继续排查漏剪时再看。`,
      };
    }
    return {
      tone: 'default' as const,
      title: `可以直接验收这 ${confirmedHighlights} 个已确认片段`,
      description: '当前已经没有额外的系统补充回合。直接下载已确认片段 ZIP，检查这些进球和助攻是否满足要求即可。',
    };
  })();
  const annotatedVideoReasonLabel = (() => {
    switch (result.annotatedVideoReason) {
      case 'tracking_low_coverage':
        return '当前因为跟踪覆盖率偏低，系统已自动保留标注视频。';
      case 'highlight_review':
        return '当前因为存在系统补充片段，系统已自动保留标注视频。';
      case 'risk_review':
        return '当前因为结果存在复核风险，系统已自动保留标注视频。';
      case 'debug':
        return '当前处于调试模式，系统保留了完整标注视频。';
      default:
        return '用来核对系统是否一直跟着目标球员。蓝橙色状态代表已丢失或重新找回目标，稳定跟踪时会显示绿色目标框。';
    }
  })();
  const hasAuxiliaryReviewArtifacts = Boolean(result.highlightVideo || result.annotatedVideo || diagnostics);
  const deliveryModeLabel = getDeliveryModeLabel({
    confirmedHighlights,
    possibleHighlights,
    diagnosticsOutcome,
  });
  const acceptanceChecklist = dedupeMessages([
    confirmedClips.length > 0 && reviewClips.length > 0
      ? '先下载已确认片段 ZIP；只有怀疑漏剪时，再展开高级排错区继续看系统补充片段。'
      : availableClips.length > 0
        ? confirmedClips.length > 0
          ? '先下载已确认片段 ZIP，按 score / assist 文件夹直接验收。'
          : '当前没有 confirmed 主交付，先展开高级排错区检查系统补充片段。'
        : null,
    reviewClips.length > 0 && confirmedClips.length === 0
      ? '当前只有高级排错片段，说明系统还没有稳定确认到目标球员的进球或助攻。'
      : null,
    possibleHighlights > 0
      ? '系统补充片段只是兜底，不是主交付。'
      : null,
    result.annotatedVideo && (hasTrackingRisk || diagnosticsOutcome === 'global_makes_without_target')
      ? '如果你怀疑人物跟丢或明显漏剪，再打开跟踪标注视频核对。'
      : null,
    ...((diagnostics?.recommendedActions ?? []).map((action) => normalizeRecommendedAction(action, {
      hasConfirmedClips: confirmedClips.length > 0,
      hasReviewClips: reviewClips.length > 0,
    }))),
  ]);
  const normalizedRecommendedActions = (diagnostics?.recommendedActions ?? [])
    .map((action) => normalizeRecommendedAction(action, {
      hasConfirmedClips: confirmedClips.length > 0,
      hasReviewClips: reviewClips.length > 0,
    }))
    .filter((action): action is string => Boolean(action));
  const primaryDownloadDescription = confirmedClips.length > 0
    ? '这是当前最值得先验收的结果。'
    : reviewClips.length > 0
      ? '当前没有 confirmed 主交付，只剩高级排错片段可供你检查。'
      : '当前没有可下载片段。';
  const clipSectionTitle = confirmedClips.length > 0 ? '自动导出的已确认片段' : '自动导出的高级排错片段';
  const renderClipGroup = (group: typeof groupedClips[number]) => (
    <div key={group.key} className="space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-slate-900">{group.title}</h3>
            <Chip color={group.chipColor} variant="soft">{group.clips.length}</Chip>
          </div>
          <p className="mt-1 text-sm text-slate-500">{group.description}</p>
        </div>
      </div>

      {group.clips.map((clip) => {
        const clipDisplay = getClipDisplay(clip);
        const isActiveClip = activeClip?.filename === clip.filename;
        return (
          <div
            key={clip.filename}
            className={`rounded-2xl border p-4 transition ${
              isActiveClip ? 'border-orange-200 bg-orange-50/70' : 'border-slate-200 bg-slate-50/80'
            }`}
          >
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="font-medium text-slate-900">片段 {clip.index}</div>
                <div className="mt-1 text-sm text-slate-500">
                  {clip.start.toFixed(2)}s - {clip.end.toFixed(2)}s · {clip.duration.toFixed(2)}s
                </div>
                {clip.highlightRole === 'possible' && clip.candidateReason ? (
                  <div className="mt-1 text-sm text-orange-600">
                    {getCandidateReasonLabel(clip.candidateReason) || '系统补充回合'}
                  </div>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Chip color={clipDisplay.color} variant="soft">{clipDisplay.label}</Chip>
                {typeof clip.highlightConfidence === 'number' && clip.highlightConfidence > 0 ? (
                  <Chip color="accent" variant="soft">
                    置信度 {(clip.highlightConfidence * 100).toFixed(0)}%
                  </Chip>
                ) : null}
                {clip.highlightRole === 'possible' && clip.candidateSource ? (
                  <Chip color="warning" variant="soft">
                    {getCandidateSourceLabel(clip.candidateSource) || '系统补充来源'}
                  </Chip>
                ) : null}
                <Button
                  variant={isActiveClip ? 'secondary' : 'ghost'}
                  onClick={() => handlePreviewClip(clip)}
                >
                  <span className="inline-flex items-center gap-2">
                    <Eye size={15} />
                    {isActiveClip ? '正在预览' : '预览片段'}
                  </span>
                </Button>
                <Button variant="ghost" onClick={() => void handleDownloadClip(clip)}>
                  <span className="inline-flex items-center gap-2">
                    <Download size={15} />
                    单独下载
                  </span>
                </Button>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );

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
              <div className="flex flex-wrap gap-2">
                <Chip variant="soft" color="success">片段结果已就绪</Chip>
                <Chip variant="soft" color={possibleHighlights > 0 ? 'accent' : 'default'}>
                  {deliveryModeLabel}
                </Chip>
                {hasTrackingRisk ? (
                  <Chip variant="soft" color="warning">跟踪需重点复核</Chip>
                ) : null}
              </div>
              <div className="space-y-3">
                <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl">
                  自动剪辑结果
                </h1>
                <p className="max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                  先验收已确认的进球和助攻片段；只有在你怀疑漏剪时，才需要继续看系统补充片段。拼接视频和标注视频都只是附加材料。
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row">
              <Button variant="ghost" onClick={() => navigate('/')}>
                <span className="inline-flex items-center gap-2">
                  <ArrowLeft size={16} />
                  返回首页
                </span>
              </Button>
              <Button variant="ghost" onClick={() => void handleRefresh()}>
                <span className="inline-flex items-center gap-2">
                  <RefreshCcw size={16} />
                  刷新
                </span>
              </Button>
              <Button variant="secondary" onClick={handleReuseSource}>
                <span className="inline-flex items-center gap-2">
                  <RefreshCcw size={16} />
                  重新框选并重跑
                </span>
              </Button>
              {confirmedClips.length > 0 ? (
                <Button variant="primary" isDisabled={actionLoading} onClick={() => void handleDownloadConfirmedClips()}>
                  <span className="inline-flex items-center gap-2">
                    <Download size={16} />
                    {actionLoading ? '下载中...' : `下载已确认片段 (${confirmedClips.length})`}
                  </span>
                </Button>
              ) : reviewClips.length > 0 ? (
                <Button variant="secondary" isDisabled={actionLoading} onClick={() => setShowSupplementalClips(true)}>
                  <span className="inline-flex items-center gap-2">
                    <Eye size={16} />
                    打开高级排错区
                  </span>
                </Button>
              ) : null}
            </div>
          </div>

          <Card className={`mb-6 border p-5 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur ${
            acceptanceState.tone === 'warning'
              ? 'border-orange-200 bg-orange-50/85'
              : acceptanceState.tone === 'accent'
                ? 'border-indigo-200 bg-indigo-50/85'
              : 'border-white/40 bg-white/82'
          }`}>
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-2">
                <div className="text-sm uppercase tracking-[0.18em] text-slate-400">交付</div>
                <h2 className="text-2xl font-semibold tracking-tight text-slate-950">
                  {acceptanceState.title}
                </h2>
                <p className="max-w-3xl text-sm leading-6 text-slate-600">
                  {acceptanceState.description}
                </p>
                {result.annotatedVideo && (hasTrackingRisk || diagnosticsOutcome === 'global_makes_without_target') ? (
                  <div className="pt-1">
                    <Button variant="secondary" onClick={() => void handleDownloadAnnotated()}>
                      <span className="inline-flex items-center gap-2">
                        <Download size={16} />
                        下载跟踪标注视频
                      </span>
                    </Button>
                  </div>
                ) : null}
              </div>

              <div className="grid gap-3 sm:grid-cols-3 lg:min-w-[360px]">
                <div className="rounded-2xl border border-slate-200 bg-white/80 p-4">
                  <div className="text-sm text-slate-500">主交付片段</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{relatedHighlights}</div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white/80 p-4">
                  <div className="text-sm text-slate-500">可直接验收</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{confirmedHighlights}</div>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-white/80 p-4">
                  <div className="text-sm text-slate-500">高级排错回合</div>
                  <div className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{possibleHighlights}</div>
                </div>
              </div>
            </div>
          </Card>

          <div className="space-y-6">
              {availableClips.length > 0 ? (
                <Card className="border border-white/40 bg-white/82 p-4 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
                  <div className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-slate-900">
                        <CheckSquare size={18} className="text-orange-500" />
                        <h2 className="text-xl font-semibold tracking-tight">{clipSectionTitle}</h2>
                      </div>
                      <p className="text-sm leading-6 text-slate-500">
                        {primaryDownloadDescription} 助攻片段会尽量从目标球员的控球或传球动作前开始；系统补充片段不会再混入主交付。
                      </p>
                      <p className="text-sm leading-6 text-slate-500">
                        主交付 ZIP 只包含 `score / assist`。只有你主动进入高级排错区时，才会单独下载系统补充片段和查看 `manifest.json`。
                      </p>
                      <div className="flex flex-wrap gap-2">
                        <Chip color="success" variant="soft">目标球员进球 {targetScores}</Chip>
                        <Chip color="warning" variant="soft">目标球员助攻 {targetAssists}</Chip>
                        <Chip color="accent" variant="soft">系统补充 {possibleHighlights}</Chip>
                      </div>
                    </div>
                    <div className="flex flex-col gap-2 sm:items-end">
                      {confirmedClips.length > 0 ? (
                        <Button
                          variant="primary"
                          isDisabled={actionLoading}
                          onClick={() => void handleDownloadConfirmedClips()}
                        >
                          <span className="inline-flex items-center gap-2">
                            <Download size={16} />
                            下载已确认片段 ZIP
                          </span>
                        </Button>
                      ) : null}
                      {confirmedClips.length === 0 && reviewClips.length > 0 ? (
                        <Button
                          variant="secondary"
                          onClick={() => setShowSupplementalClips(true)}
                        >
                          <span className="inline-flex items-center gap-2">
                            <Eye size={16} />
                            查看高级排错区
                          </span>
                        </Button>
                      ) : null}
                    </div>
                  </div>

                  {activeClip ? (
                    <div className="mb-6 space-y-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                        <div className="space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <h3 className="text-lg font-semibold text-slate-900">
                              当前预览：片段 {activeClip.index}
                            </h3>
                            {activeClipDisplay ? (
                              <Chip color={activeClipDisplay.color} variant="soft">
                                {activeClipDisplay.label}
                              </Chip>
                            ) : null}
                            {typeof activeClip.highlightConfidence === 'number' && activeClip.highlightConfidence > 0 ? (
                              <Chip color="accent" variant="soft">
                                置信度 {(activeClip.highlightConfidence * 100).toFixed(0)}%
                              </Chip>
                            ) : null}
                          </div>
                          <p className="text-sm leading-6 text-slate-500">
                            {activeClip.start.toFixed(2)}s - {activeClip.end.toFixed(2)}s · {activeClip.duration.toFixed(2)}s
                          </p>
                          {activeClip.highlightRole === 'possible' && activeClip.candidateReason ? (
                            <p className="text-sm leading-6 text-orange-600">
                              {getCandidateReasonLabel(activeClip.candidateReason) || '系统补充回合'}
                            </p>
                          ) : null}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Button
                            variant="ghost"
                            isDisabled={!canPreviewPreviousClip}
                            onClick={() => previewAdjacentClip(-1)}
                          >
                            <span className="inline-flex items-center gap-2">
                              <ChevronLeft size={15} />
                              上一段
                            </span>
                          </Button>
                          <Button
                            variant="ghost"
                            isDisabled={!canPreviewNextClip}
                            onClick={() => previewAdjacentClip(1)}
                          >
                            <span className="inline-flex items-center gap-2">
                              下一段
                              <ChevronRight size={15} />
                            </span>
                          </Button>
                          <Button variant="secondary" onClick={() => void handleDownloadClip(activeClip)}>
                            <span className="inline-flex items-center gap-2">
                              <Download size={15} />
                              下载当前片段
                            </span>
                          </Button>
                        </div>
                      </div>

                      <VideoPlayer
                        src={ApiService.getStreamUrl(activeClip.filename)}
                        title={`片段 ${activeClip.index} 预览`}
                        onDownload={() => void handleDownloadClip(activeClip)}
                      />
                    </div>
                  ) : null}

                  <div className="space-y-5">
                    {confirmedGroups.length > 0 ? (
                      <div className="space-y-5">
                        <div>
                          <div className="text-xs uppercase tracking-[0.16em] text-slate-400">优先验收</div>
                          <div className="mt-2 text-sm text-slate-600">
                            下面这些片段已经确认归因到目标球员，先看它们。
                          </div>
                        </div>
                        {confirmedGroups.map(renderClipGroup)}
                      </div>
                    ) : null}

                    {reviewGroup ? (
                      <div className="space-y-4 rounded-3xl border border-slate-200 bg-slate-50/70 p-4">
                        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                          <div className="space-y-2">
                            <div className="text-xs uppercase tracking-[0.16em] text-slate-400">高级排错</div>
                            <div className="flex items-center gap-2">
                              <h3 className="text-lg font-semibold text-slate-900">系统补充片段</h3>
                              <Chip color="accent" variant="soft">{reviewGroup.clips.length}</Chip>
                            </div>
                            <p className="text-sm leading-6 text-slate-500">
                              这些片段只用于排查漏剪，不属于主交付。已经有确认片段时，默认不需要先看这里。
                            </p>
                          </div>
                          <div className="flex flex-col gap-2 sm:items-end">
                            <Button
                              variant="secondary"
                              onClick={() => setShowSupplementalClips((current) => !current)}
                            >
                              {showSupplementalClips ? '收起系统补充片段' : '展开系统补充片段'}
                            </Button>
                            <Button
                              variant="ghost"
                              isDisabled={actionLoading}
                              onClick={() => void handleDownloadReviewClips()}
                            >
                              <span className="inline-flex items-center gap-2">
                                <Download size={16} />
                                下载高级排错片段 ZIP
                              </span>
                            </Button>
                          </div>
                        </div>

                        {showSupplementalClips ? renderClipGroup(reviewGroup) : null}
                      </div>
                    ) : null}
                  </div>
                </Card>
              ) : null}

              {hasAuxiliaryReviewArtifacts ? (
                <Card className="border border-white/40 bg-white/82 p-4 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-2">
                      <div className="text-sm uppercase tracking-[0.18em] text-slate-400">辅助复核</div>
                      <h2 className="text-xl font-semibold tracking-tight text-slate-950">辅助复核材料</h2>
                      <p className="text-sm leading-6 text-slate-500">
                        这里放的是辅助判断材料，不是主交付。只有在你怀疑漏剪、人物跟丢，或者想核对系统为什么补出某些片段时再展开。
                      </p>
                    </div>
                    <Button
                      variant="secondary"
                      onClick={() => setShowAuxiliaryReview((current) => !current)}
                    >
                      {showAuxiliaryReview ? '收起辅助复核' : '展开辅助复核'}
                    </Button>
                  </div>

                  {showAuxiliaryReview ? (
                    <div className="mt-6 space-y-6">
                      {result.highlightVideo ? (
                        <VideoPlayer
                          src={ApiService.getStreamUrl(result.highlightVideo)}
                          title="拼接视频（仅已确认片段，可选）"
                          onDownload={() => void handleDownload()}
                        />
                      ) : (
                        <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                          当前没有额外生成已确认片段拼接视频，片段 ZIP 仍然是主交付。
                        </div>
                      )}

                      {result.annotatedVideo ? (
                        <div className="space-y-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                            <div className="space-y-2">
                              <div className="flex items-center gap-2 text-slate-900">
                                <Eye size={18} className="text-orange-500" />
                                <h3 className="text-lg font-semibold tracking-tight">跟踪标注视频（可选）</h3>
                              </div>
                              <p className="text-sm leading-6 text-slate-500">
                                {annotatedVideoReasonLabel}
                              </p>
                            </div>
                            <Button variant="ghost" onClick={() => void handleDownloadAnnotated()}>
                              <span className="inline-flex items-center gap-2">
                                <Download size={16} />
                                下载标注视频
                              </span>
                            </Button>
                          </div>

                          <VideoPlayer
                            src={ApiService.getStreamUrl(result.annotatedVideo)}
                            title="跟踪标注视频"
                          />
                        </div>
                      ) : null}

                      {diagnostics ? (
                        <div className="space-y-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                          <div>
                            <div className="text-sm uppercase tracking-[0.18em] text-slate-400">诊断说明</div>
                            <h3 className="mt-2 text-lg font-semibold tracking-tight text-slate-950">辅助诊断（可选）</h3>
                          </div>
                          <div className="rounded-2xl border border-slate-200 bg-white/80 p-4 text-sm leading-6 text-slate-700">
                            {diagnostics.summary || result.message || '当前没有额外诊断信息'}
                          </div>
                          {diagnostics.reasons?.length ? (
                            <div className="space-y-2">
                              {diagnostics.reasons.map((reason) => (
                                <div key={reason} className="rounded-2xl border border-slate-200 bg-white/80 px-4 py-3 text-sm text-slate-700">
                                  {reason}
                                </div>
                              ))}
                            </div>
                          ) : null}
                          {normalizedRecommendedActions.length > 0 ? (
                            <div className="space-y-2">
                              {normalizedRecommendedActions.map((action) => (
                                <div key={action} className="rounded-2xl border border-orange-100 bg-orange-50/70 px-4 py-3 text-sm text-slate-700">
                                  {action}
                                </div>
                              ))}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </Card>
              ) : null}

              <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                <div className="space-y-5">
                  <div className="space-y-2">
                    <div className="text-sm uppercase tracking-[0.18em] text-slate-400">验收建议</div>
                    <h2 className="text-xl font-semibold tracking-tight text-slate-950">验收建议（可选）</h2>
                    <p className="text-sm leading-6 text-slate-500">
                      这里不再展示内部实现细节，只保留当前结果为什么这样交付，以及你下一步该怎么验收。
                    </p>
                  </div>

                  <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                    <div className="text-xs uppercase tracking-[0.16em] text-slate-400">当前说明</div>
                    <div className="mt-2 text-sm leading-6 text-slate-700">
                      {diagnostics?.summary || result.message || '已成功生成相关片段结果'}
                    </div>
                  </div>

                  {userTargetPlayerBox ? (
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">起始帧说明</div>
                      <div className="mt-2 space-y-2 text-sm leading-6 text-slate-700">
                        <div>
                          你选择的起始帧：
                          <span className="ml-1 font-medium text-slate-900">
                            {typeof userSelectionTime === 'number'
                              ? `${formatDuration(userSelectionTime)} (${userSelectionTime.toFixed(2)}s)`
                              : '未记录'}
                          </span>
                        </div>
                        {systemShiftedTrackingStart ? (
                          <div>
                            系统实际追踪起点：
                            <span className="ml-1 font-medium text-slate-900">
                              {formatDuration(effectiveSelectionTime)} ({effectiveSelectionTime.toFixed(2)}s)
                            </span>
                            <span className="ml-2 text-slate-500">
                              为了补抓更早回合，系统自动把追踪起点前移了一些。
                            </span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}

                  {acceptanceChecklist.length > 0 ? (
                    <div className="space-y-2">
                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">建议顺序</div>
                      {acceptanceChecklist.map((item) => (
                        <div key={item} className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm leading-6 text-slate-700">
                          {item}
                        </div>
                      ))}
                    </div>
                  ) : null}

                  {diagnostics?.reasons?.length ? (
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">补充说明</div>
                      <div className="mt-2 space-y-2">
                        {diagnostics.reasons.map((reason) => (
                          <div key={reason} className="rounded-2xl border border-white/70 bg-white/85 px-4 py-3 text-sm leading-6 text-slate-700">
                            {reason}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {result.annotatedVideo ? (
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                      <div className="text-xs uppercase tracking-[0.16em] text-slate-400">标注视频保留原因</div>
                      <div className="mt-2 text-sm leading-6 text-slate-700">
                        {annotatedVideoReasonLabel}
                      </div>
                    </div>
                  ) : null}
                </div>
              </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Result;
