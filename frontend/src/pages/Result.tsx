import React, { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { Button, Card, Chip, EmptyState, Input, Modal, Tabs, Toast } from '@heroui/react';
import { ArrowLeft, Copy, Download, Eye, RefreshCcw, Share2 } from 'lucide-react';
import { VideoPlayer } from '@/components/result/VideoPlayer';
import { ResultStats } from '@/components/result/ResultStats';
import { ErrorAlert, LoadingState } from '@/components/common';
import { useErrorHandler, useLoading } from '@/hooks';
import { ApiService } from '@/services/api';
import { useTaskResult } from '@/services';
import { HttpRequestError } from '@/services/http';
import { useAppStore } from '@/store/app';
import type { ShotTimestamp } from '@/types';

const getHighlightDisplay = (timestamp: ShotTimestamp) => {
  if (timestamp.highlight_role === 'assist') {
    return { label: '你的助攻', color: 'warning' as const };
  }
  if (timestamp.highlight_role === 'score' || timestamp.owner === 'target') {
    return { label: '你的进球', color: 'success' as const };
  }
  return { label: '全场进球', color: 'default' as const };
};

const isNotFoundError = (error: Error) => {
  const status = (error as HttpRequestError).status;
  return status === 404 || error.message.includes('资源不存在') || error.message.includes('404');
};

export const Result: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const taskId = fileId ?? '';
  const navigate = useNavigate();
  const { addNotification } = useAppStore();
  const [shareModalVisible, setShareModalVisible] = useState(false);
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

  const shareUrl = `${window.location.origin}/result/${taskId}`;

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
        throw new Error('当前没有可下载的个人高光视频');
      }

      downloadVideoFile(result.highlightVideo, `basketball_highlight_${result.highlightVideo}`);

      addNotification({
        type: 'success',
        title: '下载开始',
        message: '个人高光视频下载已开始',
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

  const handleCopyLink = withErrorHandling(async () => {
    await navigator.clipboard.writeText(shareUrl);
    Toast.toast.success('链接已复制到剪贴板');
  });

  const handleRefresh = withErrorHandling(async () => {
    await refetch();
    Toast.toast.success('已刷新处理结果');
  });

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

  const targetTimestamps = result.timestamps ?? [];
  const allMadeTimestamps = result.allMadeTimestamps ?? [];
  const targetScores =
    result.targetScores ??
    result.targetShots ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'score' || timestamp.owner === 'target').length;
  const targetAssists =
    result.targetAssists ??
    targetTimestamps.filter((timestamp) => timestamp.highlight_role === 'assist').length;
  const targetHighlights = result.targetHighlights ?? targetTimestamps.length;
  const trackingStatusLabel = (() => {
    switch (result.tracking?.latestStatus) {
      case 'tracking':
        return '稳定跟踪';
      case 'revalidated':
        return '已重新校准';
      case 'reacquired':
        return '已重新找回目标';
      case 'lost':
        return '当前已丢失目标';
      case 'initialized':
        return '已初始化';
      case 'disabled':
        return '未启用';
      default:
        return result.tracking?.enabled ? '已启用目标跟踪' : '未启用';
    }
  })();

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
              <Chip variant="soft" color="success">结果已就绪</Chip>
              <div className="space-y-3">
                <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl">
                  个人高光结果
                </h1>
                <p className="max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                  当前展示的是归因到你的个人高光结果，包含你的进球和助攻。
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
              <Button variant="ghost" onClick={() => void handleRefresh()}>
                <span className="inline-flex items-center gap-2">
                  <RefreshCcw size={16} />
                  刷新
                </span>
              </Button>
              <Button variant="secondary" onClick={() => setShareModalVisible(true)}>
                <span className="inline-flex items-center gap-2">
                  <Share2 size={16} />
                  分享
                </span>
              </Button>
              <Button variant="primary" isDisabled={!result.highlightVideo || actionLoading} onClick={() => void handleDownload()}>
                <span className="inline-flex items-center gap-2">
                  <Download size={16} />
                  {actionLoading ? '下载中...' : '下载'}
                </span>
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2 space-y-6">
              <VideoPlayer
                src={result.highlightVideo ? ApiService.getStreamUrl(result.highlightVideo) : ''}
                title="个人高光视频"
                onDownload={result.highlightVideo ? () => void handleDownload() : undefined}
                onShare={() => setShareModalVisible(true)}
                className={!result.highlightVideo ? 'hidden' : ''}
              />

              {!result.highlightVideo ? (
                <Card className="border border-white/40 bg-white/82 p-8 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                  <div className="text-center text-sm text-slate-500">
                    {result.message || '当前没有生成个人高光视频'}
                  </div>
                </Card>
              ) : null}

              {result.annotatedVideo ? (
                <Card className="border border-white/40 bg-white/82 p-4 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
                  <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-slate-900">
                        <Eye size={18} className="text-orange-500" />
                        <h2 className="text-xl font-semibold tracking-tight">跟踪标注视频</h2>
                      </div>
                      <p className="text-sm leading-6 text-slate-500">
                        用来核对系统是否一直跟着你自己。蓝橙色状态代表已丢失或重新找回目标，稳定跟踪时会显示绿色目标框。
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
                </Card>
              ) : null}

              <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                <Tabs defaultSelectedKey="target-shots">
                  <div className="mb-6 flex flex-wrap gap-2">
                    <Tabs.List>
                      <Tabs.Tab id="target-shots">你的高光 ({targetHighlights})</Tabs.Tab>
                      <Tabs.Tab id="all-made-shots">全场进球 ({allMadeTimestamps.length})</Tabs.Tab>
                      <Tabs.Tab id="task-info">任务信息</Tabs.Tab>
                    </Tabs.List>
                  </div>

                  <Tabs.Panel id="target-shots">
                    <div className="space-y-3">
                      {targetTimestamps.length > 0 ? targetTimestamps.map((timestamp, index) => (
                        <div key={`${timestamp.frame}-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <div className="font-medium text-slate-900">镜头 {index + 1}</div>
                              <div className="mt-1 text-sm text-slate-500">
                                {timestamp.timestamp.toFixed(2)}s · 帧 {timestamp.frame}
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Chip color={getHighlightDisplay(timestamp).color} variant="soft">
                                {getHighlightDisplay(timestamp).label}
                              </Chip>
                              {typeof timestamp.highlight_confidence === 'number' && timestamp.highlight_confidence > 0 ? (
                                <Chip color="accent" variant="soft">
                                  高光置信度 {(timestamp.highlight_confidence * 100).toFixed(0)}%
                                </Chip>
                              ) : null}
                              {timestamp.target_visible ? <Chip color="default" variant="soft">目标可见</Chip> : null}
                            </div>
                          </div>
                        </div>
                      )) : (
                        <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 px-4 py-8 text-center text-sm text-slate-500">
                          {result.message || '当前没有归因到你的个人高光'}
                        </div>
                      )}
                    </div>
                  </Tabs.Panel>

                  <Tabs.Panel id="all-made-shots">
                    <div className="space-y-3">
                      {allMadeTimestamps.length > 0 ? allMadeTimestamps.map((timestamp, index) => (
                        <div key={`${timestamp.frame}-${index}`} className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <div className="font-medium text-slate-900">全场进球 {index + 1}</div>
                              <div className="mt-1 text-sm text-slate-500">
                                {timestamp.timestamp.toFixed(2)}s · 帧 {timestamp.frame}
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Chip color={getHighlightDisplay(timestamp).color} variant="soft">
                                {getHighlightDisplay(timestamp).label}
                              </Chip>
                              {typeof timestamp.highlight_confidence === 'number' && timestamp.highlight_confidence > 0 ? (
                                <Chip color="accent" variant="soft">
                                  高光置信度 {(timestamp.highlight_confidence * 100).toFixed(0)}%
                                </Chip>
                              ) : typeof timestamp.owner_confidence === 'number' ? (
                                <Chip color="accent" variant="soft">
                                  出手归因 {(timestamp.owner_confidence * 100).toFixed(0)}%
                                </Chip>
                              ) : null}
                            </div>
                          </div>
                        </div>
                      )) : (
                        <div className="rounded-2xl border border-dashed border-slate-200 bg-slate-50/60 px-4 py-8 text-center text-sm text-slate-500">
                          当前没有检测到全场进球
                        </div>
                      )}
                    </div>
                  </Tabs.Panel>

                  <Tabs.Panel id="task-info">
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">任务 ID</div>
                        <div className="mt-2 break-all font-mono text-sm text-slate-700">{taskId}</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">跟踪状态</div>
                        <div className="mt-2 text-sm text-slate-700">{trackingStatusLabel}</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">你的进球</div>
                        <div className="mt-2 text-sm text-slate-700">{targetScores}</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">你的助攻</div>
                        <div className="mt-2 text-sm text-slate-700">{targetAssists}</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">跟踪覆盖率</div>
                        <div className="mt-2 text-sm text-slate-700">{((result.tracking?.coverage ?? 0) * 100).toFixed(1)}%</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">重新找回目标</div>
                        <div className="mt-2 text-sm text-slate-700">{result.tracking?.reacquiredCount ?? 0} 次</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">阻止误切换</div>
                        <div className="mt-2 text-sm text-slate-700">{result.tracking?.guardedSwitches ?? 0} 次</div>
                      </div>
                      <div className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4">
                        <div className="text-xs uppercase tracking-[0.16em] text-slate-400">结果说明</div>
                        <div className="mt-2 text-sm text-slate-700">{result.message || '已成功生成个人高光'}</div>
                      </div>
                    </div>
                  </Tabs.Panel>
                </Tabs>
              </Card>
            </div>

            <div className="lg:col-span-1">
              <ResultStats result={result} />
            </div>
          </div>
        </div>
      </div>

      {shareModalVisible ? (
        <Modal>
          <Modal.Backdrop isDismissable onClick={() => setShareModalVisible(false)} />
          <Modal.Container>
            <Modal.Dialog>
              <Modal.Header>
                <Modal.Heading>分享结果</Modal.Heading>
              </Modal.Header>
              <Modal.Body>
                <div className="space-y-4">
                  <div>
                    <div className="mb-2 text-sm text-slate-500">分享链接</div>
                    <Input value={shareUrl} readOnly />
                  </div>
                  <div>
                    <div className="mb-2 text-sm text-slate-500">分享描述</div>
                    <div className="rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm leading-6 text-slate-600">
                      查看我的篮球个人高光视频，这是基于我选中的出镜画面做人物跟踪后，按我的进球和助攻生成的集锦。
                    </div>
                  </div>
                </div>
              </Modal.Body>
              <Modal.Footer>
                <Button variant="ghost" onClick={() => setShareModalVisible(false)}>取消</Button>
                <Button variant="primary" onClick={() => void handleCopyLink()}>
                  <span className="inline-flex items-center gap-2">
                    <Copy size={16} />
                    复制链接
                  </span>
                </Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal>
      ) : null}
    </div>
  );
};

export default Result;
