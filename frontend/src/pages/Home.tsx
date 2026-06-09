/**
 * 主页面组件
 */
import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Button, Card, Chip } from '@heroui/react';
import { ArrowRight, Radar, Sparkles, Target } from 'lucide-react';
import { VideoUpload } from '@/components/upload/VideoUpload';
import { PlayerSelector } from '@/components/ui/PlayerSelector';
import ErrorAlert from '@/components/common/ErrorAlert';
import { useTaskStore } from '@/store';
import { ApiService, useUploadVideo, useProcessVideo } from '@/services';
import type { PlayerSelectionBox, VideoFile, ReusableVideoSource } from '@/types';
import { syncSelectionBoxToFrame, validateTargetPlayerBox } from '@/utils';
import { useErrorHandler, useLoading } from '@/hooks';

export const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<VideoFile | null>(null);
  const [targetPlayerBox, setTargetPlayerBox] = useState<PlayerSelectionBox | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [reusableSource, setReusableSource] = useState<ReusableVideoSource | null>(null);
  const [isLoadingReusableSource, setIsLoadingReusableSource] = useState(false);
  const hasConfirmedSelectionFrame = Boolean(selectedFile?.selectionFrame && selectedFile.selectionFrameConfirmed);
  const [searchParams, setSearchParams] = useSearchParams();
  const reuseTaskId = searchParams.get('reuseTaskId')?.trim() || '';

  // Store hooks
  const { createTask } = useTaskStore();

  // Custom hooks
  const { error, hasError, clearError, handleError, withErrorHandling } = useErrorHandler();
  const { loading: customLoading, withLoading } = useLoading();

  const navigate = useNavigate();

  // API hooks
  const uploadMutation = useUploadVideo();
  const processMutation = useProcessVideo();

  useEffect(() => {
    let cancelled = false;

    if (!reuseTaskId) {
      setReusableSource(null);
      setIsLoadingReusableSource(false);
      return () => {
        cancelled = true;
      };
    }

    setIsLoadingReusableSource(true);
    clearError();

    void ApiService.getReusableSource(reuseTaskId)
      .then((source) => {
        if (cancelled) {
          return;
        }
        setReusableSource(source);
      })
      .catch((nextError) => {
        if (cancelled) {
          return;
        }
        setReusableSource(null);
        handleError(nextError instanceof Error ? nextError : new Error('加载源视频失败'));
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoadingReusableSource(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [reuseTaskId, clearError, handleError]);

  const clearReusableMode = useCallback((options?: { keepSelectedFile?: boolean }) => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete('reuseTaskId');
    setSearchParams(nextParams, { replace: true });
    setReusableSource(null);

    if (!options?.keepSelectedFile) {
      setSelectedFile(null);
      setTargetPlayerBox(null);
      setUploadProgress(0);
    }
  }, [searchParams, setSearchParams]);

  // 处理文件选择
  const handleFileSelect = useCallback((file: VideoFile | null) => {
    if (file?.file && reuseTaskId) {
      clearReusableMode({ keepSelectedFile: true });
    }
    setSelectedFile(file);
    setTargetPlayerBox(syncSelectionBoxToFrame(file?.targetPlayerBox ?? null, file?.selectionFrame));
    setUploadProgress(0);
  }, [reuseTaskId, clearReusableMode]);

  const handleTargetPlayerChange = useCallback((selection: PlayerSelectionBox | null) => {
    setTargetPlayerBox(syncSelectionBoxToFrame(selection, selectedFile?.selectionFrame));
  }, [selectedFile?.selectionFrame]);

  const handleStartProcessing = withLoading(
    withErrorHandling(
      async () => {
        if (!selectedFile) {
          throw new Error('请先选择视频文件');
        }

        const effectiveTargetPlayerBox = syncSelectionBoxToFrame(
          targetPlayerBox,
          selectedFile.selectionFrame,
        );
        const selectionValidation = validateTargetPlayerBox(effectiveTargetPlayerBox);
        if (!selectionValidation.valid) {
          throw new Error(selectionValidation.error || '人物选区无效');
        }

        setTargetPlayerBox(effectiveTargetPlayerBox);

        setUploadProgress(0);

        let fileId = selectedFile.sourceFileId;

        if (!fileId && selectedFile.file) {
          const uploadResult = await uploadMutation.mutateAsync({
            file: selectedFile.file,
            onProgress: (progress) => {
              setUploadProgress(progress);
            },
          });

          if (!uploadResult.success) {
            throw new Error(uploadResult.message || '上传失败');
          }

          fileId = uploadResult.fileId;
        }

        if (!fileId) {
          throw new Error('当前源视频不可复用，请重新上传原视频');
        }

        createTask({
          ...selectedFile,
          targetPlayerBox: effectiveTargetPlayerBox,
        });

        const processResp = await processMutation.mutateAsync({
          fileId,
          targetPlayerBox: effectiveTargetPlayerBox,
        });

        navigate(`/progress/${processResp.taskId}`);
      },
      {
        showMessage: false,
      }
    )
  );

  const isProcessing = uploadMutation.isPending || processMutation.isPending || customLoading || isLoadingReusableSource;

  return (
    <div className="home-page min-h-screen overflow-x-hidden bg-[linear-gradient(180deg,#fff8f1_0%,#fffdf8_18%,#f8fafc_44%,#eef2ff_100%)]">
      {hasError && error ? (
        <div className="fixed left-1/2 top-4 z-50 w-full max-w-md -translate-x-1/2 px-4">
          <ErrorAlert
            title="处理请求失败"
            message={error.message}
            type="error"
            showIcon
            closable
            onClose={clearError}
          />
        </div>
      ) : null}

      <div className="relative isolate">
        <div className="absolute left-1/2 top-0 -z-10 h-[480px] w-[480px] -translate-x-1/2 rounded-full bg-orange-300/20 blur-3xl" />
        <div className="absolute right-[-120px] top-[120px] -z-10 h-[340px] w-[340px] rounded-full bg-indigo-300/20 blur-3xl" />

        <div className="mx-auto flex w-full max-w-7xl flex-col gap-12 px-4 pb-10 pt-8 sm:px-6 lg:px-8 lg:pb-16 lg:pt-12">
          <header className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-950 text-lg font-semibold text-white shadow-lg shadow-orange-500/20">
                H
              </div>
              <div>
                <p className="text-lg font-semibold tracking-tight text-slate-950">HoopCut</p>
                <p className="text-sm text-slate-500">篮球个人高光自动剪辑</p>
              </div>
            </div>
            <Chip variant="soft" color="warning" className="hidden md:flex">
              本地视频处理
            </Chip>
          </header>

          <section className="grid gap-8 lg:grid-cols-[minmax(0,1.05fr)_360px] lg:items-end">
            <div className="space-y-6">
              <div className="flex flex-wrap gap-2">
                <Chip variant="soft" color="warning">AI 进球检测</Chip>
                <Chip variant="soft" color="accent">人物追踪初始化</Chip>
                <Chip variant="soft" color="success">自动导出个人高光</Chip>
              </div>

              <div className="max-w-4xl space-y-4">
                <h1 className="font-serif text-5xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-6xl lg:text-7xl">
                  把一整场比赛
                  <br />
                  压成目标球员的高光片。
                </h1>
                <p className="max-w-2xl text-base leading-8 text-slate-600 sm:text-lg">
                  上传一段完整比赛视频，先从系统给出的候选截图里选中目标球员，再框选这个人。HoopCut 会自动分析整段素材，默认只交付和这个人相关的进球、助攻片段；系统补充回合只会放进高级排错区。
                </p>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row">
                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => window.scrollTo({ top: 760, behavior: 'smooth' })}
                  className="rounded-full px-6"
                >
                  <span className="inline-flex items-center gap-2">
                    立即上传视频
                    <ArrowRight size={18} />
                  </span>
                </Button>
                <div className="flex items-center text-sm text-slate-500">
                  当前版本聚焦于单目标球员的进球、助攻片段提取，不做全队战术分析。
                </div>
              </div>
            </div>

            <Card className="border border-white/15 bg-slate-950 text-white shadow-[0_24px_80px_rgba(15,23,42,0.30)]">
              <div className="space-y-6 p-6">
                <div className="flex items-center justify-between">
                  <span className="text-sm uppercase tracking-[0.24em] text-slate-400">流程</span>
                  <Sparkles size={16} className="text-orange-300" />
                </div>
                <div className="space-y-4">
                  {[
                    ['01', '上传原始比赛视频', '支持大文件，本地分块上传'],
                    ['02', '先选一张目标截图，再框选人物', '优先选择目标球员最早且清晰的出镜画面'],
                    ['03', '自动分析并交付进球 / 助攻片段', '结果页默认交付 confirmed ZIP，拼接视频和高级排错材料都只作补充'],
                  ].map(([index, title, desc]) => (
                    <div key={index} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                      <div className="mb-2 text-xs font-semibold tracking-[0.22em] text-orange-300">{index}</div>
                      <div className="text-base font-semibold text-white">{title}</div>
                      <div className="mt-1 text-sm leading-6 text-slate-400">{desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </section>

          <section className="grid gap-4 md:grid-cols-3">
            {[
              { icon: Radar, title: '先锁定目标球员', desc: '不是泛泛识别比赛，而是先把目标人物定准，并自动补充参考画面。' },
              { icon: Target, title: '只围绕目标来筛镜头', desc: '先锁定目标人物，再分析整段视频里的相关进球和助攻。' },
              { icon: Sparkles, title: '默认流程自动处理', desc: '上传、选人、开始处理，最后直接验收片段 ZIP。' },
            ].map(({ icon: Icon, title, desc }) => (
              <Card key={title} className="border border-white/40 bg-white/72 shadow-[0_16px_50px_rgba(15,23,42,0.08)] backdrop-blur">
                <div className="space-y-3 p-5">
                  <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-orange-100 text-orange-600">
                    <Icon size={20} />
                  </div>
                  <div className="text-lg font-semibold text-slate-900">{title}</div>
                  <p className="text-sm leading-6 text-slate-600">{desc}</p>
                </div>
              </Card>
            ))}
          </section>
        </div>
      </div>

      <div className="mx-auto w-full max-w-7xl px-4 pb-16 sm:px-6 lg:px-8">
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <VideoUpload
              onFileSelect={handleFileSelect}
              onExitReusableSource={() => clearReusableMode()}
              initialSource={reusableSource}
              loading={isProcessing}
              progress={uploadProgress}
              disabled={isProcessing}
            />

            {reusableSource ? (
              <Card className="border border-indigo-100 bg-indigo-50/80 p-5 shadow-[0_18px_50px_rgba(99,102,241,0.08)]">
                <div className="space-y-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-500">复用源视频</p>
                  <h2 className="text-xl font-semibold text-slate-900">已载入上一次处理的视频</h2>
                  <p className="text-sm leading-6 text-slate-600">
                    当前直接复用原始比赛视频，无需重新上传。你只需要重新选择起始截图或微调人物框选，然后重新开始自动剪辑。
                  </p>
                  <div className="pt-2">
                    <Button variant="secondary" size="sm" onClick={() => clearReusableMode()}>
                      改用新的比赛视频
                    </Button>
                  </div>
                </div>
              </Card>
            ) : null}

            {hasConfirmedSelectionFrame && selectedFile?.selectionFrame ? (
              <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                <PlayerSelector
                  frame={selectedFile.selectionFrame}
                  value={targetPlayerBox}
                  onChange={handleTargetPlayerChange}
                  disabled={isProcessing}
                />
              </Card>
            ) : null}

            {selectedFile && !hasConfirmedSelectionFrame ? (
              <Card className="border border-orange-100 bg-orange-50/80 p-6 shadow-[0_18px_50px_rgba(249,115,22,0.10)]">
                <div className="space-y-2">
                  <p className="text-sm font-semibold uppercase tracking-[0.2em] text-orange-500">第 2 步</p>
                  <h2 className="text-xl font-semibold text-slate-900">先选中框选起始截图</h2>
                  <p className="text-sm leading-6 text-slate-600">
                    请先在上方候选截图里选择目标球员最早且清晰出镜的一张。只有这一步确认后，下面才会出现框选人物区域；候选都不合适时，再手动拖视频补选。
                  </p>
                </div>
              </Card>
            ) : null}
          </div>

          <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
            <Card className="border border-slate-900/5 bg-slate-950 text-white shadow-[0_30px_90px_rgba(15,23,42,0.26)]">
              <div className="space-y-5 p-6">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.24em] text-orange-300">开始前检查</p>
                  <h2 className="text-2xl font-semibold tracking-tight">开始自动剪辑</h2>
                  <p className="text-sm leading-6 text-slate-400">
                    当前版本会直接按默认策略自动处理，不需要你手动调参数。重点是把和目标球员相关的进球、助攻一次性切出来；系统补充回合只在排错时才会展开。
                  </p>
                </div>

                <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
                  <div className="flex items-center justify-between">
                    <span>视频已上传</span>
                    <span>
                      {!selectedFile
                        ? '未选择'
                        : selectedFile.uploadStatus === 'uploading'
                          ? '上传中'
                          : selectedFile.uploadStatus === 'uploaded'
                            ? '已完成'
                            : '待上传'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>框选起始帧已确认</span>
                    <span>{hasConfirmedSelectionFrame ? '已完成' : '待操作'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>目标球员已框选</span>
                    <span>{targetPlayerBox ? '已完成' : hasConfirmedSelectionFrame ? '待操作' : '等待上一步'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>自动切片策略</span>
                    <span>系统默认处理</span>
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => void handleStartProcessing()}
                  isDisabled={
                    !selectedFile
                    || !targetPlayerBox
                    || isProcessing
                    || selectedFile.uploadStatus === 'uploading'
                  }
                  className="w-full rounded-full font-semibold"
                >
                  <span className="inline-flex items-center gap-2">
                    {isProcessing ? '处理中...' : '开始自动剪辑'}
                    <ArrowRight size={18} />
                  </span>
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};
