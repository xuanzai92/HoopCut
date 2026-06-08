/**
 * 主页面组件
 */
import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Card, Chip } from '@heroui/react';
import { ArrowRight, Radar, Sparkles, Target } from 'lucide-react';
import { VideoUpload } from '@/components/upload/VideoUpload';
import { ConfigPanel } from '@/components/ui/ConfigPanel';
import { PlayerSelector } from '@/components/ui/PlayerSelector';
import ErrorAlert from '@/components/common/ErrorAlert';
import { useTaskStore, useConfigActions } from '@/store';
import { useUploadVideo, useProcessVideo } from '@/services';
import type { PlayerSelectionBox, VideoFile, ProcessingConfig } from '@/types';
import { validateProcessingConfig, validateTargetPlayerBox } from '@/utils';
import { useErrorHandler, useLoading } from '@/hooks';

export const Home: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<VideoFile | null>(null);
  const [targetPlayerBox, setTargetPlayerBox] = useState<PlayerSelectionBox | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [processingConfig, setProcessingConfig] = useState<ProcessingConfig>({
    beforeSeconds: 3,
    afterSeconds: 1,
  });

  // Store hooks
  const { createTask } = useTaskStore();
  const { updateProcessingConfig } = useConfigActions();
  
  // Custom hooks
  const { error, hasError, clearError, withErrorHandling } = useErrorHandler();
  const { loading: customLoading, withLoading } = useLoading();

  const navigate = useNavigate();

  // API hooks
  const uploadMutation = useUploadVideo();
  const processMutation = useProcessVideo();

  // 处理文件选择
  const handleFileSelect = useCallback((file: VideoFile | null) => {
    setSelectedFile(file);
    setTargetPlayerBox(null);
    setUploadProgress(0);
  }, []);

  const handleTargetPlayerChange = useCallback((selection: PlayerSelectionBox | null) => {
    setTargetPlayerBox(selection);
  }, []);

  // 处理配置变更
  const handleConfigChange = useCallback((config: ProcessingConfig) => {
    setProcessingConfig(config);
  }, []);

  const handleStartProcessing = withLoading(
    withErrorHandling(
      async () => {
        if (!selectedFile) {
          throw new Error('请先选择视频文件');
        }

        const configValidation = validateProcessingConfig(processingConfig);
        if (!configValidation.valid) {
          throw new Error(configValidation.error || '配置参数无效');
        }

        const selectionValidation = validateTargetPlayerBox(targetPlayerBox);
        if (!selectionValidation.valid) {
          throw new Error(selectionValidation.error || '人物选区无效');
        }

        setUploadProgress(0);

        const uploadResult = await uploadMutation.mutateAsync({
          file: selectedFile.file,
          onProgress: (progress) => {
            setUploadProgress(progress);
          },
        });

        if (!uploadResult.success) {
          throw new Error(uploadResult.message || '上传失败');
        }

        const fileId = uploadResult.fileId;
        createTask(selectedFile, processingConfig);
        updateProcessingConfig(processingConfig);

        const processResp = await processMutation.mutateAsync({
          fileId,
          beforeSeconds: processingConfig.beforeSeconds,
          afterSeconds: processingConfig.afterSeconds,
          targetPlayerBox,
        });

        navigate(`/progress/${processResp.taskId}`);
      },
      {
        showMessage: false,
      }
    )
  );

  const isProcessing = uploadMutation.isPending || processMutation.isPending || customLoading;

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
                <p className="text-sm text-slate-500">Basketball highlight generator</p>
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
                  压成你自己的高光片。
                </h1>
                <p className="max-w-2xl text-base leading-8 text-slate-600 sm:text-lg">
                  上传比赛视频，定位你第一次清晰出镜的时间点，框选自己，HoopCut 会从那一帧开始做人像追踪、识别你的进球和助攻，并导出可直接分享的个人集锦。
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
                  当前版本聚焦于个人集锦，不做全队战术分析。
                </div>
              </div>
            </div>

            <Card className="border border-white/15 bg-slate-950 text-white shadow-[0_24px_80px_rgba(15,23,42,0.30)]">
              <div className="space-y-6 p-6">
                <div className="flex items-center justify-between">
                  <span className="text-sm uppercase tracking-[0.24em] text-slate-400">Workflow</span>
                  <Sparkles size={16} className="text-orange-300" />
                </div>
                <div className="space-y-4">
                  {[
                    ['01', '上传原始比赛视频', '支持大文件，本地分块上传'],
                    ['02', '截取出镜起点并框选', '锁定你自己的追踪目标'],
                    ['03', '生成可分享集锦', '输出只属于你的高光片'],
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
              { icon: Radar, title: '先锁定你', desc: '不是泛泛识别比赛，而是先把目标球员定准。' },
              { icon: Target, title: '只剪你的有效镜头', desc: '围绕你本人相关进球和助攻做高光抽取。' },
              { icon: Sparkles, title: '少步骤直出成片', desc: '上传、框选、开始处理，避免冗余设置。' },
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
            <VideoUpload onFileSelect={handleFileSelect} loading={isProcessing} progress={uploadProgress} disabled={isProcessing} />

            {selectedFile?.selectionFrame ? (
              <Card className="border border-white/40 bg-white/82 p-6 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
                <PlayerSelector
                  frame={selectedFile.selectionFrame}
                  value={targetPlayerBox}
                  onChange={handleTargetPlayerChange}
                  disabled={isProcessing}
                />
              </Card>
            ) : null}
          </div>

          <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
            <ConfigPanel
              config={processingConfig}
              onChange={handleConfigChange}
              disabled={isProcessing}
            />

            <Card className="border border-slate-900/5 bg-slate-950 text-white shadow-[0_30px_90px_rgba(15,23,42,0.26)]">
              <div className="space-y-5 p-6">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.24em] text-orange-300">Ready Check</p>
                  <h2 className="text-2xl font-semibold tracking-tight">开始生成个人集锦</h2>
                  <p className="text-sm leading-6 text-slate-400">
                    当前版本会从你选中的出镜画面开始做目标跟踪，并导出归因到你的进球和助攻镜头。
                  </p>
                </div>

                <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
                  <div className="flex items-center justify-between">
                    <span>视频已上传</span>
                    <span>{selectedFile ? '已就绪' : '未选择'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>目标球员已框选</span>
                    <span>{targetPlayerBox ? '已完成' : '待操作'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>处理参数已设置</span>
                    <span>{`${processingConfig.beforeSeconds.toFixed(1)}s / ${processingConfig.afterSeconds.toFixed(1)}s`}</span>
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => void handleStartProcessing()}
                  isDisabled={!selectedFile || !targetPlayerBox || isProcessing}
                  className="w-full rounded-full font-semibold"
                >
                  <span className="inline-flex items-center gap-2">
                    {isProcessing ? '处理中...' : '开始生成个人集锦'}
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
