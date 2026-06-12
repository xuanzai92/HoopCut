import React, { useEffect, useRef, useState } from 'react';
import { Button, Card } from '@heroui/react';
import {
  ChevronLeft,
  ChevronRight,
  Clock3,
  Film,
  Plus,
  RefreshCcw,
  Scissors,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { ConfigPanel } from '@/components/ui/ConfigPanel';
import ErrorAlert from '@/components/common/ErrorAlert';
import { useTaskStore } from '@/store';
import { useErrorHandler, useLoading } from '@/hooks';
import { useUploadVideo, useProcessVideo } from '@/services';
import type { ProcessingConfig, ReusableVideoSource, VideoFile } from '@/types';
import {
  DEFAULT_PROCESSING_CONFIG,
  formatDuration,
  formatFileSize,
  validateProcessingConfig,
  validateVideoFile,
} from '@/utils';

const loadVideoDuration = (previewUrl: string): Promise<number> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    let settled = false;

    const cleanup = (handler: () => void) => {
      if (settled) {
        return;
      }

      settled = true;
      video.pause();
      video.removeAttribute('src');
      video.load();
      handler();
    };

    const timeout = window.setTimeout(() => {
      cleanup(() => reject(new Error('获取视频时长超时')));
    }, 10000);

    video.preload = 'metadata';
    video.crossOrigin = 'anonymous';
    video.onloadedmetadata = () => {
      window.clearTimeout(timeout);
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      cleanup(() => resolve(duration));
    };
    video.onerror = () => {
      window.clearTimeout(timeout);
      cleanup(() => reject(new Error('无法读取视频文件')));
    };
    video.src = previewUrl;
    video.load();
  });
};

const isObjectUrl = (url?: string) => Boolean(url?.startsWith('blob:'));

const releasePreviewUrl = (url?: string) => {
  if (isObjectUrl(url)) {
    URL.revokeObjectURL(url as string);
  }
};

const normalizeMoment = (value: number): number => {
  return Math.round(Math.max(value, 0) * 10) / 10;
};

const upsertMoment = (moments: number[], nextMoment: number): number[] => {
  const normalized = normalizeMoment(nextMoment);
  if (moments.some((moment) => Math.abs(moment - normalized) < 0.05)) {
    return moments;
  }

  return [...moments, normalized].sort((left, right) => left - right);
};

interface ManualWorkflowProps {
  initialSource?: ReusableVideoSource | null;
  onExitReusableSource?: () => void;
}

export const ManualWorkflow: React.FC<ManualWorkflowProps> = ({
  initialSource = null,
  onExitReusableSource,
}) => {
  const [selectedFile, setSelectedFile] = useState<VideoFile | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [previewCurrentTime, setPreviewCurrentTime] = useState(0);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [manualMoments, setManualMoments] = useState<number[]>([]);
  const [config, setConfig] = useState<ProcessingConfig>(DEFAULT_PROCESSING_CONFIG);
  const [isPreparingSource, setIsPreparingSource] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const { createTask } = useTaskStore();
  const { error, hasError, clearError, handleError, withErrorHandling } = useErrorHandler();
  const { loading: customLoading, withLoading } = useLoading();
  const navigate = useNavigate();
  const uploadMutation = useUploadVideo();
  const processMutation = useProcessVideo();

  useEffect(() => {
    return () => {
      releasePreviewUrl(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    if (!initialSource) {
      return;
    }

    let cancelled = false;
    setIsPreparingSource(true);
    clearError();

    void loadVideoDuration(initialSource.sourceStreamUrl)
      .then((duration) => {
        if (cancelled) {
          return;
        }

        setSelectedFile({
          preview: initialSource.sourceStreamUrl,
          duration,
          size: initialSource.fileSize,
          type: initialSource.mimeType || 'video/mp4',
          name: initialSource.filename,
          sourceTaskId: initialSource.taskId,
          sourceFileId: initialSource.fileId,
          sourceUrl: initialSource.sourceStreamUrl,
          reusable: true,
          reusableSource: initialSource,
          uploadStatus: 'uploaded',
        });
        setPreviewUrl((current) => {
          releasePreviewUrl(current);
          return initialSource.sourceStreamUrl;
        });
        setPreviewCurrentTime(0);
        setManualMoments(
          Array.isArray(initialSource.manualMoments)
            ? initialSource.manualMoments.map((moment) => normalizeMoment(Number(moment))).filter((moment) => Number.isFinite(moment))
            : [],
        );
      })
      .catch((nextError) => {
        if (!cancelled) {
          setSelectedFile(null);
          setPreviewUrl((current) => {
            releasePreviewUrl(current);
            return '';
          });
          handleError(nextError instanceof Error ? nextError : new Error('加载源视频失败'));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsPreparingSource(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [initialSource, clearError, handleError]);

  const isProcessing = uploadMutation.isPending || processMutation.isPending || customLoading || isPreparingSource;

  const seekPreviewTo = (time: number) => {
    const video = previewVideoRef.current;
    if (!video) {
      return;
    }

    const duration = Number.isFinite(video.duration) ? video.duration : (selectedFile?.duration || 0);
    const nextTime = Math.min(Math.max(time, 0), duration || time);
    video.currentTime = nextTime;
    setPreviewCurrentTime(nextTime);
  };

  const seekPreviewBy = (offset: number) => {
    seekPreviewTo(previewCurrentTime + offset);
  };

  const handleSelectFile = withErrorHandling(async (file: File) => {
    const validation = validateVideoFile(file);
    if (!validation.valid) {
      throw new Error(validation.error || '文件验证失败');
    }

    const nextPreviewUrl = URL.createObjectURL(file);
    const duration = await loadVideoDuration(nextPreviewUrl);

    if (initialSource) {
      onExitReusableSource?.();
    }

    setSelectedFile({
      file,
      preview: nextPreviewUrl,
      duration,
      size: file.size,
      type: file.type,
      name: file.name,
      uploadStatus: 'local',
    });
    setPreviewUrl((current) => {
      releasePreviewUrl(current);
      return nextPreviewUrl;
    });
    setPreviewCurrentTime(0);
    setUploadProgress(0);
    setManualMoments([]);
  });

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    void handleSelectFile(file);
    event.currentTarget.value = '';
  };

  const handleAddCurrentMoment = () => {
    setManualMoments((current) => upsertMoment(current, previewCurrentTime));
  };

  const handleRemoveMoment = (targetMoment: number) => {
    setManualMoments((current) => current.filter((moment) => moment !== targetMoment));
  };

  const handleStartProcessing = withLoading(
    withErrorHandling(
      async () => {
        if (!selectedFile) {
          throw new Error('请先选择视频文件');
        }

        if (manualMoments.length === 0) {
          throw new Error('请先添加至少一个时间点');
        }

        const validation = validateProcessingConfig(config);
        if (!validation.valid) {
          throw new Error(validation.error || '剪辑参数无效');
        }

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
          setSelectedFile((current) => current ? {
            ...current,
            sourceFileId: uploadResult.fileId,
            uploadStatus: 'uploaded',
          } : current);
        }

        if (!fileId) {
          throw new Error('当前源视频不可复用，请重新上传原视频');
        }

        createTask(selectedFile, config);

        const processResp = await processMutation.mutateAsync({
          fileId,
          mode: 'manual',
          manualMoments,
          beforeSeconds: config.beforeSeconds,
          afterSeconds: config.afterSeconds,
        });

        navigate(`/progress/${processResp.taskId}`);
      },
      { showMessage: false },
    ),
  );

  return (
    <div className="space-y-6">
      {hasError && error ? (
        <ErrorAlert
          title="手动剪片失败"
          message={error.message}
          type="error"
          showIcon
          closable
          onClose={clearError}
        />
      ) : null}

      {!selectedFile ? (
        <Card className="border border-white/40 bg-white/82 p-8 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur">
          <div className="space-y-5 text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-3xl bg-orange-100 text-orange-600">
              <UploadCloud size={28} />
            </div>
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold tracking-tight text-slate-950">上传比赛视频</h2>
              <p className="mx-auto max-w-2xl text-sm leading-6 text-slate-600">
                默认主流程不再让系统先帮你挑“优选画面”。你只需要上传视频，然后自己拖到关键时间点直接切片。
              </p>
            </div>
            <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button
                variant="primary"
                size="lg"
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="inline-flex items-center gap-2">
                  <UploadCloud size={18} />
                  选择本地视频
                </span>
              </Button>
              <div className="text-sm text-slate-500">支持大文件上传，本地处理，不会把视频发到远端。</div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.wmv,.flv"
              className="hidden"
              onChange={handleFileInputChange}
            />
          </div>
        </Card>
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <Card className="border border-white/40 bg-white/82 p-5 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-slate-900">
                    <Film size={18} className="text-orange-500" />
                    <h2 className="text-xl font-semibold tracking-tight">自己选时间点</h2>
                  </div>
                  <p className="text-sm leading-6 text-slate-600">
                    直接把播放器拖到你想截取的时刻，点一下“添加当前时间点”。这条模式不会再跑人物跟踪和自动归因。
                  </p>
                  <div className="flex flex-wrap gap-2 text-sm text-slate-500">
                    <span>{selectedFile.name}</span>
                    <span>·</span>
                    <span>{selectedFile.duration ? formatDuration(selectedFile.duration) : '00:00'}</span>
                    <span>·</span>
                    <span>{formatFileSize(selectedFile.size)}</span>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button variant="secondary" onClick={() => fileInputRef.current?.click()}>
                    <span className="inline-flex items-center gap-2">
                      <RefreshCcw size={16} />
                      更换视频
                    </span>
                  </Button>
                  {initialSource ? (
                    <Button variant="ghost" onClick={() => onExitReusableSource?.()}>
                      改用新的比赛视频
                    </Button>
                  ) : null}
                </div>
              </div>

              <video
                ref={previewVideoRef}
                src={previewUrl}
                crossOrigin="anonymous"
                className="w-full rounded-[24px] bg-black shadow-[0_16px_50px_rgba(15,23,42,0.28)]"
                controls
                playsInline
                onTimeUpdate={(event) => setPreviewCurrentTime(event.currentTarget.currentTime)}
                onSeeked={(event) => setPreviewCurrentTime(event.currentTarget.currentTime)}
                onLoadedMetadata={(event) => setPreviewCurrentTime(event.currentTarget.currentTime)}
              />

              <div className="mt-4 rounded-3xl border border-slate-200 bg-slate-50/80 p-4">
                <div className="mb-3 flex items-center justify-between gap-3 text-sm text-slate-600">
                  <span>当前播放位置：{formatDuration(previewCurrentTime)}（{previewCurrentTime.toFixed(1)}s）</span>
                  <span>已选时间点：{manualMoments.length}</span>
                </div>

                <input
                  type="range"
                  min={0}
                  max={selectedFile.duration || 0}
                  step={0.1}
                  value={previewCurrentTime}
                  onChange={(event) => seekPreviewTo(Number(event.currentTarget.value))}
                  className="hero-range w-full"
                  disabled={!selectedFile.duration}
                />

                <div className="mt-3 flex flex-wrap gap-2">
                  <Button variant="secondary" size="sm" onClick={() => seekPreviewTo(0)}>
                    开头
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => seekPreviewBy(-5)}>
                    <span className="inline-flex items-center gap-1">
                      <ChevronLeft size={14} />
                      退 5s
                    </span>
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => seekPreviewBy(-1)}>
                    退 1s
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => seekPreviewBy(1)}>
                    进 1s
                  </Button>
                  <Button variant="secondary" size="sm" onClick={() => seekPreviewBy(5)}>
                    <span className="inline-flex items-center gap-1">
                      进 5s
                      <ChevronRight size={14} />
                    </span>
                  </Button>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handleAddCurrentMoment}
                    isDisabled={isProcessing}
                  >
                    <span className="inline-flex items-center gap-1">
                      <Plus size={14} />
                      添加当前时间点
                    </span>
                  </Button>
                </div>
              </div>
            </Card>

            <Card className="border border-white/40 bg-white/82 p-5 shadow-[0_20px_70px_rgba(15,23,42,0.08)] backdrop-blur sm:p-6">
              <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-slate-900">
                    <Clock3 size={18} className="text-orange-500" />
                    <h2 className="text-xl font-semibold tracking-tight">待导出时间点</h2>
                  </div>
                  <p className="text-sm leading-6 text-slate-600">
                    这里就是最终要切出的片段中心点。你可以继续增加、删除，或者点某个时间点回到对应位置再检查一遍。
                  </p>
                </div>
                {manualMoments.length > 0 ? (
                  <Button variant="ghost" size="sm" onClick={() => setManualMoments([])}>
                    <span className="inline-flex items-center gap-2">
                      <Trash2 size={14} />
                      清空
                    </span>
                  </Button>
                ) : null}
              </div>

              {manualMoments.length > 0 ? (
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                  {manualMoments.map((moment, index) => (
                    <div
                      key={`${moment}-${index}`}
                      className="rounded-2xl border border-slate-200 bg-slate-50/80 p-4 text-left transition hover:border-orange-200 hover:bg-orange-50/70"
                      onClick={() => seekPreviewTo(moment)}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault();
                          seekPreviewTo(moment);
                        }
                      }}
                      role="button"
                      tabIndex={0}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm text-slate-500">片段 {index + 1}</div>
                          <div className="mt-1 text-lg font-semibold text-slate-900">
                            {formatDuration(moment)}
                          </div>
                          <div className="mt-1 text-sm text-slate-500">{moment.toFixed(1)}s</div>
                        </div>
                        <button
                          type="button"
                          className="rounded-full border border-slate-200 p-2 text-slate-500 transition hover:border-rose-200 hover:bg-rose-50 hover:text-rose-500"
                          onClick={(event) => {
                            event.stopPropagation();
                            handleRemoveMoment(moment);
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50/70 px-4 py-10 text-center text-sm text-slate-500">
                  还没有添加时间点。先在上面拖到你要的时刻，再点“添加当前时间点”。
                </div>
              )}
            </Card>
          </div>

          <div className="space-y-6 xl:sticky xl:top-6 xl:self-start">
            <ConfigPanel
              config={config}
              onChange={setConfig}
              disabled={isProcessing}
            />

            <Card className="border border-slate-900/5 bg-slate-950 text-white shadow-[0_30px_90px_rgba(15,23,42,0.26)]">
              <div className="space-y-5 p-6">
                <div className="space-y-2">
                  <p className="text-sm uppercase tracking-[0.24em] text-orange-300">开始前检查</p>
                  <h2 className="text-2xl font-semibold tracking-tight">开始按时间点剪片</h2>
                  <p className="text-sm leading-6 text-slate-400">
                    默认主流程只做你明确选择的片段。这样页面更简单，处理速度也会比完整自动分析快很多。
                  </p>
                </div>

                <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300">
                  <div className="flex items-center justify-between">
                    <span>视频已准备</span>
                    <span>{selectedFile.uploadStatus === 'uploaded' ? '已完成' : '待上传'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>已选时间点</span>
                    <span>{manualMoments.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>前置保留</span>
                    <span>{config.beforeSeconds.toFixed(1)}s</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>后置保留</span>
                    <span>{config.afterSeconds.toFixed(1)}s</span>
                  </div>
                </div>

                {(isProcessing || uploadProgress > 0) ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm text-slate-300">
                      <span>{uploadMutation.isPending ? `上传中 ${uploadProgress}%` : '处理中...'}</span>
                      <span>请稍候</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-white/10">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-orange-500 to-amber-400 transition-all duration-300"
                        style={{ width: `${uploadMutation.isPending ? uploadProgress : 100}%` }}
                      />
                    </div>
                  </div>
                ) : null}

                <Button
                  variant="primary"
                  size="lg"
                  onClick={() => void handleStartProcessing()}
                  isDisabled={!selectedFile || manualMoments.length === 0 || isProcessing}
                  className="w-full rounded-full font-semibold"
                >
                  <span className="inline-flex items-center gap-2">
                    {isProcessing ? '处理中...' : '开始导出片段'}
                    <Scissors size={18} />
                  </span>
                </Button>
              </div>
            </Card>
          </div>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mov,.avi,.mkv,.webm,.wmv,.flv"
        className="hidden"
        onChange={handleFileInputChange}
      />
    </div>
  );
};

export default ManualWorkflow;
