/**
 * 视频上传组件
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, Button, Card, Chip } from '@heroui/react';
import { Film, ImagePlus, Trash2, UploadCloud, Video } from 'lucide-react';
import { validateVideoFile, formatFileSize, formatDuration } from '@/utils';
import type { SelectionFrame, VideoFile } from '@/types';
import { useErrorHandler } from '@/hooks/useErrorHandler';

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

    const timeout = setTimeout(() => {
      cleanup(() => reject(new Error('获取视频信息超时')));
    }, 10000);

    video.preload = 'metadata';
    video.onloadedmetadata = () => {
      clearTimeout(timeout);
      cleanup(() => resolve(video.duration));
    };
    video.onerror = () => {
      clearTimeout(timeout);
      cleanup(() => reject(new Error('无法读取视频文件，请检查文件格式')));
    };
    video.src = previewUrl;
    video.load();
  });
};

const extractSelectionFrame = (
  previewUrl: string,
  timeInSeconds = 0
): Promise<SelectionFrame> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    let settled = false;

    const finalize = (handler: () => void) => {
      if (settled) {
        return;
      }

      settled = true;
      video.pause();
      video.removeAttribute('src');
      video.load();
      handler();
    };

    const timeout = setTimeout(() => {
      finalize(() => reject(new Error('抽取框选画面超时，请更换视频后重试')));
    }, 10000);

    const cleanup = (callback: () => void) => {
      clearTimeout(timeout);
      finalize(callback);
    };

    const captureFrame = (captureTime: number) => {
      if (video.videoWidth <= 0 || video.videoHeight <= 0) {
        cleanup(() => reject(new Error('无法读取视频画面尺寸')));
        return;
      }

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext('2d');
      if (!context) {
        cleanup(() => reject(new Error('无法创建视频预览画布')));
        return;
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      cleanup(() =>
        resolve({
          imageUrl: canvas.toDataURL('image/jpeg', 0.92),
          width: canvas.width,
          height: canvas.height,
          time: captureTime,
          frame: video.readyState >= 1 && video.duration > 0
            ? Math.round(captureTime * 30)
            : undefined,
        })
      );
    };

    const captureAtCurrentTime = () => {
      const captureTime = Math.max(video.currentTime || 0, 0);
      if (video.readyState >= 2) {
        captureFrame(captureTime);
        return;
      }

      video.onloadeddata = () => {
        captureFrame(Math.max(video.currentTime || 0, 0));
      };
    };

    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;
    video.onloadedmetadata = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      const targetTime = Math.max(0, Math.min(timeInSeconds, duration > 0 ? duration : timeInSeconds));

      if (targetTime <= 0.01) {
        captureAtCurrentTime();
        return;
      }

      video.onseeked = () => {
        captureAtCurrentTime();
      };
      video.currentTime = targetTime;
    };
    video.onerror = () => {
      cleanup(() => reject(new Error('无法抽取视频画面，请检查文件是否损坏')));
    };
    video.src = previewUrl;
    video.load();
  });
};

interface VideoUploadProps {
  onFileSelect: (file: VideoFile | null) => void;
  loading?: boolean;
  progress?: number;
  disabled?: boolean;
  className?: string;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onFileSelect,
  loading = false,
  progress = 0,
  disabled = false,
  className = '',
}) => {
  const [selectedFile, setSelectedFile] = useState<VideoFile | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCapturingFrame, setIsCapturingFrame] = useState(false);
  const [previewCurrentTime, setPreviewCurrentTime] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const { error, hasError, handleError, clearError } = useErrorHandler();

  const handleFileSelect = useCallback(async (file: File) => {
    clearError();
    setIsProcessing(true);
    let preview: string | null = null;

    try {
      const validation = validateVideoFile(file);
      if (!validation.valid) {
        throw new Error(validation.error || '文件验证失败');
      }

      preview = URL.createObjectURL(file);
      const [duration, selectionFrame] = await Promise.all([
        loadVideoDuration(preview),
        extractSelectionFrame(preview, 0),
      ]);

      const videoFile: VideoFile = {
        file,
        preview,
        duration,
        size: file.size,
        type: file.type,
        name: file.name,
        selectionFrame,
        targetPlayerBox: null,
      };

      setSelectedFile(videoFile);
      setPreviewUrl(preview);
      setPreviewCurrentTime(0);
      onFileSelect(videoFile);
    } catch (err) {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
      handleError(err instanceof Error ? err : new Error('处理文件时发生未知错误'));
    } finally {
      setIsProcessing(false);
    }

    return false;
  }, [onFileSelect, handleError, clearError]);

  const handleCaptureCurrentFrame = useCallback(async () => {
    if (!selectedFile || !previewUrl) {
      return;
    }

    clearError();
    setIsCapturingFrame(true);

    try {
      const currentTime = Math.max(previewVideoRef.current?.currentTime || 0, 0);
      const selectionFrame = await extractSelectionFrame(previewUrl, currentTime);
      const updatedFile: VideoFile = {
        ...selectedFile,
        selectionFrame,
        targetPlayerBox: null,
      };

      setSelectedFile(updatedFile);
      setPreviewCurrentTime(selectionFrame.time);
      onFileSelect(updatedFile);
    } catch (err) {
      handleError(err instanceof Error ? err : new Error('截取当前画面失败'));
    } finally {
      setIsCapturingFrame(false);
    }
  }, [selectedFile, previewUrl, onFileSelect, handleError, clearError]);

  const handleRemoveFile = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl('');
    setPreviewCurrentTime(0);
    clearError();
    onFileSelect(null);
  }, [previewUrl, onFileSelect, clearError]);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className={`video-upload ${className}`}>
      {!selectedFile ? (
        <Card className="overflow-hidden border border-white/10 bg-slate-950 text-white shadow-[0_35px_120px_rgba(15,23,42,0.45)]">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            onDrop={(event) => {
              event.preventDefault();
              if (disabled || loading || isProcessing || isCapturingFrame) {
                return;
              }
              const files = Array.from(event.dataTransfer.files);
              if (files.length > 0) {
                void handleFileSelect(files[0]);
              }
            }}
            onDragOver={(event) => event.preventDefault()}
            className="upload-zone relative flex w-full flex-col items-center gap-5 px-6 py-12 text-center sm:px-10"
            disabled={disabled || loading || isProcessing || isCapturingFrame}
          >
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(249,115,22,0.26),_transparent_40%),linear-gradient(180deg,rgba(255,255,255,0.06),rgba(255,255,255,0))]" />
            <div className="relative flex h-16 w-16 items-center justify-center rounded-full border border-white/15 bg-white/10 backdrop-blur">
              <UploadCloud size={30} className="text-orange-300" />
            </div>
            <div className="relative space-y-3">
              <h3 className="text-2xl font-semibold tracking-tight text-white">上传篮球视频</h3>
              <p className="mx-auto max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
                拖拽文件到这里，或点击选择本地比赛视频。上传后你可以先定位第一次清晰出镜的画面，再框选自己。
              </p>
            </div>
            <div className="relative flex flex-wrap justify-center gap-2">
              <Chip variant="soft" color="warning">自动检测投篮</Chip>
              <Chip variant="soft" color="accent">个人高光剪辑</Chip>
              <Chip variant="soft" color="success">多格式兼容</Chip>
            </div>
            <div className="relative text-sm text-slate-400">
              支持 MP4、AVI、MOV、WMV、FLV、WebM、MKV，文件大小 1MB - 2GB
            </div>
          </button>
        </Card>
      ) : (
        <Card className="selected-file-preview space-y-4 border border-white/10 bg-white/85 p-4 shadow-[0_24px_80px_rgba(15,23,42,0.10)] backdrop-blur sm:p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 flex-1">
              <div className="mb-2 flex items-center gap-2">
                <Video size={18} className="text-orange-500" />
                <h4 className="min-w-0 truncate text-lg font-semibold text-slate-900">
                  {selectedFile.name}
                </h4>
              </div>
              <div className="grid grid-cols-1 gap-2 text-sm text-slate-600 sm:grid-cols-3">
                <div className="flex justify-between sm:flex-col">
                  <span>文件大小:</span>
                  <span className="font-medium">{formatFileSize(selectedFile.size)}</span>
                </div>
                <div className="flex justify-between sm:flex-col">
                  <span>视频时长:</span>
                  <span className="font-medium">
                    {selectedFile.duration ? formatDuration(selectedFile.duration) : '未知'}
                  </span>
                </div>
                <div className="flex justify-between sm:flex-col">
                  <span>文件格式:</span>
                  <span className="font-medium uppercase">
                    {selectedFile.name.split('.').pop() || 'Unknown'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex-shrink-0">
              <Button
                variant="danger-soft"
                onClick={handleRemoveFile}
                isDisabled={loading || isCapturingFrame}
                size="sm"
              >
                <span className="inline-flex items-center gap-2">
                  <Trash2 size={14} />
                  移除
                </span>
              </Button>
            </div>
          </div>

          <Alert status="accent">
            <div className="font-medium text-current">
              如果你不是在开头出镜，先把下面视频拖到你第一次清晰出镜的位置，再截取当前画面。
            </div>
          </Alert>

          <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-4">
            <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <ImagePlus size={18} className="text-orange-500" />
                  <span className="font-semibold text-slate-900">选择框选起点</span>
                </div>
                <p className="mb-0 mt-2 text-sm text-slate-600">
                  当前播放位置只是候选画面。点击右侧按钮后，后续框选和跟踪才会从该时间点开始。
                </p>
              </div>
              <Button
                variant="primary"
                onClick={() => void handleCaptureCurrentFrame()}
                isDisabled={disabled || loading || isProcessing || isCapturingFrame}
              >
                <span className="inline-flex items-center gap-2">
                  <Film size={16} />
                  {isCapturingFrame ? '截取中...' : '使用当前播放位置作为框选帧'}
                </span>
              </Button>
            </div>

            {previewUrl ? (
              <video
                ref={previewVideoRef}
                src={previewUrl}
                className="w-full rounded-[20px] bg-black shadow-[0_16px_50px_rgba(15,23,42,0.28)]"
                controls
                muted
                playsInline
                onTimeUpdate={(event) => {
                  setPreviewCurrentTime(event.currentTarget.currentTime);
                }}
                onSeeked={(event) => {
                  setPreviewCurrentTime(event.currentTarget.currentTime);
                }}
                onLoadedMetadata={(event) => {
                  setPreviewCurrentTime(event.currentTarget.currentTime);
                }}
              />
            ) : null}

            <div className="mt-3 flex flex-col gap-1 text-sm text-slate-600 sm:flex-row sm:items-center sm:justify-between">
              <span>当前播放位置：{formatDuration(previewCurrentTime)}（{previewCurrentTime.toFixed(2)}s）</span>
              <span>
                当前框选帧：
                {selectedFile.selectionFrame
                  ? ` ${formatDuration(selectedFile.selectionFrame.time)}（${selectedFile.selectionFrame.time.toFixed(2)}s）`
                  : ' 尚未选择'}
              </span>
            </div>
          </div>

          {(loading || isProcessing) && (
            <div className="mt-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm font-medium text-slate-700">
                  {isProcessing ? '读取文件中...' : loading ? `上传中 ${progress}%` : '处理中...'}
                </span>
                <span className="text-sm text-slate-500">请稍候</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-200">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-orange-500 to-amber-400 transition-all duration-300"
                  style={{ width: `${isProcessing ? 100 : progress}%` }}
                />
              </div>
            </div>
          )}
        </Card>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            void handleFileSelect(file);
          }
          event.currentTarget.value = '';
        }}
      />

      {hasError && error ? (
        <div className="mt-4">
          <Alert status="danger" className="items-start">
            <div className="flex flex-1 items-start justify-between gap-3">
              <div className="font-medium text-current">{error.message}</div>
              <button
                type="button"
                onClick={clearError}
                className="text-current/60 transition hover:text-current"
                aria-label="关闭上传错误提示"
              >
                ×
              </button>
            </div>
          </Alert>
        </div>
      ) : null}
    </div>
  );
};
