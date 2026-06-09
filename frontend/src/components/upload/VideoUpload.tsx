/**
 * 视频上传组件
 */
import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Alert, Button, Card, Chip } from '@heroui/react';
import { ChevronLeft, ChevronRight, Film, ImagePlus, RotateCcw, Trash2, UploadCloud, Video } from 'lucide-react';
import { validateVideoFile, formatFileSize, formatDuration, syncSelectionBoxToFrame } from '@/utils';
import type { PlayerSelectionBox, ReusableVideoSource, SelectionFrame, VideoFile } from '@/types';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { ApiService } from '@/services/api';
import { withRetry } from '@/services/http';

const REMOTE_SELECTION_FRAME_RETRY_ATTEMPTS = 4;
const REMOTE_SELECTION_FRAME_RETRY_DELAY_MS = 800;

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
    video.crossOrigin = 'anonymous';
    video.onloadedmetadata = () => {
      clearTimeout(timeout);
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      cleanup(() => resolve(duration));
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
          source: 'local',
          recommended: false,
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
    video.crossOrigin = 'anonymous';
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

const clampTime = (timeInSeconds: number, duration?: number): number => {
  if (!Number.isFinite(timeInSeconds)) {
    return 0;
  }

  if (!duration || duration <= 0) {
    return Math.max(0, timeInSeconds);
  }

  return Math.min(Math.max(0, timeInSeconds), duration);
};

const buildCandidateTimes = (duration: number): number[] => {
  if (!Number.isFinite(duration) || duration <= 0) {
    return [0];
  }

  const safeDuration = duration > 0.25 ? duration - 0.25 : duration;
  const candidateCount = duration <= 45
    ? 8
    : duration <= 5 * 60
      ? 12
      : duration <= 20 * 60
        ? 16
        : duration <= 60 * 60
          ? 20
          : 24;
  const times = new Set<number>([0]);

  if (candidateCount <= 1 || safeDuration <= 0.01) {
    return [0];
  }

  // 覆盖整段视频，同时对前段加密，减少用户必须手动拖视频找人的概率。
  const earlyRatios = duration >= 30 * 60
    ? [0.003, 0.008, 0.015, 0.03, 0.05, 0.08, 0.12, 0.18, 0.26]
    : duration >= 10 * 60
      ? [0.005, 0.015, 0.03, 0.06, 0.1, 0.16, 0.24]
      : duration >= 2 * 60
        ? [0.03, 0.07, 0.12, 0.18, 0.26]
        : duration >= 10
          ? [0.05, 0.12, 0.22]
          : [];
  earlyRatios.forEach((ratio) => {
    const time = Number((safeDuration * ratio).toFixed(2));
    times.add(clampTime(time, safeDuration));
  });

  for (let index = 1; index < candidateCount; index += 1) {
    const ratio = index / (candidateCount - 1);
    const time = Number((safeDuration * ratio).toFixed(2));
    times.add(clampTime(time, safeDuration));
  }

  return Array.from(times).sort((left, right) => left - right);
};

const areFramesNear = (left: SelectionFrame, right: SelectionFrame): boolean => {
  return Math.abs(left.time - right.time) < 0.35;
};

const normalizeSelectionFrame = (
  frame: SelectionFrame,
  defaults: Partial<SelectionFrame> = {},
): SelectionFrame => {
  const normalizedFrame: SelectionFrame = {
    ...defaults,
    ...frame,
    source: frame.source ?? defaults.source ?? 'local',
    recommended: frame.recommended ?? defaults.recommended ?? false,
  };

  normalizedFrame.suggestedBox = syncSelectionBoxToFrame(
    frame.suggestedBox ?? defaults.suggestedBox ?? null,
    normalizedFrame,
  );

  return normalizedFrame;
};

const getSelectionFramePriority = (frame: SelectionFrame): number => {
  return (
    (frame.recommended ? 2 : 0)
    + (frame.source === 'smart' ? 1 : 0)
    + (frame.recommendationScore ?? 0)
  );
};

const mergeSelectionFrameCandidates = (
  primaryFrames: SelectionFrame[],
  secondaryFrames: SelectionFrame[] = [],
  pinnedFrame?: SelectionFrame,
): SelectionFrame[] => {
  const merged: SelectionFrame[] = [];

  const upsertFrame = (frame: SelectionFrame) => {
    const normalizedFrame = normalizeSelectionFrame(frame);
    const existingIndex = merged.findIndex((candidate) => areFramesNear(candidate, normalizedFrame));
    if (existingIndex < 0) {
      merged.push(normalizedFrame);
      return;
    }

    if (getSelectionFramePriority(normalizedFrame) > getSelectionFramePriority(merged[existingIndex])) {
      merged[existingIndex] = normalizedFrame;
    }
  };

  primaryFrames.forEach(upsertFrame);
  secondaryFrames.forEach(upsertFrame);
  if (pinnedFrame) {
    upsertFrame(pinnedFrame);
  }

  return merged.sort((left, right) => {
    const priorityDiff = getSelectionFramePriority(right) - getSelectionFramePriority(left);
    if (Math.abs(priorityDiff) > 1e-6) {
      return priorityDiff;
    }
    return left.time - right.time;
  });
};

const extractSelectionFrameCandidates = async (
  previewUrl: string,
  duration: number,
): Promise<SelectionFrame[]> => {
  const candidateTimes = buildCandidateTimes(duration);
  const candidateFrames: Array<SelectionFrame | null> = [];

  for (const time of candidateTimes) {
    try {
      candidateFrames.push(await extractSelectionFrame(previewUrl, time));
    } catch {
      candidateFrames.push(null);
    }
  }

  return candidateFrames
    .filter((frame): frame is SelectionFrame => frame !== null)
    .reduce<SelectionFrame[]>((frames, frame) => {
      if (frames.some((existingFrame) => areFramesNear(existingFrame, frame))) {
        return frames;
      }
      frames.push(normalizeSelectionFrame(frame, { source: 'local', recommended: false }));
      return frames;
    }, []);
};

const isObjectUrl = (url?: string): boolean => Boolean(url?.startsWith('blob:'));

const releasePreviewUrl = (url?: string) => {
  if (isObjectUrl(url)) {
    URL.revokeObjectURL(url as string);
  }
};

interface PrepareVideoFileOptions {
  previewUrl: string;
  name: string;
  size: number;
  type: string;
  file?: File;
  sourceTaskId?: string;
  sourceFileId?: string;
  targetPlayerBox?: PlayerSelectionBox | null;
  reusable?: boolean;
  reusableSource?: ReusableVideoSource | null;
}

interface PreparedVideoFile {
  videoFile: VideoFile;
  candidateFrames: SelectionFrame[];
  initialPreviewTime: number;
}

const prepareVideoFile = async ({
  previewUrl,
  name,
  size,
  type,
  file,
  sourceTaskId,
  sourceFileId,
  targetPlayerBox,
  reusable = false,
  reusableSource = null,
}: PrepareVideoFileOptions): Promise<PreparedVideoFile> => {
  const duration = await loadVideoDuration(previewUrl);
  const candidateFrames = await extractSelectionFrameCandidates(previewUrl, duration);

  let selectionFrame = candidateFrames[0] ?? await extractSelectionFrame(previewUrl, 0);
  let selectionFrameConfirmed = false;

  if (
    targetPlayerBox &&
    Number.isFinite(targetPlayerBox.selectionTime) &&
    targetPlayerBox.selectionTime >= 0
  ) {
    selectionFrame = await extractSelectionFrame(previewUrl, targetPlayerBox.selectionTime);
    selectionFrameConfirmed = true;
  }

  const normalizedCandidateFrames = (
    candidateFrames.some((candidateFrame) => areFramesNear(candidateFrame, selectionFrame))
      ? candidateFrames
      : [selectionFrame, ...candidateFrames]
  ).sort((left, right) => left.time - right.time);

  return {
    videoFile: {
      file,
      preview: previewUrl,
      duration,
      size,
      type,
      name,
      selectionFrame,
      selectionFrameConfirmed,
      targetPlayerBox: syncSelectionBoxToFrame(targetPlayerBox ?? null, selectionFrame),
      sourceTaskId,
      sourceFileId,
      sourceUrl: previewUrl,
      reusable,
      reusableSource,
      uploadStatus: reusable || sourceFileId ? 'uploaded' : file ? 'uploading' : 'local',
    },
    candidateFrames: normalizedCandidateFrames.length > 0 ? normalizedCandidateFrames : [selectionFrame],
    initialPreviewTime: selectionFrameConfirmed ? selectionFrame.time : 0,
  };
};

interface VideoUploadProps {
  onFileSelect: (file: VideoFile | null) => void;
  onExitReusableSource?: () => void;
  initialSource?: ReusableVideoSource | null;
  loading?: boolean;
  progress?: number;
  disabled?: boolean;
  className?: string;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({
  onFileSelect,
  onExitReusableSource,
  initialSource = null,
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
  const [selectionFrameCandidates, setSelectionFrameCandidates] = useState<SelectionFrame[]>([]);
  const [remoteCandidateStatus, setRemoteCandidateStatus] = useState<'idle' | 'uploading' | 'loading' | 'ready' | 'error'>('idle');
  const [remoteUploadProgress, setRemoteUploadProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previewVideoRef = useRef<HTMLVideoElement>(null);
  const initializedSourceKeyRef = useRef<string | null>(null);
  const selectedFileRef = useRef<VideoFile | null>(null);
  const selectionFrameCandidatesRef = useRef<SelectionFrame[]>([]);
  const selectionRequestTokenRef = useRef(0);
  const { error, hasError, handleError, clearError } = useErrorHandler();

  const syncSelectedFile = useCallback((nextFile: VideoFile | null) => {
    selectedFileRef.current = nextFile;
    setSelectedFile(nextFile);
    onFileSelect(nextFile);
  }, [onFileSelect]);

  useEffect(() => {
    selectedFileRef.current = selectedFile;
  }, [selectedFile]);

  useEffect(() => {
    selectionFrameCandidatesRef.current = selectionFrameCandidates;
  }, [selectionFrameCandidates]);

  const applySmartSelectionFrames = useCallback((smartFrames: SelectionFrame[]) => {
    const latestFile = selectedFileRef.current;
    if (!latestFile || smartFrames.length === 0) {
      return;
    }

    const normalizedSmartFrames = smartFrames.map((frame) => normalizeSelectionFrame(frame, {
      source: 'smart',
      recommended: true,
    }));
    const mergedFrames = mergeSelectionFrameCandidates(
      normalizedSmartFrames,
      selectionFrameCandidatesRef.current,
      latestFile.selectionFrame,
    );
    setSelectionFrameCandidates(mergedFrames);

    const matchingSelectionFrame = latestFile.selectionFrame
      ? mergedFrames.find((candidate) => areFramesNear(candidate, latestFile.selectionFrame as SelectionFrame))
      : mergedFrames[0];
    const nextSelectionFrame = latestFile.selectionFrameConfirmed
      ? (matchingSelectionFrame ?? latestFile.selectionFrame)
      : (mergedFrames[0] ?? latestFile.selectionFrame);

    const nextTargetPlayerBox = syncSelectionBoxToFrame(
      latestFile.targetPlayerBox
      ?? (
        latestFile.selectionFrameConfirmed
          ? matchingSelectionFrame?.suggestedBox ?? null
          : null
      ),
      nextSelectionFrame,
    );

    syncSelectedFile({
      ...latestFile,
      selectionFrame: nextSelectionFrame,
      targetPlayerBox: nextTargetPlayerBox,
    });
  }, [syncSelectedFile]);

  const loadRemoteSelectionFrames = useCallback(async (
    fileId: string,
    requestToken: number,
  ) => {
    setRemoteCandidateStatus('loading');
    try {
      const smartFrames = await withRetry(
        () => ApiService.getSelectionFrameCandidates(fileId),
        REMOTE_SELECTION_FRAME_RETRY_ATTEMPTS,
        REMOTE_SELECTION_FRAME_RETRY_DELAY_MS,
      );
      if (selectionRequestTokenRef.current !== requestToken) {
        return;
      }

      if (smartFrames.length > 0) {
        applySmartSelectionFrames(smartFrames);
        setRemoteCandidateStatus('ready');
        return;
      }

      setRemoteCandidateStatus('idle');
    } catch {
      if (selectionRequestTokenRef.current === requestToken) {
        setRemoteCandidateStatus('error');
      }
    }
  }, [applySmartSelectionFrames]);

  const handleFileSelect = useCallback(async (file: File) => {
    clearError();
    setIsProcessing(true);
    setRemoteCandidateStatus('uploading');
    setRemoteUploadProgress(0);
    let preview: string | null = null;
    const requestToken = selectionRequestTokenRef.current + 1;
    selectionRequestTokenRef.current = requestToken;

    try {
      const validation = validateVideoFile(file);
      if (!validation.valid) {
        throw new Error(validation.error || '文件验证失败');
      }

      preview = URL.createObjectURL(file);
      const prepared = await prepareVideoFile({
        previewUrl: preview,
        name: file.name,
        size: file.size,
        type: file.type,
        file,
      });

      syncSelectedFile(prepared.videoFile);
      setSelectionFrameCandidates(prepared.candidateFrames);
      setPreviewUrl((currentPreviewUrl) => {
        releasePreviewUrl(currentPreviewUrl);
        return preview as string;
      });
      setPreviewCurrentTime(prepared.initialPreviewTime);

      void (async () => {
        let uploadedFileId: string | null = null;
        try {
          const uploadResult = await ApiService.uploadVideo(
            { file },
            (progressValue) => {
              if (selectionRequestTokenRef.current !== requestToken) {
                return;
              }
              setRemoteUploadProgress(progressValue);
            },
          );
          if (selectionRequestTokenRef.current !== requestToken) {
            return;
          }

          uploadedFileId = uploadResult.fileId;
          const latestFile = selectedFileRef.current ?? prepared.videoFile;
          syncSelectedFile({
            ...latestFile,
            sourceFileId: uploadResult.fileId,
            uploadStatus: 'uploaded',
          });
        } catch {
          if (selectionRequestTokenRef.current !== requestToken) {
            return;
          }

          const latestFile = selectedFileRef.current ?? prepared.videoFile;
          syncSelectedFile({
            ...latestFile,
            uploadStatus: 'failed',
          });
          setRemoteCandidateStatus('error');
          return;
        }

        if (uploadedFileId) {
          await loadRemoteSelectionFrames(uploadedFileId, requestToken);
        }
      })();
    } catch (err) {
      releasePreviewUrl(preview ?? undefined);
      setRemoteCandidateStatus('idle');
      handleError(err instanceof Error ? err : new Error('处理文件时发生未知错误'));
    } finally {
      setIsProcessing(false);
    }

    return false;
  }, [syncSelectedFile, handleError, clearError, loadRemoteSelectionFrames]);

  useEffect(() => {
    let cancelled = false;
    const sourceKey = initialSource ? `${initialSource.taskId}:${initialSource.fileId}` : null;

    if (!initialSource || !sourceKey || initializedSourceKeyRef.current === sourceKey) {
      return () => {
        cancelled = true;
      };
    }

    clearError();
    setIsProcessing(true);
    setRemoteCandidateStatus(initialSource?.fileId ? 'loading' : 'idle');
    setRemoteUploadProgress(100);
    const requestToken = selectionRequestTokenRef.current + 1;
    selectionRequestTokenRef.current = requestToken;

    void prepareVideoFile({
      previewUrl: initialSource.sourceStreamUrl,
      name: initialSource.filename,
      size: initialSource.fileSize,
      type: initialSource.mimeType || 'video/mp4',
      sourceTaskId: initialSource.taskId,
      sourceFileId: initialSource.fileId,
      targetPlayerBox: initialSource.targetPlayerBox ?? null,
      reusable: true,
      reusableSource: initialSource,
    })
      .then((prepared) => {
        if (cancelled) {
          return;
        }

        initializedSourceKeyRef.current = sourceKey;
        syncSelectedFile(prepared.videoFile);
        setSelectionFrameCandidates(prepared.candidateFrames);
        setPreviewUrl((currentPreviewUrl) => {
          releasePreviewUrl(currentPreviewUrl);
          return initialSource.sourceStreamUrl;
        });
        setPreviewCurrentTime(prepared.initialPreviewTime);
        if (prepared.videoFile.sourceFileId) {
          void loadRemoteSelectionFrames(prepared.videoFile.sourceFileId, requestToken);
        }
      })
      .catch((nextError) => {
        if (cancelled) {
          return;
        }
        setRemoteCandidateStatus('error');
        handleError(nextError instanceof Error ? nextError : new Error('加载源视频失败'));
      })
      .finally(() => {
        if (!cancelled) {
          setIsProcessing(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [initialSource, syncSelectedFile, handleError, clearError, loadRemoteSelectionFrames]);

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
        selectionFrameConfirmed: true,
        targetPlayerBox: null,
      };

      syncSelectedFile(updatedFile);
      setSelectionFrameCandidates((currentCandidates) => {
        return mergeSelectionFrameCandidates(currentCandidates, [selectionFrame], selectionFrame);
      });
      setPreviewCurrentTime(selectionFrame.time);
    } catch (err) {
      handleError(err instanceof Error ? err : new Error('截取当前画面失败'));
    } finally {
      setIsCapturingFrame(false);
    }
  }, [selectedFile, previewUrl, syncSelectedFile, handleError, clearError]);

  const handleRemoveFile = useCallback(() => {
    const wasReusable = Boolean(selectedFile?.reusable);
    selectionRequestTokenRef.current += 1;
    releasePreviewUrl(previewUrl);
    syncSelectedFile(null);
    setPreviewUrl('');
    setPreviewCurrentTime(0);
    setSelectionFrameCandidates([]);
    setRemoteCandidateStatus('idle');
    setRemoteUploadProgress(0);
    clearError();
    if (wasReusable) {
      onExitReusableSource?.();
    }
  }, [selectedFile?.reusable, previewUrl, syncSelectedFile, clearError, onExitReusableSource]);

  const seekPreviewTo = useCallback((nextTime: number) => {
    const previewVideo = previewVideoRef.current;
    if (!previewVideo) {
      return;
    }

    const clampedTime = clampTime(nextTime, selectedFile?.duration);
    previewVideo.currentTime = clampedTime;
    setPreviewCurrentTime(clampedTime);
  }, [selectedFile?.duration]);

  const seekPreviewBy = useCallback((offsetSeconds: number) => {
    const previewVideo = previewVideoRef.current;
    if (!previewVideo) {
      return;
    }

    seekPreviewTo((previewVideo.currentTime || 0) + offsetSeconds);
  }, [seekPreviewTo]);

  const handleSelectCandidateFrame = useCallback((selectionFrame: SelectionFrame) => {
    if (!selectedFile) {
      return;
    }

    const updatedFile: VideoFile = {
      ...selectedFile,
      selectionFrame,
      selectionFrameConfirmed: true,
      targetPlayerBox: syncSelectionBoxToFrame(selectionFrame.suggestedBox ?? null, selectionFrame),
    };

    syncSelectedFile(updatedFile);
    setPreviewCurrentTime(selectionFrame.time);
    seekPreviewTo(selectionFrame.time);
  }, [seekPreviewTo, selectedFile, syncSelectedFile]);

  const hasConfirmedSelectionFrame = Boolean(selectedFile?.selectionFrame && selectedFile.selectionFrameConfirmed);

  useEffect(() => {
    return () => {
      releasePreviewUrl(previewUrl);
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
              <h3 className="text-2xl font-semibold tracking-tight text-white">
                {initialSource && isProcessing ? '正在加载上一次的视频' : '上传篮球视频'}
              </h3>
              <p className="mx-auto max-w-xl text-sm leading-6 text-slate-300 sm:text-base">
                {initialSource && isProcessing
                  ? '系统正在直接复用上一次处理的原始比赛视频，稍后你就可以重新选择截图并重跑。'
                  : '拖拽文件到这里，或点击选择本地比赛视频。系统会先秒出本地候选截图，再在后台补充 AI 推荐截图。你只需要选中目标球员清晰出镜的一张画面，再框选这个人。'}
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
                variant={selectedFile?.reusable ? 'secondary' : 'danger-soft'}
                onClick={handleRemoveFile}
                isDisabled={loading || isCapturingFrame}
                size="sm"
              >
                <span className="inline-flex items-center gap-2">
                  <Trash2 size={14} />
                  {selectedFile?.reusable ? '改用新视频' : '移除'}
                </span>
              </Button>
            </div>
          </div>

          <Alert status="accent">
            <div className="font-medium text-current">
              系统会先给出本地候选截图，再补充 AI 推荐截图。优先看带 “AI 推荐” 的候选；如果多个候选都清晰，再优先选择更早出镜的一帧。没有确认这一步前，不会进入下一步框人。
            </div>
          </Alert>

          {remoteCandidateStatus === 'uploading' ? (
            <Alert status="warning">
              <div className="font-medium text-current">
                正在后台上传视频（{Math.max(remoteUploadProgress, 1)}%）并准备 AI 推荐截图。当前先展示本地候选截图，你可以先开始挑图。
              </div>
            </Alert>
          ) : null}

          {remoteCandidateStatus === 'loading' ? (
            <Alert status="warning">
              <div className="font-medium text-current">
                视频已上传，正在生成 AI 推荐截图。生成完成后，候选区会自动刷新。
              </div>
            </Alert>
          ) : null}

          {remoteCandidateStatus === 'ready' ? (
            <Alert status="success">
              <div className="font-medium text-current">
                AI 推荐截图已就绪。优先看带 “AI 推荐” 的候选，系统还会尽量预填一个人物框供你微调。
              </div>
            </Alert>
          ) : null}

          {remoteCandidateStatus === 'error' ? (
            <Alert status="warning">
              <div className="font-medium text-current">
                AI 推荐截图暂时没有生成成功。当前仍然可以直接使用本地候选截图和手动补选继续处理。
              </div>
            </Alert>
          ) : null}

          <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-4">
            <div className="mb-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <ImagePlus size={18} className="text-orange-500" />
                  <span className="font-semibold text-slate-900">先选一张框选起始截图</span>
                </div>
                <p className="mb-0 mt-2 text-sm text-slate-600">
                  AI 推荐截图会排在前面；其余本地候选截图继续保留，作为兜底。系统会先用你选中的这一帧建立人物参考，必要时再自动前移一点去补抓更早回合，所以还是尽量选择目标球员更早且更清晰的出镜画面。
                </p>
              </div>
            </div>

            {selectionFrameCandidates.length > 0 ? (
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {selectionFrameCandidates.map((candidateFrame, index) => {
                  const isSelected = Boolean(
                    selectedFile.selectionFrame && areFramesNear(selectedFile.selectionFrame, candidateFrame),
                  );
                  const isConfirmedCandidate = isSelected && hasConfirmedSelectionFrame;

                  return (
                    <button
                      key={`${candidateFrame.time}-${index}`}
                      type="button"
                      className={`overflow-hidden rounded-2xl border text-left transition ${
                        isConfirmedCandidate
                          ? 'border-orange-300 bg-orange-50 shadow-[0_14px_40px_rgba(249,115,22,0.16)]'
                          : isSelected
                            ? 'border-slate-300 bg-white shadow-[0_10px_30px_rgba(15,23,42,0.10)]'
                            : 'border-slate-200 bg-white/90 hover:border-orange-200 hover:bg-orange-50/50'
                      }`}
                      onClick={() => handleSelectCandidateFrame(candidateFrame)}
                      disabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      <img
                        src={candidateFrame.imageUrl}
                        alt={`候选截图 ${index + 1}`}
                        className="aspect-video w-full bg-black object-cover"
                      />
                      <div className="space-y-3 p-3">
                        <div className="flex items-center justify-between gap-3">
                          <div>
                            <div className="text-sm font-semibold text-slate-900">候选截图 {index + 1}</div>
                            <div className="mt-1 text-sm text-slate-500">
                              {formatDuration(candidateFrame.time)}（{candidateFrame.time.toFixed(2)}s）
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {candidateFrame.recommended ? (
                              <Chip variant="soft" color="success">AI 推荐</Chip>
                            ) : null}
                            {!candidateFrame.recommended && index === 0 ? (
                              <Chip variant="soft" color="warning">本地兜底</Chip>
                            ) : null}
                          </div>
                        </div>
                        <div className="text-sm text-slate-600">
                          {isConfirmedCandidate
                            ? '当前已选中这张截图，下一步直接框人。'
                            : candidateFrame.suggestedBox
                              ? '点击后会带上系统预填的人物框，你只需要微调。'
                              : '点击这张截图，直接进入下一步框人。'}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="rounded-2xl border border-dashed border-slate-200 bg-white/90 px-4 py-10 text-center text-sm text-slate-500">
                正在准备候选截图...
              </div>
            )}

            <div className="mt-4 space-y-4">
              <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-slate-900">
                      <Film size={18} className="text-orange-500" />
                      <span className="font-semibold">候选都不合适？手动补选一帧</span>
                    </div>
                    <p className="mt-2 text-sm text-slate-600">
                      只有在上面的候选截图都不合适时，才需要拖动视频自己补选。补选时仍然优先选择目标球员更早、更清晰的出镜画面。
                    </p>
                  </div>
                  <Button
                    variant="secondary"
                    onClick={() => void handleCaptureCurrentFrame()}
                    isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                  >
                    <span className="inline-flex items-center gap-2">
                      <Film size={16} />
                      {isCapturingFrame ? '截取中...' : '使用当前播放位置'}
                    </span>
                  </Button>
                </div>

                {previewUrl ? (
                  <video
                    ref={previewVideoRef}
                    src={previewUrl}
                    crossOrigin="anonymous"
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

                <div className="mt-4 rounded-2xl border border-slate-200 bg-slate-50/80 p-3">
                  <div className="mb-3 flex items-center justify-between gap-3 text-sm text-slate-600">
                    <span>当前播放位置：{formatDuration(previewCurrentTime)}（{previewCurrentTime.toFixed(2)}s）</span>
                    <span>视频总时长：{selectedFile.duration ? formatDuration(selectedFile.duration) : '未知'}</span>
                  </div>

                  <input
                    type="range"
                    min={0}
                    max={selectedFile.duration || 0}
                    step={0.1}
                    value={previewCurrentTime}
                    onChange={(event) => {
                      seekPreviewTo(Number(event.currentTarget.value));
                    }}
                    className="hero-range w-full"
                    disabled={!selectedFile.duration}
                  />

                  <div className="mt-3 flex flex-wrap gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewTo(0)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      开头
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewBy(-5)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      <span className="inline-flex items-center gap-1">
                        <ChevronLeft size={14} />
                        退 5s
                      </span>
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewBy(-1)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      退 1s
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewBy(1)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      进 1s
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewBy(5)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      <span className="inline-flex items-center gap-1">
                        进 5s
                        <ChevronRight size={14} />
                      </span>
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => seekPreviewTo(selectedFile.duration || 0)}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame || !selectedFile.duration}
                    >
                      结尾
                    </Button>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
                <div className="rounded-2xl border border-slate-200 bg-white/90 p-4">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-slate-900">当前已确认的框选起始帧</div>
                      <div className="mt-1 text-sm text-slate-600">
                        {selectedFile.selectionFrame
                          ? `${formatDuration(selectedFile.selectionFrame.time)}（${selectedFile.selectionFrame.time.toFixed(2)}s）`
                          : '尚未确认'}
                      </div>
                    </div>
                    <Chip variant="soft" color={hasConfirmedSelectionFrame ? 'success' : 'warning'}>
                      {hasConfirmedSelectionFrame ? '已确认' : '未确认'}
                    </Chip>
                  </div>

                  {selectedFile.selectionFrame ? (
                    <img
                      src={selectedFile.selectionFrame.imageUrl}
                      alt="已确认的框选起始帧"
                      className="w-full rounded-2xl border border-slate-200 bg-black object-contain shadow-[0_14px_40px_rgba(15,23,42,0.12)]"
                    />
                  ) : null}

                  <div className="mt-3 text-sm text-slate-600">
                    这张静态画面才是下一步框选目标球员时真正使用的帧。
                  </div>
                </div>

                <div className="rounded-2xl border border-orange-100 bg-orange-50/80 p-4 text-sm text-slate-700">
                  <div className="mb-2 font-semibold text-slate-900">选帧建议</div>
                  <ul className="space-y-2 leading-6">
                    <li>优先选目标球员最早且清晰出镜的一帧，减少漏掉前面回合的风险。</li>
                    <li>选择完整出镜、没有严重遮挡、身体轮廓清楚的一帧。</li>
                    <li>尽量避开背身、模糊、运动拖影明显的时刻。</li>
                  </ul>

                  {hasConfirmedSelectionFrame ? (
                    <Button
                      variant="secondary"
                      size="sm"
                      className="mt-4"
                      onClick={() => {
                        const updatedFile: VideoFile = {
                          ...selectedFile,
                          selectionFrameConfirmed: false,
                          targetPlayerBox: null,
                        };
                        syncSelectedFile(updatedFile);
                      }}
                      isDisabled={disabled || loading || isProcessing || isCapturingFrame}
                    >
                      <span className="inline-flex items-center gap-2">
                        <RotateCcw size={14} />
                        重新选择起始帧
                      </span>
                    </Button>
                  ) : null}
                </div>
              </div>
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
