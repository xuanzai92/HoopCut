import React, { useState, useEffect } from 'react';
import { Button, Card } from '@heroui/react';
import { ArrowRight, RefreshCcw } from 'lucide-react';
import { VideoUpload } from '@/components/upload/VideoUpload';
import { PlayerSelector } from '@/components/ui/PlayerSelector';
import ErrorAlert from '@/components/common/ErrorAlert';
import { useTaskStore } from '@/store';
import { useUploadVideo, useProcessVideo } from '@/services';
import type { PlayerSelectionBox, ReusableVideoSource, VideoFile } from '@/types';
import { syncSelectionBoxToFrame, validateTargetPlayerBox } from '@/utils';
import { useErrorHandler, useLoading } from '@/hooks';
import { useNavigate } from 'react-router-dom';

interface AutoWorkflowProps {
  initialSource?: ReusableVideoSource | null;
  onExitReusableSource?: () => void;
}

export const AutoWorkflow: React.FC<AutoWorkflowProps> = ({
  initialSource = null,
  onExitReusableSource,
}) => {
  const [selectedFile, setSelectedFile] = useState<VideoFile | null>(null);
  const [targetPlayerBox, setTargetPlayerBox] = useState<PlayerSelectionBox | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const hasConfirmedSelectionFrame = Boolean(selectedFile?.selectionFrame && selectedFile.selectionFrameConfirmed);
  const { createTask } = useTaskStore();
  const { error, hasError, clearError, withErrorHandling } = useErrorHandler();
  const { loading: customLoading, withLoading } = useLoading();
  const navigate = useNavigate();
  const uploadMutation = useUploadVideo();
  const processMutation = useProcessVideo();

  useEffect(() => {
    if (!initialSource) {
      return;
    }

    setSelectedFile(null);
    setTargetPlayerBox(null);
    setUploadProgress(0);
  }, [initialSource]);

  const handleFileSelect = (file: VideoFile | null) => {
    if (file?.file && initialSource) {
      onExitReusableSource?.();
    }

    setSelectedFile(file);
    setTargetPlayerBox(syncSelectionBoxToFrame(file?.targetPlayerBox ?? null, file?.selectionFrame));
    setUploadProgress(0);
  };

  const handleTargetPlayerChange = (selection: PlayerSelectionBox | null) => {
    setTargetPlayerBox(syncSelectionBoxToFrame(selection, selectedFile?.selectionFrame));
  };

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
          mode: 'auto',
          targetPlayerBox: effectiveTargetPlayerBox,
        });

        navigate(`/progress/${processResp.taskId}`);
      },
      { showMessage: false },
    ),
  );

  const isProcessing = uploadMutation.isPending || processMutation.isPending || customLoading;

  return (
    <div className="space-y-6">
      {hasError && error ? (
        <div className="w-full">
          <ErrorAlert
            title="自动模式处理失败"
            message={error.message}
            type="error"
            showIcon
            closable
            onClose={clearError}
          />
        </div>
      ) : null}

      <VideoUpload
        onFileSelect={handleFileSelect}
        onExitReusableSource={onExitReusableSource}
        initialSource={initialSource}
        loading={isProcessing}
        progress={uploadProgress}
        disabled={isProcessing}
      />

      {initialSource ? (
        <Card className="border border-indigo-100 bg-indigo-50/80 p-5 shadow-[0_18px_50px_rgba(99,102,241,0.08)]">
          <div className="space-y-2">
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-500">复用源视频</p>
            <h2 className="text-xl font-semibold text-slate-900">已载入上一次处理的视频</h2>
            <p className="text-sm leading-6 text-slate-600">
              当前直接复用原始比赛视频，无需重新上传。你只需要重新选择起始截图或微调人物框选，然后重新开始自动剪辑。
            </p>
            <div className="pt-2">
              <Button variant="secondary" size="sm" onClick={() => onExitReusableSource?.()}>
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

      <Card className="border border-slate-900/5 bg-slate-950 text-white shadow-[0_30px_90px_rgba(15,23,42,0.26)]">
        <div className="space-y-5 p-6">
          <div className="space-y-2">
            <p className="text-sm uppercase tracking-[0.24em] text-orange-300">开始前检查</p>
            <h2 className="text-2xl font-semibold tracking-tight">开始自动剪辑</h2>
            <p className="text-sm leading-6 text-slate-400">
              这条模式会继续跑完整的人物锁定、自动找球和归因逻辑。适合你不想自己一段段找时间点的时候再用。
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

          {initialSource ? (
            <Button variant="ghost" onClick={() => onExitReusableSource?.()} className="w-full">
              <span className="inline-flex items-center gap-2">
                <RefreshCcw size={16} />
                改用新的比赛视频
              </span>
            </Button>
          ) : null}
        </div>
      </Card>
    </div>
  );
};

export default AutoWorkflow;
