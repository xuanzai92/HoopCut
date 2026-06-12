import React, { useState, useEffect } from 'react';
import { Button, Card } from '@heroui/react';
import { ArrowRight } from 'lucide-react';
import { VideoUpload } from '@/components/upload/VideoUpload';
import { VideoPlayerSelector } from '@/components/ui/VideoPlayerSelector';
import { ConfigPanel } from '@/components/ui/ConfigPanel';
import ErrorAlert from '@/components/common/ErrorAlert';
import { useTaskStore } from '@/store';
import { useUploadVideo, useProcessVideo } from '@/services';
import type { PlayerSelectionBox, ReusableVideoSource, VideoFile } from '@/types';
import { validateTargetPlayerBox } from '@/utils';
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
  const [videoUrl, setVideoUrl] = useState<string>('');
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
    setTargetPlayerBox(initialSource.targetPlayerBox || null);
    setUploadProgress(0);
    setVideoUrl(initialSource.sourceStreamUrl || '');
  }, [initialSource]);

  const handleFileSelect = (file: VideoFile | null) => {
    if (file?.file && initialSource) {
      onExitReusableSource?.();
    }

    setSelectedFile(file);
    setTargetPlayerBox(file?.targetPlayerBox ?? null);
    setUploadProgress(0);

    if (file?.file) {
      const url = URL.createObjectURL(file.file);
      setVideoUrl(url);
    } else {
      setVideoUrl('');
    }
  };

  const handleStartProcessing = withLoading(
    withErrorHandling(
      async () => {
        if (!selectedFile && !initialSource) {
          throw new Error('请先上传视频');
        }

        const selectionValidation = validateTargetPlayerBox(targetPlayerBox);
        if (!selectionValidation.valid) {
          throw new Error(selectionValidation.error || '请先框选目标球员');
        }

        setUploadProgress(0);

        let fileId = initialSource?.sourceFileId || selectedFile?.sourceFileId;
        if (!fileId && selectedFile?.file) {
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
          throw new Error('上传失败，请重试');
        }

        createTask({
          file: selectedFile?.file,
          filename: selectedFile?.filename || initialSource?.filename || '',
          targetPlayerBox,
        });

        const processResp = await processMutation.mutateAsync({
          fileId,
          mode: 'auto',
          targetPlayerBox: targetPlayerBox!,
        });

        navigate(`/progress/${processResp.taskId}`);
      },
      { showMessage: false },
    ),
  );

  const isProcessing = uploadMutation.isPending || processMutation.isPending || customLoading;
  const hasVideo = Boolean(videoUrl);
  const canStartProcessing = hasVideo && targetPlayerBox && !isProcessing;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {hasError && error && (
        <ErrorAlert
          title="处理失败"
          message={error.message}
          type="error"
          showIcon
          closable
          onClose={clearError}
        />
      )}

      <VideoUpload
        onFileSelect={handleFileSelect}
        onExitReusableSource={onExitReusableSource}
        initialSource={initialSource}
        loading={isProcessing}
        progress={uploadProgress}
        disabled={isProcessing}
      />

      {hasVideo && (
        <VideoPlayerSelector
          videoUrl={videoUrl}
          value={targetPlayerBox}
          onChange={setTargetPlayerBox}
          disabled={isProcessing}
        />
      )}

      {hasVideo && (
        <div className="space-y-4">
          <Button
            color="primary"
            size="lg"
            className="w-full"
            onPress={handleStartProcessing}
            isDisabled={!canStartProcessing}
            isLoading={isProcessing}
            endContent={<ArrowRight size={20} />}
          >
            {isProcessing ? '处理中...' : '开始处理'}
          </Button>
          <p className="text-sm text-gray-500 text-center">
            默认：进球前保留 6 秒，进球后保留 2 秒
          </p>
        </div>
      )}
    </div>
  );
};
