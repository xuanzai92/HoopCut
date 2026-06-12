import React, { useRef, useState, useEffect } from 'react';
import { Card, Button, Slider } from '@heroui/react';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import type { PlayerSelectionBox } from '@/types';

interface VideoPlayerSelectorProps {
  videoUrl: string;
  value: PlayerSelectionBox | null;
  onChange: (selection: PlayerSelectionBox | null) => void;
  disabled?: boolean;
}

interface Point {
  x: number;
  y: number;
}

export const VideoPlayerSelector: React.FC<VideoPlayerSelectorProps> = ({
  videoUrl,
  value,
  onChange,
  disabled = false,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDrawing, setIsDrawing] = useState(false);
  const [startPoint, setStartPoint] = useState<Point | null>(null);
  const [tempBox, setTempBox] = useState<PlayerSelectionBox | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [videoUrl]);

  const togglePlayPause = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
  };

  const handleSeek = (val: number | number[]) => {
    const video = videoRef.current;
    if (!video) return;

    const time = Array.isArray(val) ? val[0] : val;
    video.currentTime = time;
    setCurrentTime(time);
  };

  const skipBackward = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, video.currentTime - 5);
  };

  const skipForward = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.min(duration, video.currentTime + 5);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getRelativePoint = (clientX: number, clientY: number): Point | null => {
    const video = videoRef.current;
    const container = containerRef.current;
    if (!video || !container) return null;

    const rect = container.getBoundingClientRect();
    const scaleX = video.videoWidth / rect.width;
    const scaleY = video.videoHeight / rect.height;

    return {
      x: Math.round((clientX - rect.left) * scaleX),
      y: Math.round((clientY - rect.top) * scaleY),
    };
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (disabled || isPlaying) return;

    const point = getRelativePoint(e.clientX, e.clientY);
    if (!point) return;

    setIsDrawing(true);
    setStartPoint(point);
    setTempBox(null);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDrawing || !startPoint || !videoRef.current) return;

    const point = getRelativePoint(e.clientX, e.clientY);
    if (!point) return;

    const x = Math.min(startPoint.x, point.x);
    const y = Math.min(startPoint.y, point.y);
    const width = Math.abs(point.x - startPoint.x);
    const height = Math.abs(point.y - startPoint.y);

    setTempBox({
      x,
      y,
      width,
      height,
      frameWidth: videoRef.current.videoWidth,
      frameHeight: videoRef.current.videoHeight,
      selectionTime: currentTime,
      selectionFrame: Math.round(currentTime * 30),
    });
  };

  const handleMouseUp = () => {
    if (!isDrawing) return;

    setIsDrawing(false);
    setStartPoint(null);

    if (tempBox && tempBox.width > 20 && tempBox.height > 20) {
      onChange(tempBox);
    }
    setTempBox(null);
  };

  const clearSelection = () => {
    onChange(null);
  };

  const displayBox = tempBox || value;

  return (
    <Card className="p-4">
      <div className="space-y-3">
        <div
          ref={containerRef}
          className="relative bg-black rounded-lg overflow-hidden cursor-crosshair"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full"
            preload="metadata"
          />

          {!isPlaying && displayBox && (
            <div
              className="absolute border-2 border-primary bg-primary/10 pointer-events-none"
              style={{
                left: `${(displayBox.x / displayBox.frameWidth) * 100}%`,
                top: `${(displayBox.y / displayBox.frameHeight) * 100}%`,
                width: `${(displayBox.width / displayBox.frameWidth) * 100}%`,
                height: `${(displayBox.height / displayBox.frameHeight) * 100}%`,
              }}
            />
          )}

          {!isPlaying && !value && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/20 pointer-events-none">
              <p className="text-white text-sm bg-black/60 px-3 py-1 rounded">
                拖动鼠标框选目标球员
              </p>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3">
          <Button
            isIconOnly
            size="sm"
            variant="light"
            onPress={skipBackward}
            isDisabled={disabled}
          >
            <SkipBack size={16} />
          </Button>

          <Button
            isIconOnly
            size="sm"
            color="primary"
            onPress={togglePlayPause}
            isDisabled={disabled}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
          </Button>

          <Button
            isIconOnly
            size="sm"
            variant="light"
            onPress={skipForward}
            isDisabled={disabled}
          >
            <SkipForward size={16} />
          </Button>

          <div className="flex-1 flex items-center gap-2">
            <span className="text-xs text-gray-500 min-w-[50px]">
              {formatTime(currentTime)}
            </span>
            <Slider
              size="sm"
              step={0.1}
              maxValue={duration}
              value={currentTime}
              onChange={handleSeek}
              className="flex-1"
              isDisabled={disabled}
            />
            <span className="text-xs text-gray-500 min-w-[50px]">
              {formatTime(duration)}
            </span>
          </div>

          {value && (
            <Button
              size="sm"
              variant="light"
              color="danger"
              onPress={clearSelection}
              isDisabled={disabled}
            >
              清除
            </Button>
          )}
        </div>

        {value && (
          <p className="text-xs text-gray-600 text-center">
            已选择 {formatTime(value.selectionTime)} 处的目标球员
          </p>
        )}
      </div>
    </Card>
  );
};
