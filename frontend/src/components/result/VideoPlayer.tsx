import React, { useEffect, useRef, useState } from 'react';
import { Button, Card, Toast } from '@heroui/react';
import { Download, Expand, Maximize, Minimize, Pause, PictureInPicture2, Play, Volume2, VolumeX } from 'lucide-react';
import { formatDuration } from '@/utils';

interface VideoPlayerProps {
  src: string;
  poster?: string;
  title?: string;
  onDownload?: () => void;
  className?: string;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  poster,
  title,
  onDownload,
  className = '',
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const controlsTimeoutRef = useRef<number | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [showControls, setShowControls] = useState(true);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
      setIsLoading(false);
    };

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);
    const handleVolumeChange = () => {
      setVolume(video.volume);
      setIsMuted(video.muted);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('ended', handleEnded);
    video.addEventListener('volumechange', handleVolumeChange);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('ended', handleEnded);
      video.removeEventListener('volumechange', handleVolumeChange);
    };
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) {
      return;
    }

    setIsLoading(true);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setShowControls(true);

    video.pause();
    video.load();
  }, [src]);

  useEffect(() => {
    if (controlsTimeoutRef.current !== null) {
      window.clearTimeout(controlsTimeoutRef.current);
      controlsTimeoutRef.current = null;
    }

    if (isPlaying && showControls) {
      controlsTimeoutRef.current = window.setTimeout(() => {
        setShowControls(false);
      }, 2500);
    }

    return () => {
      if (controlsTimeoutRef.current !== null) {
        window.clearTimeout(controlsTimeoutRef.current);
      }
    };
  }, [isPlaying, showControls]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (isPlaying) {
      video.pause();
      return;
    }

    void video.play().catch(() => {
      Toast.toast.danger('视频播放失败');
    });
  };

  const seekTo = (time: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = time;
    setCurrentTime(time);
  };

  const changeVolume = (value: number) => {
    const video = videoRef.current;
    if (!video) return;
    video.volume = value;
    video.muted = value === 0;
    setVolume(value);
    setIsMuted(value === 0);
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;
    video.muted = !video.muted;
    setIsMuted(video.muted);
  };

  const toggleFullscreen = async () => {
    const container = containerRef.current;
    if (!container) return;

    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await container.requestFullscreen();
      }
    } catch {
      Toast.toast.danger('全屏操作失败');
    }
  };

  const handleMouseMove = () => setShowControls(true);

  const handlePictureInPicture = async () => {
    const video = videoRef.current;
    if (!video || !('requestPictureInPicture' in video)) {
      Toast.toast.danger('当前浏览器不支持画中画');
      return;
    }

    try {
      await video.requestPictureInPicture();
    } catch {
      Toast.toast.danger('进入画中画失败');
    }
  };

  return (
    <Card className={`overflow-hidden border border-white/10 bg-slate-950 text-white shadow-[0_35px_120px_rgba(15,23,42,0.45)] ${className}`}>
      {title ? (
        <div className="border-b border-white/10 px-5 py-4">
          <h3 className="text-lg font-semibold">{title}</h3>
        </div>
      ) : null}

      <div
        ref={containerRef}
        className={`relative bg-black ${isFullscreen ? 'h-screen' : 'aspect-video'}`}
        onMouseMove={handleMouseMove}
      >
        <video
          ref={videoRef}
          src={src}
          poster={poster}
          className="h-full w-full object-contain"
          preload="metadata"
          onClick={togglePlay}
        />

        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black/55 text-sm text-white/80">
            正在加载视频...
          </div>
        ) : null}

        {!isPlaying && !isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black/25">
            <button
              type="button"
              onClick={togglePlay}
              className="flex h-20 w-20 items-center justify-center rounded-full border border-white/15 bg-white/10 text-white backdrop-blur transition hover:bg-white/15"
            >
              <Play size={34} />
            </button>
          </div>
        ) : null}

        <div
          className={`absolute inset-x-0 bottom-0 bg-gradient-to-t from-black via-black/75 to-transparent p-4 transition-opacity duration-300 ${
            showControls ? 'opacity-100' : 'pointer-events-none opacity-0'
          }`}
        >
          <div className="mb-4">
            <input
              type="range"
              min={0}
              max={duration || 0}
              step={0.1}
              value={currentTime}
              onChange={(event) => seekTo(Number(event.target.value))}
              className="hero-range"
            />
          </div>

          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex flex-wrap items-center gap-2 text-sm text-white/85">
              <Button variant="ghost" onClick={togglePlay}>
                {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              </Button>
              <Button variant="ghost" onClick={toggleMute}>
                {isMuted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
              </Button>
              <div className="w-24">
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={isMuted ? 0 : volume}
                  onChange={(event) => changeVolume(Number(event.target.value))}
                  className="hero-range"
                />
              </div>
              <span>{formatDuration(currentTime)} / {formatDuration(duration)}</span>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {onDownload ? (
                <Button variant="ghost" onClick={onDownload}>
                  <Download size={16} />
                </Button>
              ) : null}
              <Button variant="ghost" onClick={() => void handlePictureInPicture()}>
                <PictureInPicture2 size={16} />
              </Button>
              <Button variant="ghost" onClick={() => void toggleFullscreen()}>
                {isFullscreen ? <Minimize size={16} /> : <Maximize size={16} />}
              </Button>
              <Button variant="ghost" onClick={() => seekTo(Math.min(duration, currentTime + 10))}>
                <Expand size={16} />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};
