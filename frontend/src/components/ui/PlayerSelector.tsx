import React, { useMemo, useRef, useState } from 'react';
import { Alert, Button, Chip } from '@heroui/react';
import { ScanFace, Trash2 } from 'lucide-react';
import { formatDuration } from '@/utils';
import type { PlayerSelectionBox, SelectionFrame } from '@/types';

interface PlayerSelectorProps {
  frame: SelectionFrame;
  value: PlayerSelectionBox | null;
  onChange: (selection: PlayerSelectionBox | null) => void;
  disabled?: boolean;
}

interface FramePoint {
  x: number;
  y: number;
}

const MIN_BOX_SIZE = 20;

const clamp = (value: number, min: number, max: number): number => {
  return Math.min(Math.max(value, min), max);
};

const buildSelectionBox = (
  start: FramePoint,
  end: FramePoint,
  frame: SelectionFrame
): PlayerSelectionBox => {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const width = Math.abs(end.x - start.x);
  const height = Math.abs(end.y - start.y);

  return {
    x: Math.round(x),
    y: Math.round(y),
    width: Math.round(width),
    height: Math.round(height),
    frameWidth: frame.width,
    frameHeight: frame.height,
    selectionTime: frame.time,
    selectionFrame: frame.frame,
  };
};

export const PlayerSelector: React.FC<PlayerSelectorProps> = ({
  frame,
  value,
  onChange,
  disabled = false,
}) => {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const dragStartRef = useRef<FramePoint | null>(null);
  const draftSelectionRef = useRef<PlayerSelectionBox | null>(null);
  const [draftSelection, setDraftSelection] = useState<PlayerSelectionBox | null>(null);

  const displaySelection = draftSelection ?? value;

  const selectionStyle = useMemo(() => {
    if (!displaySelection) {
      return undefined;
    }

    return {
      left: `${(displaySelection.x / frame.width) * 100}%`,
      top: `${(displaySelection.y / frame.height) * 100}%`,
      width: `${(displaySelection.width / frame.width) * 100}%`,
      height: `${(displaySelection.height / frame.height) * 100}%`,
    };
  }, [displaySelection, frame.height, frame.width]);

  const getFramePoint = (clientX: number, clientY: number): FramePoint | null => {
    const wrapper = wrapperRef.current;
    if (!wrapper) {
      return null;
    }

    const rect = wrapper.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
      return null;
    }

    const relativeX = clamp(clientX - rect.left, 0, rect.width);
    const relativeY = clamp(clientY - rect.top, 0, rect.height);

    return {
      x: (relativeX / rect.width) * frame.width,
      y: (relativeY / rect.height) * frame.height,
    };
  };

  const startSelection = (
    point: FramePoint,
    target: HTMLDivElement,
    pointerId?: number
  ) => {
    const nextSelection = buildSelectionBox(point, point, frame);
    dragStartRef.current = point;
    draftSelectionRef.current = nextSelection;
    setDraftSelection(nextSelection);

    if (typeof pointerId === 'number') {
      target.setPointerCapture(pointerId);
    }
  };

  const updateSelection = (point: FramePoint) => {
    const dragStart = dragStartRef.current;
    if (!dragStart) {
      return;
    }

    const nextSelection = buildSelectionBox(dragStart, point, frame);
    draftSelectionRef.current = nextSelection;
    setDraftSelection(nextSelection);
  };

  const completeSelection = (
    point?: FramePoint,
    target?: HTMLDivElement,
    pointerId?: number
  ) => {
    const dragStart = dragStartRef.current;
    if (!dragStart) {
      return;
    }

    let nextSelection = draftSelectionRef.current;
    if (point) {
      nextSelection = buildSelectionBox(dragStart, point, frame);
    }

    if (
      target &&
      typeof pointerId === 'number' &&
      target.hasPointerCapture(pointerId)
    ) {
      target.releasePointerCapture(pointerId);
    }

    dragStartRef.current = null;
    draftSelectionRef.current = null;
    setDraftSelection(null);

    if (!nextSelection) {
      return;
    }

    if (nextSelection.width < MIN_BOX_SIZE || nextSelection.height < MIN_BOX_SIZE) {
      onChange(null);
      return;
    }

    onChange(nextSelection);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (disabled) {
      return;
    }

    const point = getFramePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    startSelection(point, event.currentTarget, event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (disabled || !dragStartRef.current) {
      return;
    }

    const point = getFramePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    updateSelection(point);
  };

  const finishSelection = (event?: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) {
      return;
    }

    let point: FramePoint | undefined;
    if (event) {
      point = getFramePoint(event.clientX, event.clientY) ?? undefined;
    }

    completeSelection(point, event?.currentTarget, event?.pointerId);
  };

  const handleMouseDown = (event: React.MouseEvent<HTMLDivElement>) => {
    if (disabled) {
      return;
    }

    const point = getFramePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    startSelection(point, event.currentTarget);
  };

  const handleMouseMove = (event: React.MouseEvent<HTMLDivElement>) => {
    if (disabled || !dragStartRef.current) {
      return;
    }

    const point = getFramePoint(event.clientX, event.clientY);
    if (!point) {
      return;
    }

    updateSelection(point);
  };

  const handleMouseUp = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStartRef.current) {
      return;
    }

    const point = getFramePoint(event.clientX, event.clientY) ?? undefined;
    completeSelection(point);
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-slate-900">
            <ScanFace size={18} className="text-orange-500" />
            <span className="text-base font-semibold">框选你自己</span>
          </div>
          <p className="text-sm text-slate-500">
            在你选中的出镜画面里拖拽框住完整身体区域。后续人物跟踪会从这个时间点开始初始化目标。
          </p>
        </div>
        <Button
          onClick={() => onChange(null)}
          isDisabled={disabled || !value}
          variant="danger-soft"
        >
          <span className="inline-flex items-center gap-2">
            <Trash2 size={14} />
            清除选区
          </span>
        </Button>
      </div>

      <Alert status="accent">
        <div className="flex flex-col gap-1">
          <div className="font-medium text-current">
            当前框选帧：{formatDuration(frame.time)}（{frame.time.toFixed(2)}s）
          </div>
          <div className="text-sm text-current/80">建议框住头到脚，并尽量包含完整球衣区域。</div>
        </div>
      </Alert>

      <div className="rounded-[28px] border border-slate-200 bg-slate-50/80 p-3 shadow-inner">
        <div
          ref={wrapperRef}
          className={`relative mx-auto w-full overflow-hidden rounded-lg border border-[#D8DAD3] bg-black ${
            disabled ? 'cursor-not-allowed' : 'cursor-crosshair'
          }`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={finishSelection}
          onPointerCancel={finishSelection}
          onPointerLeave={finishSelection}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ maxWidth: '100%', touchAction: 'none' }}
        >
          <img
            src={frame.imageUrl}
            alt="人物框选帧"
            className="block h-auto w-full select-none"
            draggable={false}
          />

          {selectionStyle ? (
            <div
              className="absolute border-2 border-[#FF6B35] bg-[#FF6B35]/10 shadow-[0_0_0_9999px_rgba(15,23,42,0.25)]"
              style={selectionStyle}
            >
              <div className="absolute left-0 top-0 bg-[#FF6B35] px-2 py-1 text-xs font-semibold text-white">
                {draftSelection ? '拖拽中' : '目标球员'}
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <Chip color={value ? 'success' : 'default'} variant="soft" className="text-sm">
        {value
          ? `已选中人物区域：${value.width} x ${value.height} 像素，起始时间 ${formatDuration(value.selectionTime)}`
          : '尚未选择人物区域'}
      </Chip>
    </div>
  );
};

export default PlayerSelector;
