/**
 * 验证工具函数
 */
import { FILE_LIMITS } from './constants';
import type { PlayerSelectionBox, SelectionFrame } from '@/types';

// 支持的视频格式
export const SUPPORTED_VIDEO_FORMATS: readonly string[] = [
  ...FILE_LIMITS.SUPPORTED_MIME_TYPES,
];

// 支持的视频扩展名
export const SUPPORTED_VIDEO_EXTENSIONS: readonly string[] = [
  ...FILE_LIMITS.SUPPORTED_FORMATS,
];

// 最大文件大小（2GB，与本地后端限制一致）
export const MAX_FILE_SIZE = FILE_LIMITS.MAX_SIZE;

// 最小文件大小（1MB）
export const MIN_FILE_SIZE = FILE_LIMITS.MIN_SIZE;

// 验证视频文件
export const validateVideoFile = (file: File): { valid: boolean; error?: string } => {
  // 检查文件是否存在
  if (!file) {
    return { valid: false, error: '请选择一个文件' };
  }
  
  // 检查文件大小
  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, error: `文件大小不能超过 ${Math.round(MAX_FILE_SIZE / 1024 / 1024)}MB` };
  }
  
  if (file.size < MIN_FILE_SIZE) {
    return { valid: false, error: `文件大小不能小于 ${Math.round(MIN_FILE_SIZE / 1024 / 1024)}MB` };
  }
  
  // 检查文件类型
  const isValidType = SUPPORTED_VIDEO_FORMATS.includes(file.type);
  const fileExtension = file.name.split('.').pop()?.toLowerCase();
  const isValidExtension = fileExtension && SUPPORTED_VIDEO_EXTENSIONS.includes(fileExtension);
  
  if (!isValidType && !isValidExtension) {
    return { 
      valid: false, 
      error: `不支持的文件格式。支持的格式：${SUPPORTED_VIDEO_EXTENSIONS.join(', ')}` 
    };
  }
  
  return { valid: true };
};

// 验证处理配置
export const validateProcessingConfig = (config: {
  beforeSeconds: number;
  afterSeconds: number;
}): { valid: boolean; error?: string } => {
  // 验证进球前保留时间
  if (config.beforeSeconds < 1 || config.beforeSeconds > 15) {
    return { valid: false, error: '进球前保留时间必须在 1 到 15 秒之间' };
  }
  
  // 验证进球后保留时间
  if (config.afterSeconds < 1 || config.afterSeconds > 10) {
    return { valid: false, error: '进球后保留时间必须在 1 到 10 秒之间' };
  }
  
  return { valid: true };
};

export const validateTargetPlayerBox = (
  selection: PlayerSelectionBox | null | undefined
): { valid: boolean; error?: string } => {
  if (!selection) {
    return { valid: false, error: '请先确认目标球员清晰出镜的画面，再框选目标球员' };
  }

  const numericFields = [
    selection.x,
    selection.y,
    selection.width,
    selection.height,
    selection.frameWidth,
    selection.frameHeight,
    selection.selectionTime,
  ];

  if (numericFields.some((value) => !Number.isFinite(value))) {
    return { valid: false, error: '人物选区数据无效，请重新框选' };
  }

  if (selection.width < 20 || selection.height < 20) {
    return { valid: false, error: '人物选区过小，请重新框选完整身体区域' };
  }

  if (selection.frameWidth <= 0 || selection.frameHeight <= 0) {
    return { valid: false, error: '框选画面尺寸无效，请重新选择视频' };
  }

  if (selection.selectionTime < 0) {
    return { valid: false, error: '框选时间点无效，请重新选择你出镜的画面' };
  }

  if (
    selection.x < 0 ||
    selection.y < 0 ||
    selection.x + selection.width > selection.frameWidth ||
    selection.y + selection.height > selection.frameHeight
  ) {
    return { valid: false, error: '人物选区超出当前画面范围，请重新框选' };
  }

  return { valid: true };
};

export const syncSelectionBoxToFrame = (
  selection: PlayerSelectionBox | null | undefined,
  frame: SelectionFrame | null | undefined
): PlayerSelectionBox | null => {
  if (!selection) {
    return null;
  }

  if (!frame) {
    return { ...selection };
  }

  return {
    ...selection,
    frameWidth: frame.width,
    frameHeight: frame.height,
    selectionTime: frame.time,
    selectionFrame: frame.frame,
  };
};

export const validateProcessingConfig2 = (
  config: { beforeSeconds?: number; afterSeconds?: number } | null | undefined
): { valid: boolean; error?: string } => {
  if (!config) {
    return { valid: false, error: '配置不能为空' };
  }

  return { valid: true };
};

// 验证任务ID
export const validateTaskId = (taskId: string): boolean => {
  return /^[a-zA-Z0-9-_]{8,64}$/.test(taskId);
};

// 验证文件名
export const validateFileName = (fileName: string): boolean => {
  // 检查文件名长度
  if (fileName.length < 1 || fileName.length > 255) {
    return false;
  }
  
  // 检查非法字符
  const invalidChars = /[<>:"/\\|?*]/;
  return !invalidChars.test(fileName);
};

// 验证URL
export const validateUrl = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

// 验证邮箱
export const validateEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// 验证手机号
export const validatePhone = (phone: string): boolean => {
  const phoneRegex = /^1[3-9]\d{9}$/;
  return phoneRegex.test(phone);
};

// 通用必填验证
export const validateRequired = (value: unknown, fieldName: string): { valid: boolean; error?: string } => {
  if (value === null || value === undefined || value === '') {
    return { valid: false, error: `${fieldName}不能为空` };
  }
  return { valid: true };
};

// 数字范围验证
export const validateNumberRange = (
  value: number, 
  min: number, 
  max: number, 
  fieldName: string
): { valid: boolean; error?: string } => {
  if (isNaN(value)) {
    return { valid: false, error: `${fieldName}必须是数字` };
  }
  
  if (value < min || value > max) {
    return { valid: false, error: `${fieldName}必须在 ${min} 到 ${max} 之间` };
  }
  
  return { valid: true };
};
