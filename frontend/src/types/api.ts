// API 响应类型定义
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface PlayerSelectionBox {
  x: number;
  y: number;
  width: number;
  height: number;
  frameWidth: number;
  frameHeight: number;
  selectionTime: number;
  selectionFrame?: number;
}

export interface SelectionFrame {
  imageUrl: string;
  width: number;
  height: number;
  time: number;
  frame?: number;
}

export interface TrackingSummary {
  enabled: boolean;
  trackerType?: string;
  activeFrames: number;
  totalFrames: number;
  coverage: number;
  missingFrames: number;
  lostFrames?: number;
  reacquiredCount?: number;
  guardedSwitches?: number;
  latestStatus?: string;
  startFrame?: number;
  startTime?: number;
  error?: string;
}

export interface ShotTimestamp {
  frame: number;
  timestamp: number;
  made: boolean;
  owner?: 'target' | 'unknown';
  owner_confidence?: number;
  target_visible?: boolean;
  highlight_role?: 'score' | 'assist' | 'none';
  highlight_confidence?: number;
}

// 视频上传响应 - 匹配Flask后端格式
export interface UploadResponse {
  success: boolean;
  fileId: string;
  filename: string;
  fileSize: number;
  message: string;
}

// 任务状态类型
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed';
export type BackendTaskStatus =
  | 'pending'
  | 'starting'
  | 'detecting'
  | 'generating'
  | 'completed'
  | 'failed';

// 处理阶段类型
export type ProcessingStage = 
  | 'uploading'
  | 'analyzing'
  | 'detecting'
  | 'generating'
  | 'finalizing'
  | 'completed';

// 进度信息 - 匹配Flask后端格式
export interface ProgressInfo {
  progress: number;
  stage: string;
  status: BackendTaskStatus;
  completed: boolean;
  result?: ProcessingResult;
  error?: string;
}

// 检测结果统计
export interface DetectionStats {
  total_shots: number;
  successful_shots: number;
  missed_shots: number;
  accuracy_rate: number;
  total_duration: number;
  highlight_duration: number;
  shots_made?: number;
  dunks_made?: number;
  assists_made?: number;
  steals_made?: number;
  target_attempts?: number;
  target_makes?: number;
  target_scores?: number;
  target_assists?: number;
  target_highlights?: number;
}

// 高光片段类型
export type HighlightType = 'shot' | 'dunk' | 'pass' | 'defense';

// 高光片段
export interface Highlight {
  id: string;
  type: HighlightType;
  start_time: number;
  end_time: number;
  confidence: number;
  description?: string;
}

// 输出文件信息
export interface OutputFile {
  filename: string;
  size: number;
  duration: number;
  format: string;
  url?: string;
}

// 视频处理结果
export interface ProcessingResult {
  task_id?: string;
  status?: TaskStatus;
  output_file?: OutputFile;
  file_size?: number;
  stats?: DetectionStats;
  highlights?: Highlight[];
  processing_time?: number;
  created_at?: string;
  completed_at?: string;
  error_message?: string;
  totalShots?: number;
  madeShots?: number;
  targetShots?: number;
  targetScores?: number;
  targetAssists?: number;
  targetHighlights?: number;
  accuracy?: number;
  highlightVideo?: string | null;
  annotatedVideo?: string | null;
  timestamps?: ShotTimestamp[];
  allMadeTimestamps?: ShotTimestamp[];
  fileSize?: number;
  targetPlayerBox?: PlayerSelectionBox | null;
  tracking?: TrackingSummary;
  message?: string;
}

// 健康检查响应
export interface HealthCheckResponse {
  status: string;
  timestamp: string;
  components: {
    ai_model: boolean;
    storage: boolean;
    processing: boolean;
  };
  version: string;
}

// 视频上传参数
export interface UploadParams {
  file: File;
}

// 视频处理参数 - 匹配Flask后端格式
export interface ProcessParams {
  fileId: string;
  beforeSeconds?: number;
  afterSeconds?: number;
  targetPlayerBox?: PlayerSelectionBox | null;
}

// 视频处理响应
export interface ProcessResponse {
  success: boolean;
  taskId: string;
  message?: string;
}

// 任务查询参数
export interface TaskQueryParams {
  taskId: string;
}

// 下载参数
export interface DownloadParams {
  filename: string;
}
