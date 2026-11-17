export type DetectionMode = 'image' | 'video';

export interface HealthResponse {
  status: string;
  device: string;
  image_model_loaded: boolean;
  video_model_loaded: boolean;
}

export interface ImageScores {
  fake: number;
  real: number;
}

export interface ImagePredictionResponse {
  filename: string;
  prediction: 'fake' | 'real';
  scores: ImageScores;
  notice: string;
}

export interface VideoFrameResult {
  time_seconds: number;
  prediction: 'fake' | 'real';
  fake_score: number;
  real_score: number;
}

export interface VideoAverageScores {
  fake: number;
  real: number;
}

export interface VideoPredictionResponse {
  filename: string;
  prediction: 'fake' | 'real';
  frames_analyzed: number;
  video_duration_seconds: number;
  average_scores: VideoAverageScores;
  frame_results: VideoFrameResult[];
  notice: string;
}

export interface ApiError {
  message: string;
  statusCode?: number;
  detail?: string;
  isCorsOrNetwork?: boolean;
}
