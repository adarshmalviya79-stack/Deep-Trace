import axios, { AxiosError } from 'axios';
import {
  HealthResponse,
  ImagePredictionResponse,
  VideoPredictionResponse,
  ApiError,
} from '../types';

// Read backend URL from environment variable, default to http://127.0.0.1:8000
const ENV_API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// Primary axios instance using the configured VITE_API_URL
export const apiClient = axios.create({
  baseURL: ENV_API_URL,
  timeout: 120000, // 2 minutes to allow video inference
});

// Fallback relative client (routes through Vite dev server proxy if direct CORS fails)
const proxyClient = axios.create({
  baseURL: '',
  timeout: 120000,
});

/**
 * Standardize error messages from Axios and FastAPI
 */
export function formatApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const err = error as AxiosError<{ detail?: string | { msg: string }[] }>;
    const statusCode = err.response?.status;
    let detail = '';

    if (err.response?.data?.detail) {
      if (typeof err.response.data.detail === 'string') {
        detail = err.response.data.detail;
      } else if (Array.isArray(err.response.data.detail)) {
        detail = err.response.data.detail.map((d) => d.msg).join(', ');
      }
    }

    if (!err.response) {
      return {
        message: 'Cannot connect to backend server. Ensure FastAPI is running on ' + ENV_API_URL,
        statusCode: 0,
        detail: err.message,
        isCorsOrNetwork: true,
      };
    }

    switch (statusCode) {
      case 413:
        return {
          message: detail || 'File exceeds maximum upload size limit.',
          statusCode,
          detail,
        };
      case 415:
        return {
          message: detail || 'Unsupported media type. Please upload an accepted file format.',
          statusCode,
          detail,
        };
      case 422:
        return {
          message: detail || 'Unprocessable entity. For video analysis, ensure a human face is clearly visible.',
          statusCode,
          detail,
        };
      case 503:
        return {
          message: detail || 'Detection model checkpoint not loaded on server.',
          statusCode,
          detail,
        };
      case 500:
        return {
          message: detail || 'Internal server error while processing inference.',
          statusCode,
          detail,
        };
      default:
        return {
          message: detail || `Request failed with status ${statusCode}`,
          statusCode,
          detail,
        };
    }
  }

  if (error instanceof Error) {
    return {
      message: error.message,
      detail: error.stack,
    };
  }

  return {
    message: 'An unexpected error occurred during analysis.',
  };
}

/**
 * Health Check: verifies connectivity and loaded model status
 */
export async function checkBackendHealth(): Promise<HealthResponse> {
  try {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  } catch (directErr) {
    // If direct fails due to network/CORS, try dev server proxy
    try {
      const proxyResponse = await proxyClient.get<HealthResponse>('/health');
      return proxyResponse.data;
    } catch {
      throw formatApiError(directErr);
    }
  }
}

/**
 * Image Deepfake Prediction: POST /predict with multipart/form-data
 * Uses exact field name 'file'
 */
export async function predictImage(file: File): Promise<ImagePredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await apiClient.post<ImagePredictionResponse>('/predict', formData);
    return response.data;
  } catch (directErr) {
    // Try fallback proxy if direct hit failed due to browser CORS restriction
    if (axios.isAxiosError(directErr) && !directErr.response) {
      try {
        const proxyResponse = await proxyClient.post<ImagePredictionResponse>('/predict', formData);
        return proxyResponse.data;
      } catch (proxyErr) {
        throw formatApiError(proxyErr);
      }
    }
    throw formatApiError(directErr);
  }
}

/**
 * Video Deepfake Prediction: POST /predict-video with multipart/form-data
 * Uses exact field name 'file'
 */
export async function predictVideo(file: File): Promise<VideoPredictionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await apiClient.post<VideoPredictionResponse>('/predict-video', formData);
    return response.data;
  } catch (directErr) {
    // Try fallback proxy if direct hit failed due to browser CORS restriction
    if (axios.isAxiosError(directErr) && !directErr.response) {
      try {
        const proxyResponse = await proxyClient.post<VideoPredictionResponse>('/predict-video', formData);
        return proxyResponse.data;
      } catch (proxyErr) {
        throw formatApiError(proxyErr);
      }
    }
    throw formatApiError(directErr);
  }
}
