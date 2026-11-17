import React, { useState } from 'react';
import { DetectionMode, ImagePredictionResponse, VideoPredictionResponse, ApiError } from '../types';
import { predictImage, predictVideo } from '../services/api';
import { UploadZone } from '../components/UploadZone';
import { ImagePreview } from '../components/ImagePreview';
import { VideoPreview } from '../components/VideoPreview';
import { AnalysisLoader } from '../components/AnalysisLoader';
import { ResultCard } from '../components/ResultCard';
import {
  ImageIcon,
  VideoIcon,
  Play,
  AlertTriangle,
  ShieldCheck,
  ServerCrash,
} from 'lucide-react';

interface DetectorProps {
  initialMode?: DetectionMode;
}

export const Detector: React.FC<DetectorProps> = ({ initialMode = 'image' }) => {
  const [mode, setMode] = useState<DetectionMode>(initialMode);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Analysis status
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [imageResult, setImageResult] = useState<ImagePredictionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoPredictionResponse | null>(null);
  const [apiError, setApiError] = useState<ApiError | null>(null);

  const isImageMode = mode === 'image';

  const handleModeSwitch = (newMode: DetectionMode) => {
    if (newMode === mode) return;
    setMode(newMode);
    setSelectedFile(null);
    setImageResult(null);
    setVideoResult(null);
    setApiError(null);
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setImageResult(null);
    setVideoResult(null);
    setApiError(null);
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setImageResult(null);
    setVideoResult(null);
    setApiError(null);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setImageResult(null);
    setVideoResult(null);
    setApiError(null);
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile || isAnalyzing) return;

    setIsAnalyzing(true);
    setApiError(null);
    setImageResult(null);
    setVideoResult(null);

    try {
      if (isImageMode) {
        const response = await predictImage(selectedFile);
        setImageResult(response);
      } else {
        const response = await predictVideo(selectedFile);
        setVideoResult(response);
      }
    } catch (err: unknown) {
      const errorObj = err as ApiError;
      setApiError(errorObj);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const hasResult = imageResult !== null || videoResult !== null;

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 md:py-16 space-y-8">
      {/* Page Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-charcoal-900">
          Detection Workspace
        </h1>
        <p className="text-sm text-charcoal-500 max-w-lg mx-auto">
          Upload media to evaluate spatial and facial artifacts using convolutional neural network models.
        </p>
      </div>

      {/* Segmented Mode Control Tabs */}
      <div className="flex justify-center">
        <div className="inline-flex p-1 rounded-xl bg-neutral-200/70 border border-neutral-300/80 shadow-inner">
          <button
            onClick={() => handleModeSwitch('image')}
            disabled={isAnalyzing}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold transition-all ${
              isImageMode
                ? 'bg-white text-charcoal-900 shadow-subtle'
                : 'text-charcoal-600 hover:text-charcoal-900'
            } disabled:opacity-50`}
          >
            <ImageIcon className={`w-4 h-4 ${isImageMode ? 'text-indigo-600' : 'text-charcoal-500'}`} />
            <span>Image Detection</span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-100 text-charcoal-500">
              10MB
            </span>
          </button>

          <button
            onClick={() => handleModeSwitch('video')}
            disabled={isAnalyzing}
            className={`flex items-center gap-2 px-5 py-2 rounded-lg text-xs font-semibold transition-all ${
              !isImageMode
                ? 'bg-white text-charcoal-900 shadow-subtle'
                : 'text-charcoal-600 hover:text-charcoal-900'
            } disabled:opacity-50`}
          >
            <VideoIcon className={`w-4 h-4 ${!isImageMode ? 'text-indigo-600' : 'text-charcoal-500'}`} />
            <span>Video Detection</span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-100 text-charcoal-500">
              100MB
            </span>
          </button>
        </div>
      </div>

      {/* Active Workspace Area */}
      <div className="space-y-6">
        {/* If currently analyzing, show loader */}
        {isAnalyzing ? (
          <AnalysisLoader mode={mode} />
        ) : hasResult ? (
          /* Result view */
          <ResultCard
            mode={mode}
            imageResult={imageResult}
            videoResult={videoResult}
            uploadedFile={selectedFile}
            onReset={handleReset}
          />
        ) : (
          /* Staging / Upload view */
          <div className="space-y-6">
            {!selectedFile ? (
              <UploadZone
                mode={mode}
                onFileSelect={handleFileSelect}
                selectedFile={selectedFile}
                disabled={isAnalyzing}
              />
            ) : isImageMode ? (
              <ImagePreview
                file={selectedFile}
                onRemove={handleRemoveFile}
                onChangeFile={handleRemoveFile}
                disabled={isAnalyzing}
              />
            ) : (
              <VideoPreview
                file={selectedFile}
                onRemove={handleRemoveFile}
                onChangeFile={handleRemoveFile}
                disabled={isAnalyzing}
              />
            )}

            {/* Action Trigger Button */}
            {selectedFile && (
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 bg-white rounded-2xl border border-neutral-200/90 shadow-subtle">
                <div className="text-xs text-charcoal-600 flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-indigo-600" />
                  <span>
                    Ready to send to FastAPI endpoint{' '}
                    <code className="font-mono text-charcoal-900 bg-neutral-100 px-1 py-0.5 rounded">
                      {isImageMode ? '/predict' : '/predict-video'}
                    </code>
                  </span>
                </div>

                <button
                  onClick={handleStartAnalysis}
                  disabled={isAnalyzing}
                  className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-charcoal-900 hover:bg-indigo-600 text-white font-semibold text-xs shadow-subtle transition-all disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isImageMode ? 'Analyze Image' : 'Analyze Video'}</span>
                </button>
              </div>
            )}
          </div>
        )}

        {/* API Error Notification */}
        {apiError && !isAnalyzing && (
          <div className="p-4 bg-rose-50 border border-rose-200 rounded-2xl shadow-subtle flex items-start gap-3 text-xs text-rose-900 animate-fadeIn">
            {apiError.isCorsOrNetwork ? (
              <ServerCrash className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
            )}
            <div className="flex-1 space-y-1">
              <div className="font-semibold text-rose-950 flex items-center gap-2">
                <span>Inference Request Failed</span>
                {apiError.statusCode ? (
                  <span className="font-mono text-[10px] bg-rose-200/60 px-1.5 py-0.5 rounded text-rose-800">
                    HTTP {apiError.statusCode}
                  </span>
                ) : null}
              </div>
              <p className="text-rose-800 leading-relaxed">{apiError.message}</p>
              {apiError.detail && (
                <p className="text-rose-700 font-mono text-[11px] bg-white/70 p-2 rounded-lg border border-rose-200 mt-2 break-all">
                  {apiError.detail}
                </p>
              )}
              {apiError.statusCode === 422 && !isImageMode && (
                <p className="text-rose-800 text-[11px] mt-1 italic">
                  Tip: Ensure the video has an unobstructed face with sufficient lighting across the sampled frames.
                </p>
              )}
            </div>
            <button
              onClick={() => setApiError(null)}
              className="text-rose-500 hover:text-rose-700 font-medium text-xs ml-2"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
