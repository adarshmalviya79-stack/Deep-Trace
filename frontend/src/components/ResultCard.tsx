import React from 'react';
import {
  ImagePredictionResponse,
  VideoPredictionResponse,
  DetectionMode,
} from '../types';
import { ScoreBar } from './ScoreBar';
import { FrameResults } from './FrameResults';
import {
  ShieldAlert,
  ShieldCheck,
  Info,
  Clock,
  Layers,
  FileCheck,
  RotateCcw,
} from 'lucide-react';

interface ResultCardProps {
  mode: DetectionMode;
  imageResult: ImagePredictionResponse | null;
  videoResult: VideoPredictionResponse | null;
  uploadedFile: File | null;
  onReset: () => void;
}

export const ResultCard: React.FC<ResultCardProps> = ({
  mode,
  imageResult,
  videoResult,
  uploadedFile,
  onReset,
}) => {
  const isVideo = mode === 'video';
  const prediction = isVideo ? videoResult?.prediction : imageResult?.prediction;
  const isFake = prediction?.toLowerCase() === 'fake';

  const fakeScore = isVideo
    ? videoResult?.average_scores.fake ?? 0
    : imageResult?.scores.fake ?? 0;

  const realScore = isVideo
    ? videoResult?.average_scores.real ?? 0
    : imageResult?.scores.real ?? 0;

  const notice = isVideo
    ? videoResult?.notice
    : imageResult?.notice;

  return (
    <div className="bg-white rounded-2xl border border-neutral-200/90 shadow-elevated overflow-hidden animate-fadeIn">
      {/* Top Banner indicating Status */}
      <div
        className={`px-6 py-5 border-b flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
          isFake
            ? 'bg-rose-50/70 border-rose-200/70 text-rose-950'
            : 'bg-emerald-50/70 border-emerald-200/70 text-emerald-950'
        }`}
      >
        <div className="flex items-center gap-3.5">
          <div
            className={`w-12 h-12 rounded-xl flex items-center justify-center shadow-subtle ${
              isFake
                ? 'bg-rose-600 text-white'
                : 'bg-emerald-600 text-white'
            }`}
          >
            {isFake ? (
              <ShieldAlert className="w-6 h-6" />
            ) : (
              <ShieldCheck className="w-6 h-6" />
            )}
          </div>
          <div>
            <div className="text-[11px] font-semibold tracking-wider uppercase opacity-75">
              Classification Verdict
            </div>
            <div className="text-2xl font-bold tracking-tight flex items-center gap-2">
              <span>{isFake ? 'MANIPULATED / FAKE' : 'AUTHENTIC / REAL'}</span>
            </div>
          </div>
        </div>

        <button
          onClick={onReset}
          className="self-start sm:self-center flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-charcoal-700 bg-white hover:bg-neutral-100 border border-neutral-300 shadow-subtle transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5 text-charcoal-500" />
          Analyze Another File
        </button>
      </div>

      <div className="p-6 md:p-8 space-y-6">
        {/* Summary Meta Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
            <span className="text-charcoal-500 text-[11px] block">Analyzed File</span>
            <span
              className="font-medium text-charcoal-900 mt-1 block truncate font-mono text-[11px]"
              title={uploadedFile?.name || imageResult?.filename || videoResult?.filename}
            >
              {uploadedFile?.name || imageResult?.filename || videoResult?.filename || 'Media'}
            </span>
          </div>

          <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
            <span className="text-charcoal-500 text-[11px] block">Target Architecture</span>
            <span className="font-medium text-charcoal-900 mt-1 block">
              EfficientNet-B0
            </span>
          </div>

          {isVideo && videoResult ? (
            <>
              <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
                <span className="text-charcoal-500 text-[11px] block flex items-center gap-1">
                  <Layers className="w-3 h-3 text-charcoal-400" /> Frames Evaluated
                </span>
                <span className="font-semibold text-charcoal-900 mt-1 block font-mono">
                  {videoResult.frames_analyzed} face crops
                </span>
              </div>

              <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
                <span className="text-charcoal-500 text-[11px] block flex items-center gap-1">
                  <Clock className="w-3 h-3 text-charcoal-400" /> Video Duration
                </span>
                <span className="font-semibold text-charcoal-900 mt-1 block font-mono">
                  {videoResult.video_duration_seconds.toFixed(2)}s
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
                <span className="text-charcoal-500 text-[11px] block">Inference Method</span>
                <span className="font-medium text-charcoal-900 mt-1 block">
                  Spatial CNN (224x224)
                </span>
              </div>

              <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-100">
                <span className="text-charcoal-500 text-[11px] block">Confidence Margin</span>
                <span className="font-semibold text-charcoal-900 mt-1 block font-mono">
                  {Math.abs(fakeScore - realScore * 1).toFixed(4)}
                </span>
              </div>
            </>
          )}
        </div>

        {/* Detailed Probability Scores Section */}
        <div className="p-5 bg-neutral-50/80 rounded-2xl border border-neutral-200/80">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-xs font-semibold uppercase tracking-wider text-charcoal-800 flex items-center gap-1.5">
              <FileCheck className="w-4 h-4 text-indigo-600" />
              {isVideo ? 'Average Video Probabilities' : 'Classification Probabilities'}
            </h4>
            <span className="text-[11px] font-mono text-charcoal-500">
              Softmax Distribution
            </span>
          </div>

          <ScoreBar fakeScore={fakeScore} realScore={realScore} />
        </div>

        {/* Video Frame-by-Frame Results */}
        {isVideo && videoResult?.frame_results && (
          <div className="pt-2">
            <FrameResults frames={videoResult.frame_results} />
          </div>
        )}

        {/* Formal Disclaimer / Notice */}
        <div className="p-4 bg-amber-50/60 rounded-xl border border-amber-200/80 flex items-start gap-3 text-xs text-amber-900">
          <Info className="w-4 h-4 text-amber-700 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-semibold text-amber-950">Scientific Disclaimer</p>
            <p className="text-amber-900/90 leading-relaxed text-[11px]">
              {notice ||
                'Model estimates are statistical approximations based on training data distributions and do not constitute definitive proof of digital manipulation or biological authenticity.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
