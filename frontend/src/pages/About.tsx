import React from 'react';
import { BackendStatus } from '../components/BackendStatus';
import {
  ArrowRight,
  Code2,
  AlertCircle,
  Activity,
} from 'lucide-react';

export const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 py-10 md:py-16 space-y-12">
      {/* Header */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-100 border border-neutral-200 text-xs font-mono text-charcoal-700">
          <Code2 className="w-3.5 h-3.5 text-indigo-600" />
          <span>System Architecture & Scientific Methodology</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-charcoal-900">
          About DeepTrace
        </h1>
        <p className="text-sm text-charcoal-600 max-w-2xl leading-relaxed">
          DeepTrace combines convolutional feature extraction with computer vision face localization to identify synthesis artifacts and manipulations across digital media.
        </p>
      </div>

      {/* Backend Engine Diagnostics */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-800 flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-600" />
          Live Inference Engine Status
        </h3>
        <BackendStatus />
      </div>

      {/* Actual Technology Stack */}
      <div className="space-y-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-800">
          Core Technologies
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">EfficientNet-B0</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              Convolutional neural network featuring compound scaling that balances depth, width, and resolution for artifact detection.
            </p>
          </div>

          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">PyTorch & Torchvision</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              Powers neural tensor computations, checkpoint loading (<code className="font-mono text-[11px]">.pth</code>), and GPU CUDA acceleration.
            </p>
          </div>

          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">OpenCV</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              Extracts raw video frames, computes FPS metadata, and performs color space transformations (<code className="font-mono text-[11px]">BGR to RGB</code>).
            </p>
          </div>

          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">Haar Cascade Classifiers</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              Locates frontal faces within sampled video frames, outputting coordinates with 20% proportional padding to preserve facial borders.
            </p>
          </div>

          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">FastAPI Backend</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              High-performance asynchronous Python API exposing dedicated endpoints for image and video analysis with multipart stream parsing.
            </p>
          </div>

          <div className="p-4 bg-white rounded-xl border border-neutral-200/90 shadow-subtle">
            <div className="font-semibold text-charcoal-900 text-sm mb-1">React & TypeScript</div>
            <p className="text-xs text-charcoal-600 leading-relaxed">
              Strictly typed client with asynchronous Axios communication, drag-and-drop validation, and real-time score visualization.
            </p>
          </div>
        </div>
      </div>

      {/* Detection Workflow Pipelines */}
      <div className="space-y-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-charcoal-800">
          Inference Pipelines
        </h3>

        {/* Image Detection Workflow */}
        <div className="bg-white rounded-2xl border border-neutral-200/90 p-6 shadow-subtle space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" />
            <h4 className="font-semibold text-charcoal-900 text-sm">
              Image Detection Pipeline
            </h4>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              Image Upload
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              Preprocessing (224×224, ImageNet Norm)
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              EfficientNet-B0 Inference
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-lg border border-indigo-200 font-semibold">
              Softmax Prediction (Real vs Fake)
            </span>
          </div>

          <p className="text-xs text-charcoal-600 leading-relaxed pt-2 border-t border-neutral-100">
            Incoming images are verified for format (JPG, PNG, WEBP, BMP) and file size (≤ 10 MB), converted to RGB tensors, resized to 224×224, and normalized using standard ImageNet mean and standard deviation parameters before feedforward pass.
          </p>
        </div>

        {/* Video Detection Workflow */}
        <div className="bg-white rounded-2xl border border-neutral-200/90 p-6 shadow-subtle space-y-4">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" />
            <h4 className="font-semibold text-charcoal-900 text-sm">
              Video Detection Pipeline
            </h4>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              Video Upload (≤ 100 MB)
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              Frame Sampling (20%, 40%, 60%, 80%)
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              Haar Cascade Face Detection
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              +20% Padded Crop & Preprocess
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-neutral-100 text-charcoal-800 px-3 py-1.5 rounded-lg border border-neutral-200 font-medium">
              EfficientNet-B0 Video Model
            </span>
            <ArrowRight className="w-3.5 h-3.5 text-charcoal-400" />
            <span className="bg-indigo-50 text-indigo-700 px-3 py-1.5 rounded-lg border border-indigo-200 font-semibold">
              Average Score & Verdict
            </span>
          </div>

          <p className="text-xs text-charcoal-600 leading-relaxed pt-2 border-t border-neutral-100">
            OpenCV extracts frames at four proportional points across the video duration. A Haar Cascade classifier detects face regions with a minimum size of 60×60 pixels and applies 20% margin padding. Cropped faces are stacked in a single batch for tensorized evaluation, producing individual frame scores and an aggregated mean score.
          </p>
        </div>
      </div>

      {/* Scientific Limitation & Disclaimer */}
      <div className="p-6 bg-neutral-100/70 rounded-2xl border border-neutral-200 space-y-2">
        <div className="flex items-center gap-2 text-charcoal-900 font-semibold text-sm">
          <AlertCircle className="w-4 h-4 text-indigo-600" />
          <span>Research Limitation & Authenticity Statement</span>
        </div>
        <p className="text-xs text-charcoal-600 leading-relaxed">
          DeepTrace predictions are probabilistic estimates produced by machine learning classifiers trained on specific deepfake datasets. While effective at identifying characteristic synthesis patterns and warping boundaries, model predictions do not provide infallible legal proof of authenticity. Output scores should be interpreted as diagnostic indicators alongside other forensic methodologies.
        </p>
      </div>
    </div>
  );
};
