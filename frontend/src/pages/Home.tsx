import React from 'react';
import {
  ImageIcon,
  VideoIcon,
  ArrowRight,
  Upload,
  Cpu,
  BarChart3,
  ScanEye,
  Sparkles,
} from 'lucide-react';

interface HomeProps {
  onSelectMode: (mode: 'image' | 'video') => void;
  onNavigateToAbout: () => void;
}

export const Home: React.FC<HomeProps> = ({ onSelectMode, onNavigateToAbout }) => {
  return (
    <div className="space-y-24 py-8 md:py-16">
      {/* Hero Section */}
      <section className="text-center max-w-4xl mx-auto px-4 sm:px-6">
        {/* Subtle pill badge */}
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-100 border border-neutral-200/80 text-xs text-charcoal-700 font-medium mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-indigo-600 animate-pulse" />
          <span>Research-grade Media Verification Engine</span>
        </div>

        {/* Headline */}
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight text-charcoal-900 leading-[1.12] mb-6">
          See Beyond What's Real.
        </h1>

        {/* Supporting description */}
        <p className="text-base sm:text-lg text-charcoal-600 max-w-2xl mx-auto leading-relaxed mb-10 font-normal">
          DeepTrace applies dual EfficientNet-B0 architectures and Haar Cascade face localization to detect deepfake synthesis, face-swaps, and subtle visual artifacts across images and high-definition video.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 max-w-md mx-auto">
          <button
            onClick={() => onSelectMode('image')}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-charcoal-900 hover:bg-indigo-600 text-white font-medium text-sm shadow-subtle hover:shadow-elevated transition-all group"
          >
            <ImageIcon className="w-4 h-4 text-neutral-300 group-hover:text-white transition-colors" />
            <span>Analyze Image</span>
            <ArrowRight className="w-4 h-4 text-neutral-400 group-hover:translate-x-0.5 group-hover:text-white transition-all" />
          </button>

          <button
            onClick={() => onSelectMode('video')}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-white hover:bg-neutral-50 text-charcoal-900 font-medium text-sm border border-neutral-300 shadow-subtle hover:border-charcoal-400 transition-all group"
          >
            <VideoIcon className="w-4 h-4 text-indigo-600" />
            <span>Analyze Video</span>
            <ArrowRight className="w-4 h-4 text-charcoal-400 group-hover:translate-x-0.5 group-hover:text-charcoal-800 transition-all" />
          </button>
        </div>

        {/* Discretion note */}
        <p className="text-[11px] text-charcoal-500 mt-4 font-mono">
          Model predictions are statistical likelihoods derived from convolutional feature analysis.
        </p>
      </section>

      {/* Clean Visual Preview Section */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6">
        <div className="bg-white rounded-3xl border border-neutral-200/90 p-6 md:p-8 shadow-elevated">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-neutral-100">
            <div>
              <span className="text-[11px] font-mono uppercase tracking-wider text-indigo-600 font-semibold">
                Inference Interface Mockup
              </span>
              <h3 className="text-lg font-bold text-charcoal-900 mt-0.5">
                Authenticity Spectrum & Facial ROI Extraction
              </h3>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-charcoal-500 bg-neutral-50 px-3 py-1.5 rounded-lg border border-neutral-200">
              <ScanEye className="w-3.5 h-3.5 text-indigo-600" />
              <span>Input Tensor: 224×224 RGB</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-12 gap-6 mt-6 items-center">
            {/* Left Mockup: Simulated Media Frame */}
            <div className="md:col-span-5 bg-neutral-900 rounded-2xl p-4 text-white relative overflow-hidden aspect-[4/3] flex flex-col justify-between">
              {/* Simulated Face Bounding Box */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div className="w-36 h-36 rounded-2xl border-2 border-dashed border-indigo-400/80 bg-indigo-500/10 flex items-center justify-center">
                  <span className="text-[10px] font-mono text-indigo-200 bg-indigo-950/80 px-2 py-0.5 rounded border border-indigo-500/40">
                    Haar Cascade ROI (+20% pad)
                  </span>
                </div>
              </div>

              <div className="flex justify-between items-center z-10">
                <span className="text-[10px] font-mono bg-white/20 backdrop-blur-md px-2 py-0.5 rounded text-white">
                  SAMPLE_FRAME_02.JPG
                </span>
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
              </div>

              <div className="z-10 flex justify-between items-end text-[11px] font-mono text-neutral-300">
                <span>Timestamp: 01.28s</span>
                <span>Crop: 224×224</span>
              </div>
            </div>

            {/* Right Mockup: Metric Inspection */}
            <div className="md:col-span-7 space-y-4">
              <div className="p-4 bg-neutral-50 rounded-xl border border-neutral-200/80">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="font-semibold text-charcoal-900">EfficientNet-B0 Output Probabilities</span>
                  <span className="font-mono text-xs text-rose-600 font-semibold">91.2% Fake</span>
                </div>
                {/* Visual bar */}
                <div className="h-2.5 w-full bg-neutral-200 rounded-full overflow-hidden flex">
                  <div className="w-[91.2%] bg-rose-500" />
                  <div className="w-[8.8%] bg-emerald-500" />
                </div>
                <div className="flex justify-between text-[11px] font-mono text-charcoal-500 mt-1.5">
                  <span className="text-rose-600">Fake Score: 0.9123</span>
                  <span className="text-emerald-700">Real Score: 0.0877</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-200/70">
                  <span className="text-charcoal-500 text-[11px] block">Temporal Sampling</span>
                  <span className="font-semibold text-charcoal-900 mt-0.5 block font-mono">
                    4-Point Distribution
                  </span>
                </div>
                <div className="p-3 bg-neutral-50 rounded-xl border border-neutral-200/70">
                  <span className="text-charcoal-500 text-[11px] block">Aggregation</span>
                  <span className="font-semibold text-charcoal-900 mt-0.5 block font-mono">
                    Arithmetic Mean
                  </span>
                </div>
              </div>

              <div className="p-3 bg-indigo-50/50 rounded-xl border border-indigo-100 text-xs text-indigo-900 flex items-start gap-2">
                <Sparkles className="w-4 h-4 text-indigo-600 flex-shrink-0 mt-0.5" />
                <p className="leading-relaxed text-[11px]">
                  Directly interconnected to your live FastAPI backend service running at <span className="font-mono font-semibold">http://127.0.0.1:8000</span>.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3-Step Detection Process */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-xl mx-auto mb-12">
          <h2 className="text-2xl font-bold tracking-tight text-charcoal-900">
            How DeepTrace Analyzes Authenticity
          </h2>
          <p className="text-xs text-charcoal-500 mt-2">
            A systematic three-phase pipeline from uploaded file to neural inference.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Step 1 */}
          <div className="bg-white rounded-2xl border border-neutral-200/90 p-6 shadow-subtle hover:shadow-elevated transition-shadow flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-xl bg-neutral-100 text-charcoal-900 font-mono font-bold text-sm flex items-center justify-center mb-4 border border-neutral-200">
                01
              </div>
              <h3 className="font-semibold text-charcoal-900 text-base mb-2">
                Upload Media
              </h3>
              <p className="text-xs text-charcoal-600 leading-relaxed">
                Provide an image (JPG, PNG, WEBP, BMP up to 10MB) or video (MP4, AVI, MOV, MKV up to 100MB) directly through drag-and-drop or file selection.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-neutral-100 text-[11px] font-mono text-charcoal-500 flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-indigo-600" />
              <span>Client-side validation</span>
            </div>
          </div>

          {/* Step 2 */}
          <div className="bg-white rounded-2xl border border-neutral-200/90 p-6 shadow-subtle hover:shadow-elevated transition-shadow flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-xl bg-indigo-50 text-indigo-600 font-mono font-bold text-sm flex items-center justify-center mb-4 border border-indigo-100">
                02
              </div>
              <h3 className="font-semibold text-charcoal-900 text-base mb-2">
                AI Pattern Analysis
              </h3>
              <p className="text-xs text-charcoal-600 leading-relaxed">
                OpenCV isolates facial regions while PyTorch runs EfficientNet-B0 to detect convolutional anomalies, boundary artifacts, and blending inconsistencies.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-neutral-100 text-[11px] font-mono text-charcoal-500 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-600" />
              <span>Haar Cascade + EfficientNet</span>
            </div>
          </div>

          {/* Step 3 */}
          <div className="bg-white rounded-2xl border border-neutral-200/90 p-6 shadow-subtle hover:shadow-elevated transition-shadow flex flex-col justify-between">
            <div>
              <div className="w-10 h-10 rounded-xl bg-neutral-100 text-charcoal-900 font-mono font-bold text-sm flex items-center justify-center mb-4 border border-neutral-200">
                03
              </div>
              <h3 className="font-semibold text-charcoal-900 text-base mb-2">
                View Detection Result
              </h3>
              <p className="text-xs text-charcoal-600 leading-relaxed">
                Review the REAL or FAKE classification with exact raw softmax scores, visual comparison bars, and timestamped frame breakdowns for video.
              </p>
            </div>
            <div className="mt-6 pt-4 border-t border-neutral-100 text-[11px] font-mono text-charcoal-500 flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5 text-indigo-600" />
              <span>Transparent probability scores</span>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture Deep Dive Banner */}
      <section className="max-w-5xl mx-auto px-4 sm:px-6">
        <div className="bg-neutral-900 text-white rounded-3xl p-8 sm:p-10 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="space-y-2 text-center md:text-left">
            <span className="text-[11px] font-mono uppercase tracking-wider text-indigo-300">
              Architecture & Transparency
            </span>
            <h3 className="text-xl sm:text-2xl font-bold tracking-tight">
              Curious about how the neural network operates?
            </h3>
            <p className="text-xs text-neutral-400 max-w-xl leading-relaxed">
              Explore the detailed technical overview of EfficientNet-B0, frame sampling, Haar Cascade face detection parameters, and the PyTorch inference pipeline.
            </p>
          </div>
          <button
            onClick={onNavigateToAbout}
            className="flex-shrink-0 flex items-center gap-2 px-5 py-3 rounded-xl bg-white hover:bg-neutral-100 text-charcoal-900 font-semibold text-xs transition-colors"
          >
            <span>Read Architecture Docs</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </section>
    </div>
  );
};
