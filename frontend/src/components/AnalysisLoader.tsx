import React, { useEffect, useState } from 'react';
import { DetectionMode } from '../types';
import { Loader2, Scan, Clock } from 'lucide-react';

interface AnalysisLoaderProps {
  mode: DetectionMode;
}

export const AnalysisLoader: React.FC<AnalysisLoaderProps> = ({ mode }) => {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const isVideo = mode === 'video';

  return (
    <div className="bg-white rounded-2xl border border-neutral-200/90 p-8 sm:p-12 shadow-elevated text-center max-w-lg mx-auto">
      {/* Scanner Visual */}
      <div className="relative w-20 h-20 mx-auto mb-6 flex items-center justify-center">
        {/* Subtle pulsating outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-indigo-200 animate-ping opacity-25" />
        <div className="absolute inset-1 rounded-full border border-indigo-300 animate-pulse" />
        
        {/* Center icon */}
        <div className="w-14 h-14 rounded-full bg-indigo-50 border border-indigo-100 flex items-center justify-center text-indigo-600 shadow-inner">
          <Scan className="w-7 h-7 animate-pulse text-indigo-600" />
        </div>
      </div>

      <div className="flex items-center justify-center gap-2 mb-2">
        <Loader2 className="w-4 h-4 text-indigo-600 animate-spin" />
        <h3 className="text-lg font-semibold text-charcoal-900 tracking-tight">
          Analyzing media...
        </h3>
      </div>

      <p className="text-xs text-charcoal-600 max-w-sm mx-auto leading-relaxed mb-6">
        {isVideo
          ? 'Decoding video stream, extracting 4 sampled frames, detecting facial regions via Haar Cascade, and evaluating EfficientNet-B0 checkpoints.'
          : 'Normalizing pixel tensors and computing softmax activation via PyTorch EfficientNet-B0.'}
      </p>

      {/* Subtle Indeterminate Activity Bar */}
      <div className="w-full bg-neutral-100 h-1.5 rounded-full overflow-hidden mb-4 relative">
        <div className="absolute top-0 bottom-0 left-0 bg-indigo-600 rounded-full animate-indeterminate w-1/3" />
      </div>

      <div className="flex items-center justify-between text-[11px] text-charcoal-500 font-mono pt-2 border-t border-neutral-100">
        <span className="flex items-center gap-1">
          <Clock className="w-3 h-3 text-charcoal-400" />
          Elapsed: {elapsedSeconds}s
        </span>
        <span>
          {isVideo ? 'Video analysis takes ~10–30s' : 'Image inference usually < 3s'}
        </span>
      </div>

      <style>{`
        @keyframes indeterminate {
          0% {
            transform: translateX(-100%);
          }
          100% {
            transform: translateX(350%);
          }
        }
        .animate-indeterminate {
          animation: indeterminate 1.8s infinite ease-in-out;
        }
      `}</style>
    </div>
  );
};
