import React from 'react';
import { VideoFrameResult } from '../types';
import { ScoreBar } from './ScoreBar';
import { Clock, ShieldAlert, ShieldCheck } from 'lucide-react';

interface FrameResultsProps {
  frames: VideoFrameResult[];
}

export const FrameResults: React.FC<FrameResultsProps> = ({ frames }) => {
  if (!frames || frames.length === 0) {
    return (
      <div className="p-4 bg-neutral-50 rounded-xl text-center text-xs text-charcoal-500">
        No frame breakdown available for this video.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between pb-1">
        <h4 className="text-xs font-semibold text-charcoal-800 uppercase tracking-wider flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-indigo-600" />
          Sampled Frame Breakdown ({frames.length} frames)
        </h4>
        <span className="text-[11px] text-charcoal-500">
          Sampled at 20%, 40%, 60%, 80% intervals
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {frames.map((frame, index) => {
          const isFake = frame.prediction.toLowerCase() === 'fake';
          return (
            <div
              key={index}
              className={`p-3.5 rounded-xl border transition-all ${
                isFake
                  ? 'bg-rose-50/40 border-rose-200/80 hover:bg-rose-50/70'
                  : 'bg-emerald-50/40 border-emerald-200/80 hover:bg-emerald-50/70'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold bg-white/90 px-2 py-0.5 rounded border border-neutral-200/80 text-charcoal-800">
                    Frame #{index + 1}
                  </span>
                  <span className="text-[11px] text-charcoal-500 font-mono">
                    @{frame.time_seconds.toFixed(2)}s
                  </span>
                </div>

                <span
                  className={`inline-flex items-center gap-1 text-[11px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full border ${
                    isFake
                      ? 'bg-rose-100 text-rose-800 border-rose-300'
                      : 'bg-emerald-100 text-emerald-800 border-emerald-300'
                  }`}
                >
                  {isFake ? (
                    <>
                      <ShieldAlert className="w-3 h-3 text-rose-600" />
                      FAKE
                    </>
                  ) : (
                    <>
                      <ShieldCheck className="w-3 h-3 text-emerald-600" />
                      REAL
                    </>
                  )}
                </span>
              </div>

              {/* Score bar */}
              <ScoreBar
                fakeScore={frame.fake_score}
                realScore={frame.real_score}
                compact
              />
            </div>
          );
        })}
      </div>
    </div>
  );
};
