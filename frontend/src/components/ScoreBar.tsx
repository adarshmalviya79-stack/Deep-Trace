import React from 'react';

interface ScoreBarProps {
  fakeScore: number; // 0 to 1
  realScore: number; // 0 to 1
  compact?: boolean;
  showLabels?: boolean;
}

export const ScoreBar: React.FC<ScoreBarProps> = ({
  fakeScore,
  realScore,
  compact = false,
  showLabels = true,
}) => {
  const fakePct = Math.min(100, Math.max(0, fakeScore * 100));
  const realPct = Math.min(100, Math.max(0, realScore * 100));

  if (compact) {
    return (
      <div className="w-full">
        {showLabels && (
          <div className="flex justify-between items-center text-[11px] mb-1 font-mono">
            <span className="text-rose-600 font-medium">Fake: {fakePct.toFixed(1)}%</span>
            <span className="text-emerald-700 font-medium">Real: {realPct.toFixed(1)}%</span>
          </div>
        )}
        <div className="h-1.5 w-full bg-neutral-200 rounded-full overflow-hidden flex">
          <div
            style={{ width: `${fakePct}%` }}
            className="bg-rose-500 transition-all duration-500"
            title={`Fake: ${fakePct.toFixed(2)}%`}
          />
          <div
            style={{ width: `${realPct}%` }}
            className="bg-emerald-500 transition-all duration-500"
            title={`Real: ${realPct.toFixed(2)}%`}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Fake Bar */}
      <div>
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-xs font-medium text-charcoal-700 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-rose-500" />
            Fake Probability
          </span>
          <span className="text-sm font-semibold font-mono text-charcoal-900">
            {fakePct.toFixed(2)}%
            <span className="text-xs text-charcoal-500 font-normal ml-1">({fakeScore.toFixed(4)})</span>
          </span>
        </div>
        <div className="h-2 w-full bg-neutral-100 rounded-full overflow-hidden border border-neutral-200/60 p-[1px]">
          <div
            style={{ width: `${fakePct}%` }}
            className="h-full bg-rose-500 rounded-full transition-all duration-700 ease-out"
          />
        </div>
      </div>

      {/* Real Bar */}
      <div>
        <div className="flex justify-between items-baseline mb-1">
          <span className="text-xs font-medium text-charcoal-700 flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Real Authenticity
          </span>
          <span className="text-sm font-semibold font-mono text-charcoal-900">
            {realPct.toFixed(2)}%
            <span className="text-xs text-charcoal-500 font-normal ml-1">({realScore.toFixed(4)})</span>
          </span>
        </div>
        <div className="h-2 w-full bg-neutral-100 rounded-full overflow-hidden border border-neutral-200/60 p-[1px]">
          <div
            style={{ width: `${realPct}%` }}
            className="h-full bg-emerald-500 rounded-full transition-all duration-700 ease-out"
          />
        </div>
      </div>
    </div>
  );
};
