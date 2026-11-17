import React, { useState, useEffect } from 'react';
import { checkBackendHealth } from '../services/api';
import { HealthResponse } from '../types';
import { Activity, CheckCircle2, AlertTriangle, RefreshCw, Cpu } from 'lucide-react';

interface BackendStatusProps {
  compact?: boolean;
}

export const BackendStatus: React.FC<BackendStatusProps> = ({ compact = false }) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await checkBackendHealth();
      setHealth(data);
    } catch (err: unknown) {
      setHealth(null);
      if (typeof err === 'object' && err !== null && 'message' in err) {
        setError((err as { message: string }).message);
      } else {
        setError('FastAPI backend not reachable at http://127.0.0.1:8000');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    // Poll every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const isHealthy = health?.status === 'healthy';

  if (compact) {
    return (
      <div className="relative inline-block text-xs">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          type="button"
          className={`flex items-center gap-2 px-2.5 py-1 rounded-full border transition-all text-xs font-medium ${
            loading
              ? 'bg-amber-50/80 text-amber-800 border-amber-200/80'
              : isHealthy
              ? 'bg-emerald-50/80 text-emerald-800 border-emerald-200/80 hover:bg-emerald-100/70'
              : 'bg-rose-50/80 text-rose-800 border-rose-200/80 hover:bg-rose-100/70'
          }`}
          title="Click to view backend diagnostics"
        >
          <span
            className={`w-2 h-2 rounded-full ${
              loading
                ? 'bg-amber-500 animate-pulse'
                : isHealthy
                ? 'bg-emerald-500 ring-2 ring-emerald-200'
                : 'bg-rose-500'
            }`}
          />
          <span>
            {loading ? 'Checking backend…' : isHealthy ? `Backend: Online (${health.device})` : 'Backend: Offline'}
          </span>
        </button>

        {isExpanded && (
          <div className="absolute right-0 mt-2 w-72 bg-white rounded-xl shadow-lg border border-neutral-200 p-3.5 z-50 text-left">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-100 mb-2.5">
              <span className="font-semibold text-charcoal-900 text-xs flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-indigo-600" />
                Backend Diagnostics
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  fetchHealth();
                }}
                disabled={loading}
                className="text-charcoal-500 hover:text-charcoal-900 transition-colors p-1 rounded"
                title="Refresh Status"
              >
                <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin text-indigo-600' : ''}`} />
              </button>
            </div>

            {isHealthy ? (
              <div className="space-y-2 text-xs text-charcoal-700">
                <div className="flex justify-between items-center py-0.5">
                  <span className="text-charcoal-500">API Endpoint</span>
                  <span className="font-mono text-[11px] text-charcoal-900 bg-neutral-100 px-1.5 py-0.5 rounded">
                    http://127.0.0.1:8000
                  </span>
                </div>
                <div className="flex justify-between items-center py-0.5">
                  <span className="text-charcoal-500 flex items-center gap-1">
                    <Cpu className="w-3 h-3" /> Compute Device
                  </span>
                  <span className="font-semibold capitalize text-charcoal-900">{health.device}</span>
                </div>
                <div className="flex justify-between items-center py-0.5">
                  <span className="text-charcoal-500">Image Model (B0)</span>
                  <span className="flex items-center gap-1">
                    {health.image_model_loaded ? (
                      <span className="text-emerald-700 flex items-center gap-0.5 font-medium">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Loaded
                      </span>
                    ) : (
                      <span className="text-charcoal-500">Lazy loads on 1st request</span>
                    )}
                  </span>
                </div>
                <div className="flex justify-between items-center py-0.5">
                  <span className="text-charcoal-500">Video Model (B0)</span>
                  <span className="flex items-center gap-1">
                    {health.video_model_loaded ? (
                      <span className="text-emerald-700 flex items-center gap-0.5 font-medium">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" /> Loaded
                      </span>
                    ) : (
                      <span className="text-charcoal-500">Lazy loads on 1st request</span>
                    )}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs space-y-2">
                <div className="flex items-start gap-1.5 text-rose-700 bg-rose-50/60 p-2 rounded-lg border border-rose-100">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <p className="leading-snug">
                    Backend service is not responding at <span className="font-mono text-[11px]">http://127.0.0.1:8000</span>
                  </p>
                </div>
                <p className="text-[11px] text-charcoal-500">
                  To start the server, execute in your backend folder:
                </p>
                <div className="font-mono text-[10px] bg-charcoal-900 text-neutral-100 p-2 rounded-md">
                  uvicorn app:app --reload --port 8000
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Full detailed status block (e.g., inside About or Detector info)
  return (
    <div className="bg-white rounded-xl border border-neutral-200 p-4 shadow-subtle">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-600" />
          <h4 className="font-semibold text-charcoal-900 text-sm">Backend Engine Connectivity</h4>
        </div>
        <button
          onClick={fetchHealth}
          disabled={loading}
          className="text-xs text-charcoal-600 hover:text-indigo-600 flex items-center gap-1 font-medium transition-colors"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {isHealthy ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
          <div className="p-2.5 bg-neutral-50 rounded-lg border border-neutral-100">
            <div className="text-charcoal-500 text-[11px]">Status</div>
            <div className="font-semibold text-emerald-600 mt-0.5 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Active
            </div>
          </div>
          <div className="p-2.5 bg-neutral-50 rounded-lg border border-neutral-100">
            <div className="text-charcoal-500 text-[11px]">Compute Device</div>
            <div className="font-semibold text-charcoal-800 mt-0.5 uppercase tracking-wide">
              {health.device}
            </div>
          </div>
          <div className="p-2.5 bg-neutral-50 rounded-lg border border-neutral-100">
            <div className="text-charcoal-500 text-[11px]">Image Inference Model</div>
            <div className="font-medium text-charcoal-800 mt-0.5">
              {health.image_model_loaded ? 'In Memory' : 'On-Demand (Ready)'}
            </div>
          </div>
          <div className="p-2.5 bg-neutral-50 rounded-lg border border-neutral-100">
            <div className="text-charcoal-500 text-[11px]">Video Inference Model</div>
            <div className="font-medium text-charcoal-800 mt-0.5">
              {health.video_model_loaded ? 'In Memory' : 'On-Demand (Ready)'}
            </div>
          </div>
        </div>
      ) : (
        <div className="p-3 bg-rose-50/50 rounded-lg border border-rose-100 text-xs text-rose-800 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">FastAPI Backend Unreachable</div>
            <p className="mt-0.5 text-rose-700/90 text-[11px]">
              Could not connect to <span className="font-mono">http://127.0.0.1:8000/health</span>. {error}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
