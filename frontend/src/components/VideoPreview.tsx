import React, { useEffect, useState } from 'react';
import { X, RefreshCw, Film, CheckCircle2 } from 'lucide-react';
import { formatBytes } from './UploadZone';

interface VideoPreviewProps {
  file: File;
  onRemove: () => void;
  onChangeFile: () => void;
  disabled?: boolean;
}

export const VideoPreview: React.FC<VideoPreviewProps> = ({
  file,
  onRemove,
  onChangeFile,
  disabled = false,
}) => {
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [duration, setDuration] = useState<number | null>(null);

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setVideoUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  const handleLoadedMetadata = (e: React.SyntheticEvent<HTMLVideoElement, Event>) => {
    const video = e.currentTarget;
    if (video.duration && !isNaN(video.duration)) {
      setDuration(video.duration);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-neutral-200 p-4 shadow-subtle">
      <div className="flex flex-col sm:flex-row items-start gap-4">
        {/* Video Player */}
        <div className="relative w-full sm:w-56 bg-charcoal-900 rounded-xl overflow-hidden border border-neutral-300 flex-shrink-0 aspect-video flex items-center justify-center">
          {videoUrl ? (
            <video
              src={videoUrl}
              controls
              onLoadedMetadata={handleLoadedMetadata}
              className="w-full h-full object-contain"
            />
          ) : (
            <div className="text-xs text-charcoal-400">Loading video…</div>
          )}
          <span className="absolute top-1.5 left-1.5 bg-black/70 backdrop-blur-sm text-white font-mono text-[10px] px-1.5 py-0.5 rounded flex items-center gap-1 pointer-events-none">
            <Film className="w-3 h-3 text-indigo-400" /> VIDEO
          </span>
        </div>

        {/* Details & Actions */}
        <div className="flex-1 w-full flex flex-col justify-between py-1">
          <div>
            <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-medium mb-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Video loaded and ready for frame extraction
            </div>
            <h4
              className="font-semibold text-charcoal-900 text-sm truncate max-w-md"
              title={file.name}
            >
              {file.name}
            </h4>
            <div className="flex flex-wrap items-center gap-2 mt-1.5 text-xs text-charcoal-500 font-mono">
              <span className="bg-neutral-100 px-2 py-0.5 rounded border border-neutral-200">
                {formatBytes(file.size)}
              </span>
              {duration !== null && (
                <>
                  <span>•</span>
                  <span className="bg-neutral-100 px-2 py-0.5 rounded border border-neutral-200">
                    {duration.toFixed(1)}s
                  </span>
                </>
              )}
              <span>•</span>
              <span className="uppercase">{file.name.split('.').pop() || 'video'}</span>
            </div>
          </div>

          <p className="text-[11px] text-charcoal-500 mt-2 bg-neutral-50 p-2 rounded-lg border border-neutral-100">
            Backend will sample 4 evenly spaced frames (20%, 40%, 60%, 80%), detect faces with Haar Cascade, and classify each face crop.
          </p>

          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-neutral-100">
            <button
              onClick={onChangeFile}
              disabled={disabled}
              type="button"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-charcoal-700 hover:text-charcoal-900 bg-neutral-100 hover:bg-neutral-200 transition-colors disabled:opacity-50"
            >
              <RefreshCw className="w-3 h-3" />
              Change Video
            </button>
            <button
              onClick={onRemove}
              disabled={disabled}
              type="button"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-rose-600 hover:text-rose-700 hover:bg-rose-50 transition-colors disabled:opacity-50"
            >
              <X className="w-3.5 h-3.5" />
              Remove
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
