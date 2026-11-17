import React, { useEffect, useState } from 'react';
import { X, RefreshCw, CheckCircle2 } from 'lucide-react';
import { formatBytes } from './UploadZone';

interface ImagePreviewProps {
  file: File;
  onRemove: () => void;
  onChangeFile: () => void;
  disabled?: boolean;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  file,
  onRemove,
  onChangeFile,
  disabled = false,
}) => {
  const [previewUrl, setPreviewUrl] = useState<string>('');

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  return (
    <div className="bg-white rounded-2xl border border-neutral-200 p-4 shadow-subtle">
      <div className="flex flex-col sm:flex-row items-center gap-4">
        {/* Preview image */}
        <div className="relative w-full sm:w-36 h-36 bg-neutral-100 rounded-xl overflow-hidden border border-neutral-200 flex-shrink-0 flex items-center justify-center">
          {previewUrl ? (
            <img
              src={previewUrl}
              alt={file.name}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="text-xs text-charcoal-400">Loading preview…</div>
          )}
          <span className="absolute bottom-1 right-1 bg-black/70 backdrop-blur-sm text-white font-mono text-[10px] px-1.5 py-0.5 rounded">
            IMG
          </span>
        </div>

        {/* Details & Actions */}
        <div className="flex-1 w-full flex flex-col justify-between py-1">
          <div>
            <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-medium mb-1">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              File staged for inference
            </div>
            <h4
              className="font-semibold text-charcoal-900 text-sm truncate max-w-md"
              title={file.name}
            >
              {file.name}
            </h4>
            <div className="flex items-center gap-2 mt-1 text-xs text-charcoal-500 font-mono">
              <span className="bg-neutral-100 px-2 py-0.5 rounded border border-neutral-200">
                {formatBytes(file.size)}
              </span>
              <span>•</span>
              <span className="uppercase">{file.type || 'image'}</span>
            </div>
          </div>

          <div className="flex items-center gap-2 mt-4 pt-3 border-t border-neutral-100">
            <button
              onClick={onChangeFile}
              disabled={disabled}
              type="button"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-charcoal-700 hover:text-charcoal-900 bg-neutral-100 hover:bg-neutral-200 transition-colors disabled:opacity-50"
            >
              <RefreshCw className="w-3 h-3" />
              Change Image
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
