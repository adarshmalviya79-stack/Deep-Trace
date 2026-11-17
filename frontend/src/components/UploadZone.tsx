import React, { useRef, useState, DragEvent, ChangeEvent } from 'react';
import { Image as ImageIcon, Video as VideoIcon, AlertCircle } from 'lucide-react';
import { DetectionMode } from '../types';

interface UploadZoneProps {
  mode: DetectionMode;
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
  disabled?: boolean;
}

const ACCEPTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
const ACCEPTED_VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv'];

const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB
const MAX_VIDEO_SIZE_BYTES = 100 * 1024 * 1024; // 100 MB

export const formatBytes = (bytes: number, decimals = 1): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
};

export const UploadZone: React.FC<UploadZoneProps> = ({
  mode,
  onFileSelect,
  disabled = false,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const isImageMode = mode === 'image';
  const acceptedExtensions = isImageMode ? ACCEPTED_IMAGE_EXTENSIONS : ACCEPTED_VIDEO_EXTENSIONS;
  const maxSize = isImageMode ? MAX_IMAGE_SIZE_BYTES : MAX_VIDEO_SIZE_BYTES;
  const maxSizeLabel = isImageMode ? '10 MB' : '100 MB';

  const validateAndSelect = (file: File) => {
    setValidationError(null);

    // Validate extension
    const ext = '.' + (file.name.split('.').pop() || '').toLowerCase();
    if (!acceptedExtensions.includes(ext)) {
      setValidationError(
        `Unsupported file type "${ext}". Please upload ${acceptedExtensions.join(', ').toUpperCase()}.`
      );
      return;
    }

    // Validate size
    if (file.size > maxSize) {
      setValidationError(
        `File is too large (${formatBytes(file.size)}). Maximum allowed size for ${
          isImageMode ? 'images' : 'videos'
        } is ${maxSizeLabel}.`
      );
      return;
    }

    if (file.size === 0) {
      setValidationError('Uploaded file appears to be empty (0 bytes).');
      return;
    }

    onFileSelect(file);
  };

  const handleDragEnter = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      validateAndSelect(droppedFile);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const chosenFile = e.target.files[0];
      validateAndSelect(chosenFile);
    }
    // reset input value so re-selecting same file triggers change
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptedExtensions.join(',')}
        onChange={handleFileInputChange}
        className="hidden"
        disabled={disabled}
      />

      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`relative group rounded-2xl border-2 border-dashed p-8 md:p-12 text-center transition-all cursor-pointer select-none ${
          disabled
            ? 'opacity-60 cursor-not-allowed bg-neutral-100/50 border-neutral-300'
            : isDragOver
            ? 'border-indigo-500 bg-indigo-50/50 scale-[1.005]'
            : 'border-neutral-300/90 hover:border-charcoal-600 bg-white hover:bg-neutral-50/40 shadow-sm'
        }`}
      >
        <div className="flex flex-col items-center justify-center max-w-sm mx-auto">
          {/* Icon */}
          <div
            className={`w-14 h-14 rounded-2xl flex items-center justify-center mb-4 transition-transform group-hover:scale-105 ${
              isDragOver
                ? 'bg-indigo-600 text-white'
                : 'bg-neutral-100 text-charcoal-700 group-hover:bg-indigo-50 group-hover:text-indigo-600'
            }`}
          >
            {isImageMode ? (
              <ImageIcon className="w-7 h-7" />
            ) : (
              <VideoIcon className="w-7 h-7" />
            )}
          </div>

          <h3 className="text-base font-semibold text-charcoal-900 mb-1">
            {isDragOver ? 'Drop file here' : `Drag and drop your ${isImageMode ? 'image' : 'video'} here`}
          </h3>

          <p className="text-xs text-charcoal-500 mb-4 leading-relaxed">
            or <span className="text-indigo-600 font-semibold underline decoration-indigo-200 group-hover:decoration-indigo-500">browse from your computer</span> to begin analysis
          </p>

          <div className="flex flex-wrap items-center justify-center gap-2 text-[11px] text-charcoal-500">
            <span className="font-mono bg-neutral-100 px-2 py-0.5 rounded border border-neutral-200">
              {acceptedExtensions.map((ext) => ext.replace('.', '').toUpperCase()).join(', ')}
            </span>
            <span className="text-neutral-300">•</span>
            <span>Max {maxSizeLabel}</span>
          </div>
        </div>
      </div>

      {/* Validation Error Banner */}
      {validationError && (
        <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-xl flex items-start gap-2.5 text-xs text-rose-800 animate-fadeIn">
          <AlertCircle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium">{validationError}</p>
          </div>
          <button
            onClick={() => setValidationError(null)}
            className="text-rose-500 hover:text-rose-700 font-medium text-xs ml-2"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
};
