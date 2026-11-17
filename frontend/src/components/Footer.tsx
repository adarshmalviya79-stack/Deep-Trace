import React from 'react';
import { ShieldCheck } from 'lucide-react';

interface FooterProps {
  onNavigate: (page: 'home' | 'detector' | 'about') => void;
}

export const Footer: React.FC<FooterProps> = ({ onNavigate }) => {
  return (
    <footer className="border-t border-border-subtle bg-white mt-20">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
          {/* Brand Info */}
          <div className="md:col-span-2 space-y-3">
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-lg bg-charcoal-900 text-white flex items-center justify-center">
                <ShieldCheck className="w-4 h-4 text-indigo-400" />
              </div>
              <span className="font-semibold text-charcoal-900 text-base">DeepTrace</span>
            </div>
            <p className="text-xs text-charcoal-500 max-w-sm leading-relaxed">
              Precision deepfake image and video detection powered by PyTorch EfficientNet-B0 and Haar Cascade face localization. Designed for digital media authenticity analysis.
            </p>
          </div>

          {/* Quick Navigation */}
          <div>
            <h5 className="text-xs font-semibold uppercase tracking-wider text-charcoal-900 mb-3">
              Platform
            </h5>
            <ul className="space-y-2 text-xs text-charcoal-600">
              <li>
                <button
                  onClick={() => onNavigate('home')}
                  className="hover:text-indigo-600 transition-colors"
                >
                  Home
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('detector')}
                  className="hover:text-indigo-600 transition-colors"
                >
                  Detection Workspace
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigate('about')}
                  className="hover:text-indigo-600 transition-colors"
                >
                  Architecture & Workflow
                </button>
              </li>
            </ul>
          </div>

          {/* Technology Stack */}
          <div>
            <h5 className="text-xs font-semibold uppercase tracking-wider text-charcoal-900 mb-3">
              Inference Stack
            </h5>
            <ul className="space-y-1.5 text-xs text-charcoal-500 font-mono">
              <li>EfficientNet-B0 (CNN)</li>
              <li>PyTorch & Torchvision</li>
              <li>OpenCV Haar Cascades</li>
              <li>FastAPI Async Engine</li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-neutral-100 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-charcoal-500">
          <p>© {new Date().getFullYear()} DeepTrace Research. All rights reserved.</p>
          <p className="text-[11px] text-charcoal-400">
            Model predictions are statistical estimates and do not constitute definitive legal proof of authenticity.
          </p>
        </div>
      </div>
    </footer>
  );
};
