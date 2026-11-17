import React, { useState } from 'react';
import { ShieldCheck, Menu, X, ArrowUpRight } from 'lucide-react';
import { BackendStatus } from './BackendStatus';

interface NavbarProps {
  activePage: 'home' | 'detector' | 'about';
  setActivePage: (page: 'home' | 'detector' | 'about') => void;
  onSelectDetectorMode?: (mode: 'image' | 'video') => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activePage,
  setActivePage,
  onSelectDetectorMode,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleNavClick = (page: 'home' | 'detector' | 'about') => {
    setActivePage(page);
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleModeClick = (mode: 'image' | 'video') => {
    setActivePage('detector');
    if (onSelectDetectorMode) {
      onSelectDetectorMode(mode);
    }
    setMobileMenuOpen(false);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header className="sticky top-0 z-40 bg-[#FAF9F6]/90 backdrop-blur-md border-b border-border-subtle">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div
          onClick={() => handleNavClick('home')}
          className="flex items-center gap-2.5 cursor-pointer group"
        >
          <div className="w-8 h-8 rounded-lg bg-charcoal-900 text-white flex items-center justify-center shadow-subtle group-hover:bg-indigo-600 transition-colors">
            <ShieldCheck className="w-5 h-5 text-indigo-300 group-hover:text-white transition-colors" />
          </div>
          <div className="flex flex-col">
            <span className="font-semibold text-base tracking-tight text-charcoal-900 flex items-center gap-1.5">
              DeepTrace
              <span className="text-[10px] font-mono tracking-normal uppercase bg-neutral-100 text-charcoal-600 px-1.5 py-0.5 rounded border border-neutral-200">
                v1.0
              </span>
            </span>
            <span className="text-[10px] text-charcoal-500 font-normal tracking-wide">
              AI Authenticity Engine
            </span>
          </div>
        </div>

        {/* Desktop Navigation */}
        <nav className="hidden md:flex items-center gap-1">
          <button
            onClick={() => handleNavClick('home')}
            className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activePage === 'home'
                ? 'text-indigo-600 bg-indigo-50/60 font-semibold'
                : 'text-charcoal-600 hover:text-charcoal-900 hover:bg-neutral-100/70'
            }`}
          >
            Home
          </button>
          <button
            onClick={() => handleNavClick('detector')}
            className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activePage === 'detector'
                ? 'text-indigo-600 bg-indigo-50/60 font-semibold'
                : 'text-charcoal-600 hover:text-charcoal-900 hover:bg-neutral-100/70'
            }`}
          >
            Detector
          </button>
          <button
            onClick={() => handleNavClick('about')}
            className={`px-3.5 py-1.5 rounded-lg text-sm font-medium transition-colors ${
              activePage === 'about'
                ? 'text-indigo-600 bg-indigo-50/60 font-semibold'
                : 'text-charcoal-600 hover:text-charcoal-900 hover:bg-neutral-100/70'
            }`}
          >
            About
          </button>
        </nav>

        {/* Right action / Backend Status */}
        <div className="hidden md:flex items-center gap-3">
          <BackendStatus compact />
          <button
            onClick={() => handleModeClick('image')}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-charcoal-900 hover:bg-indigo-600 rounded-lg shadow-subtle transition-colors"
          >
            <span>Launch Detector</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Mobile menu toggle */}
        <div className="flex md:hidden items-center gap-2">
          <BackendStatus compact />
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 rounded-lg text-charcoal-700 hover:bg-neutral-100 transition-colors"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-border-subtle bg-white px-4 py-4 space-y-2 shadow-lg">
          <button
            onClick={() => handleNavClick('home')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium ${
              activePage === 'home' ? 'text-indigo-600 bg-indigo-50 font-semibold' : 'text-charcoal-700 hover:bg-neutral-50'
            }`}
          >
            Home
          </button>
          <button
            onClick={() => handleNavClick('detector')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium ${
              activePage === 'detector' ? 'text-indigo-600 bg-indigo-50 font-semibold' : 'text-charcoal-700 hover:bg-neutral-50'
            }`}
          >
            Detector
          </button>
          <button
            onClick={() => handleNavClick('about')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium ${
              activePage === 'about' ? 'text-indigo-600 bg-indigo-50 font-semibold' : 'text-charcoal-700 hover:bg-neutral-50'
            }`}
          >
            About Architecture & Workflow
          </button>
          <div className="pt-2 border-t border-neutral-100 flex gap-2">
            <button
              onClick={() => handleModeClick('image')}
              className="flex-1 text-center py-2 text-xs font-medium bg-neutral-100 text-charcoal-800 rounded-lg hover:bg-neutral-200"
            >
              Analyze Image
            </button>
            <button
              onClick={() => handleModeClick('video')}
              className="flex-1 text-center py-2 text-xs font-medium bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              Analyze Video
            </button>
          </div>
        </div>
      )}
    </header>
  );
};
