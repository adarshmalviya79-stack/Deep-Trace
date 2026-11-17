import React, { useState } from 'react';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { Home } from './pages/Home';
import { Detector } from './pages/Detector';
import { About } from './pages/About';
import { DetectionMode } from './types';

export const App: React.FC = () => {
  const [activePage, setActivePage] = useState<'home' | 'detector' | 'about'>('home');
  const [detectorMode, setDetectorMode] = useState<DetectionMode>('image');

  const handleSelectDetectorMode = (mode: DetectionMode) => {
    setDetectorMode(mode);
    setActivePage('detector');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavigate = (page: 'home' | 'detector' | 'about') => {
    setActivePage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#FAF9F6] text-charcoal-900">
      {/* Top Navigation */}
      <Navbar
        activePage={activePage}
        setActivePage={handleNavigate}
        onSelectDetectorMode={handleSelectDetectorMode}
      />

      {/* Main Page Content */}
      <main className="flex-1">
        {activePage === 'home' && (
          <Home
            onSelectMode={handleSelectDetectorMode}
            onNavigateToAbout={() => handleNavigate('about')}
          />
        )}
        {activePage === 'detector' && (
          <Detector
            key={detectorMode}
            initialMode={detectorMode}
          />
        )}
        {activePage === 'about' && <About />}
      </main>

      {/* Footer */}
      <Footer onNavigate={handleNavigate} />
    </div>
  );
};

export default App;
