import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import './App.css';
import ImageUpload from './components/ImageUpload';
import PredictionResult from './components/PredictionResult';
import SteeringWheel from './components/SteeringWheel';
import VideoProcessor from './components/VideoProcessor';
import ModelStats from './components/ModelStats';
import Header from './components/Header';
import { Upload, Camera, Video, BarChart3 } from 'lucide-react';

function App() {
  const [activeTab, setActiveTab] = useState('image');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [confidence, setConfidence] = useState(null);
  const [processingTime, setProcessingTime] = useState(null);

  const handlePrediction = useCallback((result) => {
    setPrediction(result.angle);
    setConfidence(result.confidence);
    setProcessingTime(result.processing_time);
  }, []);

  const tabs = [
    { id: 'image', label: 'Image Upload', icon: Upload },
    { id: 'camera', label: 'Live Camera', icon: Camera },
    { id: 'video', label: 'Video Processing', icon: Video },
    { id: 'stats', label: 'Model Stats', icon: BarChart3 }
  ];

  return (
    <div className="app">
      <Header />
      
      {/* Navigation Tabs */}
      <nav className="nav-tabs">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? 'active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            <Icon size={20} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-grid">
          {/* Left Panel - Input */}
          <motion.div 
            className="input-panel"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            {activeTab === 'image' && (
              <ImageUpload 
                onPrediction={handlePrediction}
                setLoading={setLoading}
              />
            )}
            {activeTab === 'video' && (
              <VideoProcessor 
                onPrediction={handlePrediction}
                setLoading={setLoading}
              />
            )}
            {activeTab === 'stats' && <ModelStats />}
          </motion.div>

          {/* Right Panel - Results */}
          <motion.div 
            className="results-panel"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="steering-display">
              <SteeringWheel 
                angle={prediction || 0} 
                animated={!!prediction}
              />
              
              {prediction !== null && (
                <PredictionResult 
                  angle={prediction}
                  confidence={confidence}
                  processingTime={processingTime}
                  loading={loading}
                />
              )}
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}

export default App;