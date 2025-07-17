import React from 'react';
import { motion } from 'framer-motion';
import { Clock, Zap, Target } from 'lucide-react';

const PredictionResult = ({ angle, confidence, processingTime, loading }) => {
  if (loading) {
    return (
      <motion.div 
        className="loading-container"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
      >
        <div className="loading-spinner"></div>
        <span>Processing image...</span>
      </motion.div>
    );
  }

  const getConfidenceColor = (conf) => {
    if (conf > 0.8) return '#4ade80';
    if (conf > 0.6) return '#fbbf24';
    return '#ef4444';
  };

  const getAngleCategory = (angle) => {
    const absAngle = Math.abs(angle);
    if (absAngle < 5) return 'Straight driving';
    if (absAngle < 15) return 'Gentle turn';
    if (absAngle < 30) return 'Moderate turn';
    return 'Sharp turn';
  };

  return (
    <motion.div 
      className="prediction-result"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3>Prediction Results</h3>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <Target className="metric-icon" />
          <div className="metric-content">
            <span className="metric-label">Steering Angle</span>
            <span className="metric-value">{angle.toFixed(2)}°</span>
            <span className="metric-sub">{getAngleCategory(angle)}</span>
          </div>
        </div>

        <div className="metric-card">
          <Zap className="metric-icon" />
          <div className="metric-content">
            <span className="metric-label">Confidence</span>
            <span 
              className="metric-value"
              style={{ color: getConfidenceColor(confidence) }}
            >
              {(confidence * 100).toFixed(1)}%
            </span>
            <span className="metric-sub">Model certainty</span>
          </div>
        </div>

        <div className="metric-card">
          <Clock className="metric-icon" />
          <div className="metric-content">
            <span className="metric-label">Processing Time</span>
            <span className="metric-value">{processingTime}ms</span>
            <span className="metric-sub">Inference speed</span>
          </div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div className="confidence-bar">
        <div className="confidence-label">
          <span>Prediction Confidence</span>
          <span>{(confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="confidence-track">
          <motion.div 
            className="confidence-fill"
            style={{ backgroundColor: getConfidenceColor(confidence) }}
            initial={{ width: 0 }}
            animate={{ width: `${confidence * 100}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </div>
      </div>

      {/* Driving Assessment */}
      <div className="driving-assessment">
        <h4>Driving Assessment</h4>
        <div className="assessment-items">
          <div className="assessment-item">
            <span>Vehicle Control:</span>
            <span className={Math.abs(angle) < 15 ? 'good' : 'moderate'}>
              {Math.abs(angle) < 15 ? 'Stable' : 'Active steering'}
            </span>
          </div>
          <div className="assessment-item">
            <span>Road Condition:</span>
            <span className="good">Clear visibility</span>
          </div>
          <div className="assessment-item">
            <span>Safety Level:</span>
            <span className={confidence > 0.7 ? 'good' : 'caution'}>
              {confidence > 0.7 ? 'High confidence' : 'Moderate confidence'}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PredictionResult;