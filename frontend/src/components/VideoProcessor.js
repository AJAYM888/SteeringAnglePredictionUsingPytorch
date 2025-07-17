import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import axios from 'axios';
import { Video, Play, Pause, RotateCcw } from 'lucide-react';

const VideoProcessor = ({ onPrediction, setLoading }) => {
  const [video, setVideo] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [predictions, setPredictions] = useState([]);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setVideo(URL.createObjectURL(file));
      setProcessing(true);
      setLoading(true);

      const formData = new FormData();
      formData.append('video', file);

      try {
        const response = await axios.post('http://localhost:5000/api/predict-video', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        setPredictions(response.data.predictions);
        if (response.data.predictions.length > 0) {
          onPrediction(response.data.predictions[0]);
        }
      } catch (error) {
        console.error('Video processing error:', error);
      } finally {
        setProcessing(false);
        setLoading(false);
      }
    }
  }, [onPrediction, setLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov']
    },
    multiple: false
  });

  return (
    <div className="video-processor">
      <h2>Video Processing</h2>
      
      {!video ? (
        <motion.div
          {...getRootProps()}
          className={`video-dropzone ${isDragActive ? 'active' : ''}`}
          whileHover={{ scale: 1.02 }}
        >
          <input {...getInputProps()} />
          <Video size={48} className="upload-icon" />
          <p>Drop a driving video here, or click to select</p>
          <span>Supports: MP4, AVI, MOV</span>
        </motion.div>
      ) : (
        <div className="video-container">
          <video
            src={video}
            controls
            className="video-player"
            onTimeUpdate={(e) => {
              const frame = Math.floor(e.target.currentTime * 30); // Assuming 30fps
              setCurrentFrame(frame);
              if (predictions[frame]) {
                onPrediction(predictions[frame]);
              }
            }}
          />
          
          {processing && (
            <div className="processing-overlay">
              <div className="processing-spinner"></div>
              <span>Processing video frames...</span>
            </div>
          )}
        </div>
      )}

      {predictions.length > 0 && (
        <div className="video-stats">
          <h3>Video Analysis</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span>Total Frames:</span>
              <span>{predictions.length}</span>
            </div>
            <div className="stat-item">
              <span>Avg Steering:</span>
              <span>
                {(predictions.reduce((sum, p) => sum + Math.abs(p.angle), 0) / predictions.length).toFixed(1)}°
              </span>
            </div>
            <div className="stat-item">
              <span>Max Turn:</span>
              <span>
                {Math.max(...predictions.map(p => Math.abs(p.angle))).toFixed(1)}°
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoProcessor;
