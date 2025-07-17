import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import axios from 'axios';
import { Upload, Image as ImageIcon, X } from 'lucide-react';

const ImageUpload = ({ onPrediction, setLoading }) => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedImage(URL.createObjectURL(file));
      setLoading(true);

      const formData = new FormData();
      formData.append('image', file);

      try {
        const response = await axios.post('http://localhost:5000/api/predict-image', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        onPrediction(response.data);
      } catch (error) {
        console.error('Prediction error:', error);
      } finally {
        setLoading(false);
      }
    }
  }, [onPrediction, setLoading]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    multiple: false
  });

  const clearImage = () => {
    setUploadedImage(null);
    onPrediction({ angle: null, confidence: null, processing_time: null });
  };

  return (
    <div className="image-upload">
      <h2>Upload Road Image</h2>
      
      {!uploadedImage ? (
        <motion.div
          {...getRootProps()}
          className={`dropzone ${isDragActive ? 'active' : ''}`}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <input {...getInputProps()} />
          <Upload size={48} className="upload-icon" />
          <p>Drag & drop an image here, or click to select</p>
          <span>Supports: JPG, PNG, JPEG</span>
        </motion.div>
      ) : (
        <motion.div 
          className="image-preview"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
        >
          <button className="clear-button" onClick={clearImage}>
            <X size={20} />
          </button>
          <img src={uploadedImage} alt="Uploaded road" />
          <div className="image-info">
            <ImageIcon size={16} />
            <span>Road image uploaded</span>
          </div>
        </motion.div>
      )}

      <div className="sample-images">
        <h3>Try Sample Images:</h3>
        <div className="sample-grid">
          {['straight-road.jpg', 'curve-left.jpg', 'curve-right.jpg'].map((sample, index) => (
            <motion.button
              key={sample}
              className="sample-image"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                // Handle sample image selection
                setUploadedImage(`/samples/${sample}`);
                // Make prediction call
              }}
            >
              <img src={`/samples/${sample}`} alt={`Sample ${index + 1}`} />
              <span>{sample.replace('-', ' ').replace('.jpg', '')}</span>
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ImageUpload;