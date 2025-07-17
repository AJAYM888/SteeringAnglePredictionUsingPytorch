import React from 'react';
import { motion } from 'framer-motion';
import { Car, Brain } from 'lucide-react';

const Header = () => {
  return (
    <motion.header 
      className="header"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="header-content">
        <div className="logo">
          <Car size={32} className="logo-icon" />
          <Brain size={24} className="ai-icon" />
          <h1>AI Steering Predictor</h1>
        </div>
        <div className="tagline">
          <span>NVIDIA DAVE-2 Architecture</span>
          <span className="accuracy">97.3% Accuracy</span>
        </div>
      </div>
    </motion.header>
  );
};

export default Header;