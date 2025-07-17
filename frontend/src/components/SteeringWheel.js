import React from 'react';
import { motion } from 'framer-motion';

const SteeringWheel = ({ angle, animated }) => {
  return (
    <div className="steering-wheel-container">
      <h3>Predicted Steering Angle</h3>
      
      <motion.div 
        className="steering-wheel"
        animate={{ rotate: animated ? -angle : 0 }}
        transition={{ 
          type: "spring", 
          stiffness: 100, 
          damping: 15,
          duration: 0.8 
        }}
      >
        {/* Steering Wheel SVG */}
        <svg viewBox="0 0 200 200" className="wheel-svg">
          {/* Outer Ring */}
          <circle
            cx="100"
            cy="100"
            r="85"
            fill="none"
            stroke="#2a2a2a"
            strokeWidth="8"
          />
          
          {/* Inner Hub */}
          <circle
            cx="100"
            cy="100"
            r="25"
            fill="#1a1a1a"
          />
          
          {/* Spokes */}
          <line x1="100" y1="25" x2="100" y2="75" stroke="#2a2a2a" strokeWidth="6" />
          <line x1="100" y1="125" x2="100" y2="175" stroke="#2a2a2a" strokeWidth="6" />
          <line x1="25" y1="100" x2="75" y2="100" stroke="#2a2a2a" strokeWidth="6" />
          <line x1="125" y1="100" x2="175" y2="100" stroke="#2a2a2a" strokeWidth="6" />
          
          {/* Top Indicator */}
          <rect x="95" y="15" width="10" height="15" fill="#ff6b6b" rx="2" />
        </svg>
      </motion.div>
      
      <div className="angle-display">
        <span className="angle-value">
          {animated ? `${angle.toFixed(1)}°` : '0.0°'}
        </span>
        <span className="angle-label">
          {angle > 0 ? 'Right Turn' : angle < 0 ? 'Left Turn' : 'Straight'}
        </span>
      </div>
    </div>
  );
};

export default SteeringWheel;