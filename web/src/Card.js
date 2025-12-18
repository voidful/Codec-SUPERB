import React from 'react';
import { motion } from 'framer-motion';
import './Card.css';

const Card = ({ title, children, delay = 0 }) => {
  return (
    <motion.div
      className="card glass-panel"
      initial={{ y: 30, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6, delay: delay, ease: "easeOut" }}
    >
      {title && <h2 className="card-title text-gradient">{title}</h2>}
      <div className="card-content">{children}</div>
    </motion.div>
  );
};

export default Card;
