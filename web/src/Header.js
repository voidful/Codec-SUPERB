import React from 'react';
import { motion } from 'framer-motion';
import { Github, FileText, Zap } from 'lucide-react';
import './Header.css';

function Header() {
  return (
    <motion.header
      className="glass-panel"
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <div className="header-logo">
        <Zap className="logo-icon" size={28} />
        <span className="text-gradient">Codec Superb</span>
      </div>
      <nav>
        <ul>
          <li>
            <a href="#paper" className="nav-link">
              <FileText size={18} />
              <span>Paper</span>
            </a>
          </li>
          <li>
            <a href="https://github.com/voidful/Codec-SUPERB" className="nav-link">
              <Github size={18} />
              <span>Code</span>
            </a>
          </li>
        </ul>
      </nav>
    </motion.header>
  );
}

export default Header;