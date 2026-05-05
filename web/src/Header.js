import React from 'react';
import { Database, FileText, Github, Quote } from 'lucide-react';
import './Header.css';

function Header() {
  return (
    <header>
      <div className="header-logo">
        <span className="logo-mark">CS</span>
        <span>Codec-SUPERB</span>
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
            <a href="#leaderboard" className="nav-link">
              <Database size={18} />
              <span>Results</span>
            </a>
          </li>
          <li>
            <a href="#citation" className="nav-link">
              <Quote size={18} />
              <span>Citation</span>
            </a>
          </li>
          <li>
            <a href="https://github.com/voidful/Codec-SUPERB" className="nav-link external-link">
              <Github size={18} />
              <span>GitHub</span>
            </a>
          </li>
        </ul>
      </nav>
    </header>
  );
}

export default Header;
