import React from 'react';
import { motion } from 'framer-motion';
import Leaderboard from './Leaderboard'
import './App.css'
import Header from './Header';
import Card from './Card';
import results from './results/data';

function App() {
  return (
    <div className="App">
      <div className="blob-container">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
        <div className="blob blob-3"></div>
      </div>

      <Header />

      <main className="landing-page">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1 }}
        >
          <Card title="Welcome to Codec Superb!" delay={0.2}>
            <p>Codec-SUPERB is a pioneering benchmark for evaluating audio codec models across diverse tasks. We promote advancements through a community-driven database and deep performance analysis.</p>
            <div className="image-wrapper">
              <img src="Overview.png" alt="Codec Superb Overview" />
            </div>
          </Card>

          <Card title="Leaderboard" delay={0.4}>
            <p className="leaderboard-note">Comparing performance across <strong>Speech</strong>, <strong>Audio</strong>, and <strong>Music</strong> categories.</p>
            <div className="results-section">
              <Leaderboard results={results} />
            </div>
          </Card>
        </motion.div>
      </main>

      <footer className="main-footer">
        <p>© {new Date().getFullYear()} Codec Superb Project. Built for the Audio Research Community.</p>
      </footer>
    </div>
  );
}

export default App;
