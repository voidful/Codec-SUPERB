import React from 'react';
import Leaderboard from './Leaderboard'
import './App.css'
import Header from './Header';
import Card from './Card';
import results from './results/data';

function App() {
  return (
    <div className="App">
      <Header />
      <div className="landing-page">
      <Card title="Welcome to Codec Superb!">
        <p>This study introduces Codec-SUPERB, a benchmark for evaluating sound codec models across key tasks, promoting advancements through a community-driven database and in-depth analysis.</p>
        <img src="Overview.png" alt="Codec Superb Overview" />
      </Card>
      <Card>
          <h1>Results</h1>
          <Leaderboard results={results} />
      </Card>
    </div>
    </div>
  );
}

export default App
