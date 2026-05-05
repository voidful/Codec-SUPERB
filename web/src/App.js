import React from 'react';
import Leaderboard from './Leaderboard'
import './App.css'
import Header from './Header';
import results from './results/data';

function App() {
  const modelCount = Object.keys(results).length;
  const bestOverallMel = Object.entries(results)
    .filter(([, value]) => typeof value.overall_mel === 'number')
    .sort((a, b) => a[1].overall_mel - b[1].overall_mel)[0];

  return (
    <div className="App">
      <Header />

      <main>
        <section className="paper-hero" id="paper">
          <p className="venue">Findings of ACL 2024</p>
          <h1>Codec-SUPERB: An In-Depth Analysis of Sound Codec Models</h1>
          <p className="authors">
            Haibin Wu, Ho-Lam Chung, Yi-Cheng Lin, Yuan-Kuei Wu, Xuanjun Chen,
            Yu-Chi Pai, Hsiu-Hsuan Wang, Kai-Wei Chang, Alexander H. Liu, Hung-yi Lee
          </p>
          <p className="abstract">
            Codec-SUPERB evaluates neural audio codecs through downstream speech,
            audio, and music tasks, measuring how much task-relevant information is
            preserved after tokenization and reconstruction.
          </p>
          <div className="resource-links" aria-label="Project resources">
            <a href="https://aclanthology.org/2024.findings-acl.616/">ACL Anthology</a>
            <a href="https://arxiv.org/abs/2402.13071">arXiv</a>
            <a href="https://github.com/voidful/Codec-SUPERB">Code</a>
            <a href="https://huggingface.co/datasets/voidful/codec-superb-tiny">Tiny Dataset</a>
          </div>
        </section>

        <section className="overview-section" aria-labelledby="overview-title">
          <div className="section-copy">
            <p className="section-kicker">Benchmark Scope</p>
            <h2 id="overview-title">A unified view of codec behavior across modalities</h2>
            <p>
              The benchmark keeps the same evaluation protocol across codec families and
              reports reconstruction quality by category. The tiny rerun uses the public
              Hugging Face subset with balanced Speech, Audio, and Music splits for fast
              regression checks.
            </p>
          </div>
          <figure className="overview-figure">
            <img src="Overview.png" alt="Codec-SUPERB benchmark overview" />
          </figure>
        </section>

        <section className="results-intro" id="leaderboard">
          <div>
            <p className="section-kicker">Tiny Rerun</p>
            <h2>Latest codec-superb-tiny results</h2>
            <p>
              Metrics are averaged over the 6,000-example tiny dataset and grouped by
              Speech, Audio, Music, and Overall. Lower MEL is better; higher PESQ, STOI,
              and F0Corr are better.
            </p>
          </div>
          <dl className="result-facts">
            <div>
              <dt>Models</dt>
              <dd>{modelCount}</dd>
            </div>
            <div>
              <dt>Dataset</dt>
              <dd>6k rows</dd>
            </div>
            <div>
              <dt>Best Overall MEL</dt>
              <dd>{bestOverallMel ? bestOverallMel[0] : 'N/A'}</dd>
            </div>
          </dl>
        </section>

        <section className="leaderboard-section" aria-label="Codec-SUPERB leaderboard">
          <Leaderboard results={results} />
        </section>

        <section className="citation-section" id="citation">
          <h2>Citation</h2>
          <pre>{`@inproceedings{wu-etal-2024-codec,
  title = "Codec-{SUPERB}: An In-Depth Analysis of Sound Codec Models",
  author = "Wu, Haibin and Chung, Ho-Lam and Lin, Yi-Cheng and Wu, Yuan-Kuei and Chen, Xuanjun and Pai, Yu-Chi and Wang, Hsiu-Hsuan and Chang, Kai-Wei and Liu, Alexander and Lee, Hung-yi",
  booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
  year = "2024",
  url = "https://aclanthology.org/2024.findings-acl.616",
  doi = "10.18653/v1/2024.findings-acl.616",
  pages = "10330--10348"
}`}</pre>
        </section>
      </main>

      <footer className="main-footer">
        <p>Codec-SUPERB Project - {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}

export default App;
