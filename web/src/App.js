import React, { useState } from 'react';
import ResultsTable from './ResultsTable'
import './App.css'
import Header from './Header';
import Card from './Card';
import EERvsBPSChart from './EERvsBPSChart';

function App() {
  const [dataset, setDataset] = useState('librispeech_asr_dummy');
  const results = {
    librispeech_asr_dummy: {
    "encodec_hf": {
          "mel": 1.3255124418702844,
          "sisdr": -1.6464139425474034,
          "snr": 3.655806644322121,
          "stft": 2.4461940869893115,
          "visqol-audio": 4.070156077024855,
          "visqol-speech": 3.456325437523747,
          "waveform": 0.02736434547154054
      },
      "descript_audio_codec": {
          "mel": 0.6382959640189393,
          "sisdr": -10.74340238963088,
          "snr": 10.95413057118246,
          "stft": 1.8450361457589555,
          "visqol-audio": 4.376865724825919,
          "visqol-speech": 4.596361550109254,
          "waveform": 0.01096990620972563
      },

      "speech_tokenizer": {
          "mel": 0.7914425518414746,
          "sisdr": -3.435924078095449,
          "snr": 4.874410704390644,
          "stft": 1.6290033892409441,
          "visqol-audio": 4.295075452023233,
          "visqol-speech": 4.137006619717544,
          "waveform": 0.02297312009773434
      }
    },
    superb_ks:{
    "encodec_hf": {
          "mel": 1.4755834210851446,
          "sisdr": 4.597159708177816,
          "snr": 1.4560887231325252,
          "stft": 2.4747163176884786,
          "visqol-audio": 4.246047686934658,
          "visqol-speech": 3.0283790755226687,
          "waveform": 0.02525300618883001
      },
      "descript_audio_codec": {
          "mel": 0.6061258274011123,
          "sisdr": -10.610022181009947,
          "snr": 10.952649237404867,
          "stft": 1.671503691714292,
          "visqol-audio": 4.55817749600338,
          "visqol-speech": 4.350971831092887,
          "waveform": 0.011175317768815377
      },
      "speech_tokenizer": {
          "mel": 0.8619092436085967,
          "sisdr": 5.178341394048203,
          "snr": 1.4209948465785711,
          "stft": 1.8043810769901072,
          "visqol-audio": 4.384633406781557,
          "visqol-speech": 3.6159819495449947,
          "waveform": 0.026478095256902243
      }
  }
  }

  return (
    <div className="App">
      <Header />
      <div className="landing-page">
      <Card title="Welcome to Codec Superb!">
        <p>Behold! This is the no sidebar layout with no sidebar at all!</p>
      </Card>
      <Card title="Graph">
        <p>Behold! This is the no sidebar layout with no sidebar at all!</p>
      </Card>
      <Card title="Graph">
        <EERvsBPSChart></EERvsBPSChart>
      </Card>
      <Card>
          <h1>Results</h1>
          <div>
            <label>Select Dataset: </label>
            <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {Object.keys(results).map((key) => (
                <option key={key} value={key}>
                  {key}
                </option>
              ))}
            </select>
          </div>
          <ResultsTable dataset={dataset} results={results} />
      </Card>
    </div>


    </div>
  );
}

export default App
