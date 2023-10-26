import React from 'react'
import ResultsTable from './ResultsTable'
import './App.css'

function App() {
  const results = {
    librispeech_asr_dummy: {
      descript_audio_codec: {
        mel: 0.6383,
        sisdr: -10.7434,
        stft: 1.8450,
        'visqol-audio': 4.3769,
        'visqol-speech': 4.5964,
        waveform: 0.0110,
      },
      encodec_hf: {
        mel: 1.3255,
        sisdr: -1.6464,
        stft: 2.4462,
        'visqol-audio': 4.0702,
        'visqol-speech': 3.4563,
        waveform: 0.0274,
      },
      speech_tokenizer: {
        mel: 0.7914,
        sisdr: -3.4359,
        stft: 1.6290,
        'visqol-audio': 4.2951,
        'visqol-speech': 4.1370,
        waveform: 0.0230,
      },
    },
  }

  return (
    <div className="App">
      <h1>LibriSpeech ASR Dummy Results</h1>
      <ResultsTable results={results} />
    </div>
  )
}

export default App
