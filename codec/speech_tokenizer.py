from codec.general import save_audio
from speechtokenizer import SpeechTokenizer
import torchaudio
import torch


class Codec:
    def __init__(self, pretrained_model_name="fnlp/SpeechTokenizer"):

        '''

        Reference: https://github.com/ZhangXInFD/SpeechTokenizer

        pip install -U speechtokenizer

        or you can clone the repo and install locally

        git clone https://github.com/ZhangXInFD/SpeechTokenizer.git
        cd SpeechTokenizer
        pip install .
        
        '''
        
        # git clone https://huggingface.co/fnlp/SpeechTokenizer
        self.config_path = "/home/kuanyiiii/kuanyi/neural-codec/SpeechTokenizer/speechtokenizer_hubert_avg/config.json"

        # wget https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt
        self.ckpt_path = "/home/kuanyiiii/kuanyi/neural-codec/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt"

        # Load model.
        self.model = SpeechTokenizer.load_from_checkpoint(self.config_path, self.ckpt_path)
        self.model.eval()

        # Sample rate.
        self.sampling_rate = self.model.sample_rate


    def synth(self, data):

        # Load audio.
        audio_path = data["audio"]["path"]
        wav, sampling_rate = torchaudio.load(audio_path)

        # Check audio format.
        if sampling_rate != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sampling_rate, self.sampling_rate)
        
        wav = wav.unsqueeze(0)

        # Extract discrete codes from SpeechTokenizer
        with torch.no_grad():
            codes = self.model.encode(wav) # codes: (n_q, B, T)

        RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
        RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer

        # Concatenating semantic tokens (RVQ_1) and supplementary timbre tokens and then decoding
        wav = self.model.decode(torch.cat([RVQ_1, RVQ_supplement], axis=0))
        wav = wav.detach().cpu().squeeze(0)

        # Save audio.
        audio_path = f"dummy/{data['id']}.wav"
        save_audio(wav, audio_path, self.sampling_rate)
        data['audio'] = audio_path

        return data