from codec.general import save_audio
import dac
from audiotools import AudioSignal


class Codec:
    def __init__(self, pretrained_model_name="descript-audio-codec"):

        ''' 
        Reference: https://github.com/descriptinc/descript-audio-codec
        
        pip install descript-audio-codec 

        or

        pip install git+https://github.com/descriptinc/descript-audio-codec

        '''
        
        # Load model.
        self.model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(self.model_path)
        self.device = "cuda"
        self.model.to(self.device)

        # Sample rate.
        self.sampling_rate = 44_100


    def synth(self, data):

        # Load audio.
        audio_path = data["audio"]["path"]
        audio_signal = AudioSignal(audio_path)

        # Check audio format.
        if audio_signal.sample_rate != self.sampling_rate:
            audio_signal.resample(self.sampling_rate)

        # Encode audio signal.
        compressed_audio = self.model.compress(audio_signal)

        # Decode audio signal back to an AudioSignal.
        decompressed_audio = self.model.decompress(compressed_audio).audio_data.squeeze(0)

        # Save audio.
        audio_path = f"dummy/{data['id']}.wav"
        save_audio(decompressed_audio, audio_path, self.sampling_rate)
        data['audio'] = audio_path

        return data