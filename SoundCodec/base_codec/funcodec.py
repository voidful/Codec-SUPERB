import nlp2
import torch
import os

from SoundCodec.base_codec.general import save_audio, ExtractedUnit
from audiotools import AudioSignal


class BaseCodec:
    def __init__(self):
        # Reference: https://github.com/alibaba-damo-academy/FunCodec
        try:
            from funcodec.bin.codec_inference import Speech2Token
        except:
            raise Exception(
                "Please install funcodec first. pip install git+https://github.com/voidful/FunCodec.git")
        os.makedirs("funcodec", exist_ok=True)
        self.config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Speech2Token(self.config_path, self.ckpt_path, device=self.device)

    def config(self):
        self.setting = "funcodec_zh_en_general_16k_nq32ds640"
        self.sampling_rate = 16000
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/raw/main/config.yaml',
            f"funcodec/{self.setting}")
        self.config_path = f"funcodec/{self.setting}/config.yaml"
        nlp2.download_file(
            'https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/resolve/main/model.pth',
            f"funcodec/{self.setting}")
        self.ckpt_path = f"funcodec/{self.setting}/model.pth"

    @torch.no_grad()
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_array = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy-funcodec-{self.setting}/{data['id']}.wav"
            save_audio(audio_array, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_array
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])
        code_indices, code_embeddings, recon_speech, sub_quants = self.model(
            audio_signal.audio_data[0].to(self.device))
        return ExtractedUnit(
            unit=code_indices[0].permute(1, 0, 2).squeeze(0),
            stuff_for_synth={"code_indices": code_indices, "code_embeddings": code_embeddings,
                             "recon_speech": recon_speech}
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        extract_data = stuff_for_synth
        audio_array = extract_data["recon_speech"][0].cpu().numpy()
        return audio_array
