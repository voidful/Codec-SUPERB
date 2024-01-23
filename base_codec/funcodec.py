import nlp2
import torch

from codec.general import save_audio
from audiotools import AudioSignal


class BaseCodec:
    def __init__(self):
        # Reference: https://github.com/alibaba-damo-academy/FunCodec
        try:
            from funcodec.bin.codec_inference import Speech2Token
        except:
            raise Exception(
                "Please install funcodec first. pip install git+https://github.com/alibaba-damo-academy/FunCodec.git")
        self.config()
        self.model = Speech2Token(self.config_path, self.ckpt_path, device='cuda')

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

    def synth(self, data, save_audio_flag=True):
        with torch.no_grad():
            extract_data = self.extract_unit(data, return_unit_only=False)
            audio_array = extract_data["recon_speech"][0].cpu().numpy()
            if save_audio_flag:
                audio_path = f"dummy-funcodec-{self.setting}/{data['id']}.wav"
                save_audio(extract_data["recon_speech"][0].cpu(), audio_path, self.sampling_rate)
                data['audio'] = audio_path
            else:
                data['audio']['array'] = extract_data["recon_speech"][0].cpu().numpy()
            return data

    def extract_unit(self, data, return_unit_only=True):
        with torch.no_grad():
            audio_signal = AudioSignal(data["audio"]['array'], data["audio"]['sampling_rate'])

            if audio_signal.sample_rate != self.sampling_rate:
                audio_signal.resample(self.sampling_rate)

            code_indices, code_embeddings, recon_speech, sub_quants = self.model(audio_signal.audio_data[0].cuda())
            if return_unit_only:
                return code_indices[0].permute(1, 0, 2).squeeze(0)
            return {"code_indices": code_indices, "code_embeddings": code_embeddings, "recon_speech": recon_speech}
