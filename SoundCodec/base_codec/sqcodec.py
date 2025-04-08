# Reference: https://github.com/zhai-lw/SQCodec

from functools import lru_cache

import torch

from SoundCodec.base_codec.general import ExtractedUnit, save_audio


class BaseCodec:
    def __init__(self, *args, **kwargs):
        try:
            import sq_codec
        except ImportError:
            raise Exception("Please install sq_codec first. pip install sq_codec")
        self.config_name: str = ""
        self.config()
        codec = sq_codec.get_model(self.config_name, model_dir="./sq_codec")
        self.model = codec.network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.sampling_rate = codec.config.sample_rate

    def config(self):
        raise NotImplementedError

    @lru_cache
    def resample_func(self, orig_sample_rate: int):
        if orig_sample_rate != self.sampling_rate:
            import torchaudio.transforms
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sampling_rate,
                lowpass_filter_width=16,
                rolloff=0.85,
                resampling_method="sinc_interp_kaiser",
                beta=8.555504641634386,
            )
        else:
            resampler = torch.nn.Identity()
        return resampler.to(self.device)

    def extract_unit(self, data_item):
        wav, sr = data_item["audio"]["array"], data_item["audio"]["sampling_rate"]
        wav = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            audio_data = self.resample_func(sr)(wav)
            audio_data, audio_length = self.model.preprocess(audio_data)
            feature = self.model.encoder(audio_data.unsqueeze(1))
            trans_feature = self.model.en_encoder(feature)
            q_trans_feature, indices, _ = self.model.quantizer(trans_feature)

        return ExtractedUnit(
            unit=indices[0].unsqueeze(0),
            stuff_for_synth=(q_trans_feature, audio_length),
        )

    def decode_unit(self, stuff_for_synth):
        q_trans_feature, audio_length = stuff_for_synth

        with torch.inference_mode():
            q_feature = self.model.en_decoder(q_trans_feature)
            audio_data = self.model.decoder(q_feature).squeeze(1)

        return audio_data[:, :audio_length].cpu().numpy()

    def synth(self, data, local_save=False):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy_sqcodec_{self.config_name}/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data
