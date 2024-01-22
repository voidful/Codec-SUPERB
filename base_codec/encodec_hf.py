import torch
from transformers import AutoModel, AutoProcessor
from codec.general import save_audio


class BaseCodec:
    def __init__(self):
        self.config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(self.pretrained_model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
        self.sampling_rate = self.processor.sampling_rate

    def config(self):
        self.pretrained_model_name = "facebook/encodec_24khz"

    @torch.no_grad()
    def synth(self, data, save_audio=True):
        encoder_outputs, padding_mask = self.extract_unit(data, return_unit_only=False)
        audio_values = \
            self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
        if save_audio:
            audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
            save_audio(audio_values[0].cpu(), audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values[0].cpu().numpy()
        return data

    @torch.no_grad()
    def extract_unit(self, data, return_unit_only=True):
        audio_sample = data["audio"]["array"]
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
        encoder_outputs = self.model.encode(input_values, padding_mask)
        if return_unit_only:
            return encoder_outputs.audio_codes.squeeze()
        return encoder_outputs, padding_mask
