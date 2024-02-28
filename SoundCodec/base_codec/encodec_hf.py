import torch
from transformers import AutoModel, AutoProcessor
from SoundCodec.base_codec.general import save_audio, ExtractedUnit


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
    def synth(self, data, local_save=True):
        extracted_unit = self.extract_unit(data)
        data['unit'] = extracted_unit.unit
        audio_values = self.decode_unit(extracted_unit.stuff_for_synth)
        if local_save:
            audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
            save_audio(audio_values, audio_path, self.sampling_rate)
            data['audio'] = audio_path
        else:
            data['audio']['array'] = audio_values
        return data

    @torch.no_grad()
    def extract_unit(self, data):
        audio_sample = data["audio"]["array"]
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
        input_values = inputs["input_values"].to(self.device)
        padding_mask = inputs["padding_mask"].to(self.device) if inputs["padding_mask"] is not None else None
        encoder_outputs = self.model.encode(input_values, padding_mask)
        return ExtractedUnit(
            unit=encoder_outputs.audio_codes.squeeze(),
            stuff_for_synth=(encoder_outputs, padding_mask)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        encoder_outputs, padding_mask = stuff_for_synth
        audio_values = \
            self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
        return audio_values[0].cpu().numpy()
