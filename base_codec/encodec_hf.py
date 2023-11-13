from transformers import AutoModel, AutoProcessor
from codec.general import save_audio
import torch

class BaseCodec:
    def __init__(self):
        self.config()
        self.model = AutoModel.from_pretrained(self.pretrained_model_name).cuda()
        self.processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
        self.sampling_rate = self.processor.sampling_rate

    def config(self):
        self.pretrained_model_name = "facebook/encodec_24khz"
    
    @torch.no_grad()
    def synth(self, data):
        encoder_outputs, padding_mask = self.extract_unit(data, return_unit_only=False)
        audio_values = \
            self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
        audio_path = f"dummy_{self.pretrained_model_name}/{data['id']}.wav"
        save_audio(audio_values[0].cpu(), audio_path, self.sampling_rate)
        data['audio'] = audio_path
        return data
    
    @torch.no_grad()
    def extract_unit(self, data, return_unit_only=True):
        audio_sample = data["audio"]["array"]
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
        encoder_outputs = self.model.encode(inputs["input_values"].cuda(), inputs["padding_mask"].cuda())
        
        if return_unit_only:
            return encoder_outputs.audio_codes.squeeze()
        return encoder_outputs, inputs["padding_mask"]
