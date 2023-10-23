from transformers import AutoModel, AutoProcessor
from codec.general import save_audio


class Codec:
    def __init__(self, pretrained_model_name="facebook/encodec_24khz"):
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.sampling_rate = self.processor.sampling_rate

    def synth(self, data):
        audio_sample = data["audio"]["array"]
        inputs = self.processor(raw_audio=audio_sample, sampling_rate=self.sampling_rate, return_tensors="pt")
        encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"])
        audio_values = \
        self.model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
        audio_path = f"dummy/{data['id']}.wav"
        save_audio(audio_values[0], audio_path, self.sampling_rate)
        data['audio'] = audio_path
        return data
