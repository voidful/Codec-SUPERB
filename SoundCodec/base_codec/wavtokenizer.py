import nlp2
import numpy
import torch

from SoundCodec.base_codec.general import save_audio, ExtractedUnit, BaseCodec, BatchExtractedUnit


class WavTokenizerBaseCodec(BaseCodec):
    def __init__(self):
        super().__init__()

    def config(self):
        self._setup_config_and_model()

    def _setup_config_and_model(self):
        try:
            from wavtokenizer.decoder.pretrained import WavTokenizer
        except ImportError:
            raise Exception(
                "Please install wavtokenizer first. pip install git+https://github.com/voidful/WavTokenizer.git"
            )

        self._download_resources()
        self.model = WavTokenizer.from_pretrained0802(self.config_path, self.ckpt_path)
        self.model.eval()
        self.model = self.model.to(self.device)
        if self.sampling_rate is None:
            self.sampling_rate = 24000

    def _download_resources(self):
        import os
        from huggingface_hub import hf_hub_download
        
        # Determine config path
        if hasattr(self, "config_repo") and hasattr(self, "config_filename"):
            self.config_path = hf_hub_download(repo_id=self.config_repo, filename=self.config_filename, local_dir="external_codecs/wavtokenizer_model")
        elif hasattr(self, "config_url"):
            nlp2.download_file(self.config_url, 'external_codecs/wavtokenizer_model')
            self.config_path = f"external_codecs/wavtokenizer_model/{os.path.basename(self.config_url)}"
        else:
            # Default
            nlp2.download_file(
                'https://raw.githubusercontent.com/voidful/WavTokenizer/master/wavtokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml',
                'external_codecs/wavtokenizer_model'
            )
            self.config_path = "external_codecs/wavtokenizer_model/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"

        # Patch config.yaml class_paths
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                import yaml
                config_data = yaml.safe_load(f)

            patched = False

            def patch_dict(d):
                nonlocal patched
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k == "class_path" and isinstance(v, str):
                            if v.startswith("decoder.") or v.startswith("encoder."):
                                d[k] = "wavtokenizer." + v
                                patched = True
                        else:
                            patch_dict(v)
                elif isinstance(d, list):
                    for item in d:
                        patch_dict(item)

            patch_dict(config_data)
            if patched:
                with open(self.config_path, "w") as f:
                    yaml.dump(config_data, f)
                print(f"Patched config.yaml class_paths in: {self.config_path}")

        # Determine ckpt path
        if hasattr(self, "ckpt_repo") and hasattr(self, "ckpt_filename"):
            self.ckpt_path = hf_hub_download(repo_id=self.ckpt_repo, filename=self.ckpt_filename, local_dir="external_codecs/wavtokenizer_model")
        elif hasattr(self, "ckpt_url"):
            nlp2.download_file(self.ckpt_url, 'external_codecs/wavtokenizer_model')
            self.ckpt_path = f"external_codecs/wavtokenizer_model/{os.path.basename(self.ckpt_url)}"
        else:
            # Default
            self.ckpt_path = hf_hub_download(repo_id="novateur/WavTokenizer-large-unify-40token", filename="wavtokenizer_large_unify_600_24k.ckpt", local_dir="external_codecs/wavtokenizer_model")

    @torch.no_grad()
    def extract_unit(self, data):
        wav = torch.tensor(numpy.array([data["audio"]['array']]), dtype=torch.float32).to(self.device)
        bandwidth_id = torch.tensor([0]).to(self.device)
        features, discrete_code = self.model.encode_infer(wav, bandwidth_id=bandwidth_id)
        return ExtractedUnit(
            unit=discrete_code[0],
            stuff_for_synth=(features, bandwidth_id)
        )

    @torch.no_grad()
    def decode_unit(self, stuff_for_synth):
        features, bandwidth_id = stuff_for_synth
        audio_out = self.model.decode(features.to(self.device), bandwidth_id=bandwidth_id.to(self.device))
        wav = audio_out.detach().cpu().numpy()
        return wav

    @torch.no_grad()
    def batch_extract_unit(self, data_list):
        """Batch extraction for WavTokenizer."""
        if len(data_list) == 1:
            # Single item, use regular method
            extracted_unit = self.extract_unit(data_list[0])
            return BatchExtractedUnit(
                units=[extracted_unit.unit],
                stuff_for_synth=[extracted_unit.stuff_for_synth],
                batch_size=1
            )
        
        # Prepare batch data
        wav_list = []
        
        for data in data_list:
            wav = torch.tensor(numpy.array(data["audio"]['array']), dtype=torch.float32)
            wav_list.append(wav)
        
        # Pad all waveforms to the same length
        max_length = max(wav.shape[0] for wav in wav_list)
        padded_wavs = []
        for wav in wav_list:
            if wav.shape[0] < max_length:
                padding = max_length - wav.shape[0]
                wav = torch.nn.functional.pad(wav, (0, padding))
            padded_wavs.append(wav)
        
        # Stack into batch tensor [B, T]
        batch_wav = torch.stack(padded_wavs, dim=0).to(self.device)
        
        # Create batch bandwidth_id
        batch_bandwidth_id = torch.zeros(len(data_list), dtype=torch.long).to(self.device)
        
        # Encode the entire batch at once
        with torch.no_grad():
            batch_features, batch_discrete_code = self.model.encode_infer(batch_wav, bandwidth_id=batch_bandwidth_id)
        
        # Process results for each item in the batch
        units = []
        stuff_for_synth = []
        
        for i in range(len(data_list)):
            # Extract features and codes for this item
            item_features = batch_features[i:i+1] if batch_features.dim() > 2 else batch_features
            item_discrete_code = batch_discrete_code[i] if batch_discrete_code.dim() > 1 else batch_discrete_code
            item_bandwidth_id = batch_bandwidth_id[i:i+1]
            
            units.append(item_discrete_code)
            stuff_for_synth.append((item_features, item_bandwidth_id))
        
        return BatchExtractedUnit(
            units=units,
            stuff_for_synth=stuff_for_synth,
            batch_size=len(data_list)
        )

    @torch.no_grad()
    def batch_decode_unit(self, batch_extracted_unit):
        """Batch decoding for WavTokenizer."""
        if batch_extracted_unit.batch_size == 1:
            # Single item, use regular method
            return [self.decode_unit(batch_extracted_unit.stuff_for_synth[0])]
        
        # Collect all features and bandwidth_ids for batch processing
        all_features = []
        all_bandwidth_ids = []
        
        for features, bandwidth_id in batch_extracted_unit.stuff_for_synth:
            all_features.append(features)
            all_bandwidth_ids.append(bandwidth_id)
        
        # Stack for batch processing
        batch_features = torch.cat(all_features, dim=0).to(self.device)
        batch_bandwidth_id = torch.cat(all_bandwidth_ids, dim=0).to(self.device)
        
        # Decode the entire batch at once
        with torch.no_grad():
            batch_audio_out = self.model.decode(batch_features, bandwidth_id=batch_bandwidth_id)
        
        # Split results for each item in the batch
        audio_values = []
        for i in range(batch_extracted_unit.batch_size):
            wav = batch_audio_out[i].detach().cpu().numpy()
            audio_values.append(wav)
        
        return audio_values


