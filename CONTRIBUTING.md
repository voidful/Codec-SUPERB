# Contributing to Codec-SUPERB

We welcome contributions to Codec-SUPERB in several areas: models, datasets, and metrics. Here's how you can contribute:

## Contributing Models

1. Fork the Codec-SUPERB repository.
2. Add your model to the `SoundCodec` directory.
   - add model to `base_codec` first, create a class from `BaseCodec` and implement the abstract methods. For example:
   ```python
    from SoundCodec.base_codec import BaseCodec
    class MyCodec(BaseCodec):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # your init code here
   
        # data_item will be a dict object with
        # {
        #   'audio': {
        #       'array': resampled_waveform,
        #       'sampling_rate': sample_rate
        #   }
        # }
        def extract_unit(self, data_item):
            # your extract unit code here
            # return ExtractedUnit(
            #     unit=acoustic_token.squeeze(0).permute(1, 0),
            #     stuff_for_synth=acoustic_token
            # )
            pass
       
        def decode_unit(self, stuff_for_synth):
            # return numpy audio array with shape
            pass
   
        def synth(self, data, local_save=False):
            # your synth code here, you can treat data as a dict
            # return updated data
            # data['unit'] 
            # data['audio'] with its path or data['audio']['array']
            pass
    ```
   - add different setting of your base model in `codec` folder
4. Add tests for your model in the `tests` directory.
4. Submit a pull request with your changes. Please include a detailed description of your model and how it improves Codec-SUPERB.

## Contributing Datasets

1. Fork the Codec-SUPERB repository.
2. Add your dataset to the `datasets` directory.
   - add a python file with dataset's name, eg. ds_name.py
   - ds_name should include a function:
     ```python
        def load_data():
             # your load data code here
             # return a dataset object
             # it should include at least two key: 'audio' and 'id'
            pass
     ```
3. Add tests for your dataset in the `tests` directory.
4. Submit a pull request with your changes. Please include a detailed description of your dataset and how it improves Codec-SUPERB.

## Contributing Metrics

1. Fork the Codec-SUPERB repository.
2. Add your metric to the `metrics.py` file, following the existing format.
3. Add tests for your metric in the `tests` directory.