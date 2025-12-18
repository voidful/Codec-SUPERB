from datasets import load_dataset as hf_load_dataset

def load_dataset(dataset_name):
    try:
        module = __import__(f"SoundCodec.dataset.{dataset_name}", fromlist=[dataset_name])
        return module.load_data()
    except ImportError:
        # Fallback to loading from Hugging Face Hub
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            if "test" in ds:
                return ds["test"]
            if "validation" in ds:
                return ds["validation"]
            if "train" in ds:
                return ds["train"]
            # return the first split if none of the above are found
            return ds[list(ds.keys())[0]]
        return ds
