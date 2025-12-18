from datasets import load_dataset as hf_load_dataset

def load_dataset(dataset_name):
    try:
        module = __import__(f"SoundCodec.dataset.{dataset_name}", fromlist=[dataset_name])
        return module.load_data()
    except ImportError:
        # Fallback to loading from Hugging Face Hub
        ds = hf_load_dataset(dataset_name)
        if isinstance(ds, dict):
            from datasets import concatenate_datasets
            all_ds = []
            for split_name, split_ds in ds.items():
                # Add category column if it doesn't exist
                if 'category' not in split_ds.column_names:
                    split_ds = split_ds.add_column('category', [split_name] * len(split_ds))
                all_ds.append(split_ds)
            return concatenate_datasets(all_ds).shuffle(seed=42)
        return ds
