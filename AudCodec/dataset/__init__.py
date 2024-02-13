def load_dataset(dataset_name):
    module = __import__(f"dataset.{dataset_name}", fromlist=[dataset_name])
    return module.load_data()
