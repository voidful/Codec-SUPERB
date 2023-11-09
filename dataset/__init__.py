import dataset.general


def load_dataset(dataset_name):
    module = __import__(f"dataset.{dataset_name}", fromlist=[dataset_name])
    return module.load_data()


def load_dataset_from_local(dataset_name, download_path):
    module = __import__(f"dataset.{dataset_name}", fromlist=[dataset_name])
    return module.load_data(download_path)
