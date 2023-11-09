from audioset_download import Downloader


def load_data():
    """ Download the eval split of AudioSet. There are 20k files, taking 70GB storage. """
    d = Downloader(root_path='AudioSet', n_jobs=8, download_type='eval', copy_and_replicate=False)
    d.download(format = 'wav')

    def map_file_to_id(data):
        data['id'] = "".join(data['file'].split("/")[-2:])
        return data

    dataset = dataset.map(map_file_to_id)
    dataset = dataset.remove_columns(['label', 'file'])
    return dataset
