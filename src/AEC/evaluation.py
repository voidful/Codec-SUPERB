import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
# for dataloader
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from CLAP.msclap import CLAP


class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta','esc50.csv'),
    }

    def __init__(self, root, resyn_path, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()

        self.resyn_path = resyn_path
        self.targets, self.audio_paths, self.resyn_audio_path = [], [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        self.df['category'] = self.df['category'].str.replace('_',' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            resyn_path = os.path.join(self.resyn_path, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)
            self.resyn_audio_path.append(resyn_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        file_path, resyn_file_path, target = self.audio_paths[index], self.resyn_audio_path[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1,-1)
        return file_path, resyn_file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        download_url(self.url, self.root, self.filename)

        # extract file
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)

def AEC_eval(syn_path, # ref_path,
            model_type="2023",
):
    """
    Args:
        syn_path (str): Path to the synthesis audio files.
        model_type (str): Which model to use, 2022 or 2023.
    """
    # Load dataset
    root_path = "CLAP/examples/root_path" # Folder with ESC-50-master/
    dataset = ESC50(root=root_path, resyn_path=syn_path, download=True) #If download=False code assumes base_folder='ESC-50-master' in esc50_dataset.py
    prompt = 'this is the sound of '
    y = [prompt + x for x in dataset.classes]

    # Load and initialize CLAP
    clap_model = CLAP(version = model_type, use_cuda=True)

    # Computing text embeddings
    text_embeddings = clap_model.get_text_embeddings(y)

    # Computing audio embeddings
    y_preds, resyn_y_preds, cos_similarities, y_labels = [], [], [], []
    for i in tqdm(range(len(dataset))):
        x, xr, _, one_hot_target = dataset.__getitem__(i)
        audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
        resyn_audio_embeddings = clap_model.get_audio_embeddings([xr], resample=True)
        
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
        resyn_similarity = clap_model.compute_similarity(resyn_audio_embeddings, text_embeddings)
        
        cos_similarity = F.cosine_similarity(audio_embeddings, resyn_audio_embeddings)
        # print(f"cos_similarity: {cos_similarity}")

        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        resyn_y_pred = F.softmax(resyn_similarity.detach().cpu(), dim=1).numpy()
        
        y_preds.append(y_pred)
        resyn_y_preds.append(resyn_y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())
        cos_similarities.append(cos_similarity.detach().cpu())

    y_labels, y_preds, resyn_y_preds, cos_similarities = np.concatenate(y_labels, axis=0), \
                                                        np.concatenate(y_preds, axis=0), \
                                                        np.concatenate(resyn_y_preds, axis=0), \
                                                        np.concatenate(cos_similarities, axis=0)
    
    cos_similarities = np.mean(cos_similarities)

    accuracy_groud_truth = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    accuracy_resync = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(resyn_y_preds, axis=1))

    return accuracy_groud_truth, accuracy_resync, cos_similarities

if __name__=="__main__":
    parser = ArgumentParser(description="Audio Event Classification")
    parser.add_argument(
        "--syn_path",
        default="/mnt/sda/codec-superb-slt/CLAP/examples/root_path/ESC-50-master/audio/",
        help="path to the generate data"
    )
    parser.add_argument(
        "--model_type",
        default="2023",
        help="either 2022 or 2023",
    )

    args = parser.parse_args()

    accuracy_groud_truth, accuracy_resync, cos_similarities = AEC_eval(args.syn_path)

    print(f"Acc_ground_truth: {accuracy_groud_truth * 100:.2f}%")
    print(f"Acc_resync_audio: {accuracy_resync * 100:.2f}%")
    print(f"Cos_similarity: {cos_similarities * 100:.2f}%")
