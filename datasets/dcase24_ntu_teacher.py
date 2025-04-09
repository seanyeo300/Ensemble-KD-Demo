import pandas as pd
import os
from sklearn import preprocessing
from torch.utils.data import Dataset as TorchDataset
import torch
import torchaudio
import torch.nn.functional as F
from torch.hub import download_url_to_file
import numpy as np
import librosa
from scipy.signal import convolve
import pathlib
import h5py

dataset_dir = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development" # Alibaba
assert dataset_dir is not None, "Specify 'TAU Urban Acoustic Scenes 2022 Mobile dataset' location in variable " \
                                "'dataset_dir'. The dataset can be downloaded from this URL:" \
                                " https://zenodo.org/record/6337421"

dataset_config = {
    "dataset_name": "tau24",
    "meta_csv": os.path.join(dataset_dir, "meta.csv"),
    "split_path": "split_setup",
    "split_url": "https://github.com/CPJKU/dcase2024_task1_baseline/releases/download/files/",
    "test_split_csv": "test.csv",
    "dirs_path": os.path.join("dataset", "dirs"),
    "eval_dir": os.path.join(dataset_dir), 
    "eval_meta_csv": os.path.join(dataset_dir, "meta.csv"),                                                         #Generates logit for PaSST model with corret index
    
    # "logits_file": os.path.join("predictions","ensemble", "sub5_single_PaSST.pt")                                 
    "logits_file": os.path.join("predictions","ensemble", "sub5_ensemble_6_PaSST.pt")                               # Not used for training, just prevents missing logit error
    
    
    # "eval_dir": os.path.join(dataset_dir, "TAU-urban-acoustic-scenes-2024-mobile-evaluation"),                    # You won't be using this for PaSST
    # "eval_meta_csv": os.path.join(dataset_dir,  "TAU-urban-acoustic-scenes-2024-mobile-evaluation", "meta.csv")
}
class DirDataset(TorchDataset):
    """
   Augments Waveforms with a Device Impulse Response (DIR) from h5 file
    """

    def __init__(self, ds, hmic, dir_p):
        self.ds = ds
        self.hmic = hmic
        self.dir_p = dir_p

    def __getitem__(self, index):
        x, file, label, device, city, logits = self.ds[index]
        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]
        self.device = device

        # New devices are created using device A + impulse function + DRC
        if device == 'a' and self.dir_p > np.random.rand():
            all_keys = list(self.hmic.keys())
            # Choose a random key
            dir_key = np.random.choice(all_keys)
            # print(f"Selected DIR key: {dir_key}")
            
            # Retrieve the corresponding DIR using the key
            dir = torch.from_numpy(self.hmic.get(dir_key)[()])
            # # choose a DIR at random
            # dir_idx = str(int(np.random.randint(0, len(self.hmic))))
            # dir = torch.from_numpy(self.hmic.get(dir_idx)[()])  
            # get audio file with 'new' mic response
            x = convolve(x, dir, 'full')[:, :x.shape[1]]
            x = torch.from_numpy(x)
        return x, file, label, device, city, logits

    def __len__(self):
        return len(self.ds)  

class DIRAugmentDataset(TorchDataset):
    """
   Augments Waveforms with a Device Impulse Response (DIR) from h5
    """

    def __init__(self, ds, dirs, prob):
        self.ds = ds
        self.dirs = dirs
        self.prob = prob

    def __getitem__(self, index):
        x, file, label, device, city, logits = self.ds[index]

        fsplit = file.rsplit("-", 1)
        device = fsplit[1][:-4]

        if device == 'a' and torch.rand(1) < self.prob:
            # choose a DIR at random
            dir_idx = int(np.random.randint(0, len(self.dirs)))
            dir = self.dirs[dir_idx]

            x = convolve(x, dir, 'full')[:, :x.shape[1]]
            x = torch.from_numpy(x)
        return x, file, label, device, city, logits

    def __len__(self):
        return len(self.ds)
class AddLogitsDataset(TorchDataset):
    """A dataset that loads and adds teacher logits to audio samples.
    """

    def __init__(self, dataset, map_indices, logits_file, temperature=2):
        """
        @param dataset: dataset to load data from
        @param map_indices: used to get correct indices in list of logits
        @param logits_file: logits file to load the teacher logits from
        @param temperature: used in Knowledge Distillation, change distribution of predictions
        return: x, file name, label, device, city, logits
        """
        self.dataset = dataset
        if not os.path.isfile(logits_file):
            print("Verify existence of teacher predictions.")
            raise SystemExit
        logits = torch.load(logits_file).float()
        self.logits = logits
        # self.logits = F.log_softmax(logits / temperature, dim=-1)
        self.map_indices = map_indices

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[index]
        return x, file, label, device, city, self.logits[self.map_indices[index]]

    def __len__(self):
        return len(self.dataset)
    
class BasicDCASE24Dataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads data from files
    """

    def __init__(self, meta_csv):
        """
        @param meta_csv: meta csv file for the dataset
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(dataset_dir, self.files[index]))
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)


class SimpleSelectionDataset(TorchDataset):
    """A dataset that selects a subsample from a dataset based on a set of sample ids.
        Supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __init__(self, dataset, available_indices):
        """
        @param dataset: dataset to load data from
        @param available_indices: available indices of samples for different splits
        return: waveform, file, label, device, city
        """
        self.available_indices = available_indices
        self.dataset = dataset

    def __getitem__(self, index):
        x, file, label, device, city = self.dataset[self.available_indices[index]]
        return x, file, label, device, city

    def __len__(self):
        return len(self.available_indices)


class RollDataset(TorchDataset):
    """A dataset implementing time rolling of waveforms.
    """

    def __init__(self, dataset: TorchDataset, shift_range: int, axis=1):
        """
        @param dataset: dataset to load data from
        @param shift_range: maximum shift range
        return: waveform, file, label, device, city
        """
        self.dataset = dataset
        self.shift_range = shift_range
        self.axis = axis

    def __getitem__(self, index):
        x, file, label, device, city, logits = self.dataset[index]
        sf = int(np.random.random_integers(-self.shift_range, self.shift_range))
        return x.roll(sf, self.axis), file, label, device, city, logits

    def __len__(self):
        return len(self.dataset)

def load_dirs(dirs_path, resample_rate):
    all_paths = [path for path in pathlib.Path(os.path.expanduser(dirs_path)).rglob('*.wav')]
    all_paths = sorted(all_paths)
    all_paths_name = [str(p).rsplit("/", 1)[-1] for p in all_paths]

    print("Augment waveforms with the following device impulse responses:")
    for i in range(len(all_paths_name)):
        print(i, ": ", all_paths_name[i])

    def process_func(dir_file):
        sig, _ = librosa.load(dir_file, sr=resample_rate, mono=True)
        sig = torch.from_numpy(sig[np.newaxis])
        return sig

    return [process_func(p) for p in all_paths]

def get_training_set(split=100, roll=False, dir_prob=0,resample_rate=44100):
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = get_base_training_set(dataset_config['meta_csv'], subset_split_file)
    if dir_prob > 0:
        ds = DIRAugmentDataset(ds, load_dirs(dataset_config['dirs_path'], resample_rate), dir_prob)
    if roll:
        ds = RollDataset(ds, shift_range=roll)
    return ds


def get_base_training_set(meta_csv, train_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    train_subset_indices = list(meta[meta['filename'].isin(train_files)].index) # these indices pull from the meta.csv index, not the train_subset csv
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv),
                                train_subset_indices)
    ds = AddLogitsDataset(ds, train_subset_indices, dataset_config['logits_file'])
    return ds


def get_test_set():
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = get_base_test_set(dataset_config['meta_csv'], test_split_csv)
    return ds


def get_base_test_set(meta_csv, test_files_csv):
    meta = pd.read_csv(meta_csv, sep="\t")
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataset(meta_csv), test_indices)
    return ds


class BasicDCASE24EvalDataset(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, eval_dir):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.eval_dir = eval_dir

    def __getitem__(self, index):
        sig, _ = torchaudio.load(os.path.join(self.eval_dir, self.files[index]))
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)


def get_eval_set():
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = get_base_eval_set(dataset_config['eval_meta_csv'], dataset_config['eval_dir'])
    return ds


def get_base_eval_set(meta_csv, eval_dir):
    ds = BasicDCASE24EvalDataset(meta_csv, eval_dir)
    return ds


############################################ implementation of dataloading from h5  ########################################

class BasicDCASE24Dataseth5(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads wav data from h5 files
    """

    def __init__(self, meta_csv, hf_in):
        """
        @param meta_csv: meta csv file for the dataset
        return: waveform, file, label, device and city
        """
        df = pd.read_csv(meta_csv, sep="\t")
        le = preprocessing.LabelEncoder()
        self.labels = torch.from_numpy(le.fit_transform(df[['scene_label']].values.reshape(-1)))
        self.devices = le.fit_transform(df[['source_label']].values.reshape(-1))
        self.cities = le.fit_transform(df['identifier'].apply(lambda loc: loc.split("-")[0]).values.reshape(-1))
        self.files = df[['filename']].values.reshape(-1)
        self.hf_in = hf_in

    def __getitem__(self, index):
        mel_sig_ds = self.files[index][5:-4]
        sig = torch.from_numpy(self.hf_in.get(mel_sig_ds)[()])  
        return sig, self.files[index], self.labels[index], self.devices[index], self.cities[index]

    def __len__(self):
        return len(self.files)
    
def ntu_get_training_set_dir(split=100, dir_prob = False, hf_in=None, hmic_in=None): # this variant is for DIR augmentation
    assert str(split) in ("5", "10", "25", "50", "100"), "Parameters 'split' must be in [5, 10, 25, 50, 100]"
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    subset_fname = f"split{split}.csv"
    subset_split_file = os.path.join(dataset_config['split_path'], subset_fname)
    if not os.path.isfile(subset_split_file):
        # download split{x}.csv (file containing all audio snippets for respective development-train split)
        subset_csv_url = dataset_config['split_url'] + subset_fname
        print(f"Downloading file: {subset_fname}")
        download_url_to_file(subset_csv_url, subset_split_file)
    ds = ntu_get_base_training_set(dataset_config['meta_csv'], subset_split_file, hf_in)
    if dir_prob:
        ds = DirDataset(ds, hmic_in, dir_prob)
    return ds


def ntu_get_base_training_set(meta_csv, train_files_csv, hf_in): # this variant does not use DIR augmentation
    meta = pd.read_csv(meta_csv, sep="\t")
    train_files = pd.read_csv(train_files_csv, sep='\t')['filename'].values.reshape(-1)
    train_subset_indices = list(meta[meta['filename'].isin(train_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataseth5(meta_csv, hf_in),
                                train_subset_indices)
    ds = AddLogitsDataset(ds, train_subset_indices, dataset_config['logits_file'])
    return ds

def ntu_get_test_set(hf_in = None):
    os.makedirs(dataset_config['split_path'], exist_ok=True)
    test_split_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
    if not os.path.isfile(test_split_csv):
        # download test.csv (file containing all audio snippets for development-test split)
        test_csv_url = dataset_config['split_url'] + dataset_config['test_split_csv']
        print(f"Downloading file: {dataset_config['test_split_csv']}")
        download_url_to_file(test_csv_url, test_split_csv)
    ds = ntu_get_base_test_set(dataset_config['meta_csv'], test_split_csv, hf_in)
    return ds

def ntu_get_base_test_set(meta_csv, test_files_csv, hf_in):
    meta = pd.read_csv(meta_csv, sep="\t")
    test_files = pd.read_csv(test_files_csv, sep='\t')['filename'].values.reshape(-1)
    test_indices = list(meta[meta['filename'].isin(test_files)].index)
    ds = SimpleSelectionDataset(BasicDCASE24Dataseth5(meta_csv, hf_in), test_indices)
    return ds

class BasicDCASE24EvalDataseth5(TorchDataset):
    """
    Basic DCASE'24 Dataset: loads eval data from files
    """

    def __init__(self, meta_csv, hf_in):
        """
        @param meta_csv: meta csv file for the dataset
        @param eval_dir: directory containing evaluation set
        return: waveform, file
        """
        df = pd.read_csv(meta_csv, sep="\t")
        self.files = df[['filename']].values.reshape(-1)
        self.hf_in = hf_in
        #self.eval_dir = eval_dir

    def __getitem__(self, index):
        mel_sig_ds = self.files[index][5:-4]
        # print(mel_sig_ds)
        # sig = torch.from_numpy(self.hf_in.get(mel_sig_ds)[()])          
        data = self.hf_in.get(mel_sig_ds)
        if data is None:
            raise ValueError(f"Dataset {mel_sig_ds} not found in HDF5 file.")
        sig = torch.from_numpy(data[()])
        return sig, self.files[index]

    def __len__(self):
        return len(self.files)
    
def ntu_get_eval_set(hf_in):
    assert os.path.exists(dataset_config['eval_dir']), f"No such folder: {dataset_config['eval_dir']}"
    ds = ntu_get_base_eval_set(dataset_config['eval_meta_csv'], hf_in)
    return ds

def ntu_get_base_eval_set(meta_csv, hf_in):
    ds = BasicDCASE24EvalDataseth5(meta_csv, hf_in)
    return ds

def open_h5(h5_file):
    hf_in =h5py.File(h5_file, 'r')
    return hf_in

def close_h5(hf_in):
    hf_in.close()    