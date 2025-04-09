import h5py
import pandas as pd
import torchaudio
import tqdm
import os


# Modify paths before use
dataset_dir = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development"
meta_csv = r"D:\Sean\DCASE\datasets\Extract_to_Folder\TAU-urban-acoustic-scenes-2022-mobile-development\meta.csv"

# to create audio samples h5 file
df = pd.read_csv(meta_csv, sep="\t")
# train_files = pd.read_csv(train_files_csv, sep="\t")['filename'].values.reshape(-1)
files = df['filename'].values.reshape(-1)
transform = torchaudio.transforms.Resample(new_freq = 44100)

hf = h5py.File('h5py_audio_wav', 'w')
for file_idx in tqdm(range(len(files))):
    sig, _ = torchaudio.load(os.path.join(dataset_dir, files[file_idx]))
    output_str = files[file_idx][5:-4]
    print(output_str)
    #with h5py.File(output_str, 'w') as hf:
    sig = transform(sig)
    hf.create_dataset(output_str, data = sig)

hf.close()