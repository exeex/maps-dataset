from torch.utils.data import Dataset
import os
import numpy as np
from utils import pad_along_axis
from scipy.signal import resample
from glob import glob
import mimi


class MAPS_Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_folder='maps', transform=None):
        self.files = [y.replace('.wav', '') for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.wav'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.files[idx] + ".wav")

        mp3_file = os.path.join(self.files[idx] + ".mp3")

        mid_file = os.path.join(self.files[idx] + ".mid")

        txt_file = os.path.join(self.files[idx] + ".txt")

        # stft / npy
        # npy = np.load(npy_file)
        # npy = pad_along_axis(npy, target_length=591, axis=2)

        # roll / npz
        mid = mimi.MidiFile(mid_file)
        npz = mid.get_npz()
        npz = npz['data'][0:8, 39:97, :2630]  # instrument nb
        npz = resample(npz, 197, axis=2)
        npz = npz.astype('float32')

        # sample = {'spectrogram': npy, 'piano_roll': npz}
        sample = {'piano_roll': npz}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_mid(self, idx):
        mid_file = os.path.join(self.files[idx] + ".mid")

        mid = mimi.MidiFile(mid_file)

        return mid


class MAPS_Subset(MAPS_Data):

    def __init__(self, sub_str, data_folder='maps', transform=None):
        self.files = [y.replace('.wav', '') for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.wav'))
                      if y.find(sub_str)!=-1]
        self.transform = transform


if __name__ == "__main__":
    d = MAPS_Subset("ISOL")
    print(d.files)