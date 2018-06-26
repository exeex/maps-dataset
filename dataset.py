from torch.utils.data import Dataset
import os
import numpy as np
from utils import pad_along_axis
from scipy.signal import resample
from glob import glob
import mimi
from preprosesser import _preprosess as preprosess
from torch.utils.data import DataLoader


class MAPS_Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_folder='maps', transform=None):
        self.files = [y.replace('.wav', '') for x in os.walk(data_folder) for y in glob(os.path.join(x[0], '*.wav'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_file = os.path.join(self.files[idx] + ".wav")

        mid_file = os.path.join(self.files[idx] + ".mid")

        txt_file = os.path.join(self.files[idx] + ".txt")

        # stft / npy
        npy = preprosess(wav_file)
        npy = pad_along_axis(npy, target_length=500, axis=2)

        # roll / npz
        mid = self.get_mid(idx)
        npz = mid.get_npz()
        npz = npz['data'][0:8, 39:97, :]  # instrument nb
        npz = npz.astype('float32')
        # npz = resample(npz, 197, axis=2)

        # sample = {'spectrogram': npy, 'piano_roll': npz}
        sample = {'piano_roll': npz, 'spectrogram': npy}

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
                      if y.find(sub_str) != -1]
        self.transform = transform


if __name__ == "__main__":
    d = MAPS_Subset("ISOL")

    print(d[0]['piano_roll'].shape)


    #TODO: tick per beat issue

    import time

    # dl = DataLoader(d, batch_size=4, shuffle=True, num_workers=8)
    # start = time.time()
    #
    # c = 0
    # for data in dl:
    #     print(data['spectrogram'].shape)
    #     c+=1
    #     if c > 32:
    #         break
    # end = time.time()
    # print(end - start)
