import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
from concurrent.futures import ProcessPoolExecutor
import sys


if __name__ == "__main__":
    mp3_folder = "data_c22/mp3"
    npy_folder = "data_c22/npy"
    if not os.path.exists(mp3_folder):
        os.mkdir(mp3_folder)
    if not os.path.exists(npy_folder):
        os.mkdir(npy_folder)

    files = os.listdir(mp3_folder)


def preprosess(fortest=False):
    infile_paths = [os.path.join(mp3_folder, file) for file in files]
    outfile_paths = [os.path.join(npy_folder, file + ".npy") for file in files]

    if fortest:
        data = _preprosess(infile_paths[0])
        return data

    with ProcessPoolExecutor(max_workers=10) as executor:
        for idx, data in enumerate(executor.map(_preprosess, infile_paths)):
            outfile = outfile_paths[idx]
            np.save(outfile, data)


def _preprosess(file):
    # print(file, file=sys.stderr)
    try:
        aud, sr = librosa.load(file, sr=22050)
        # print(sr)
        # D = librosa.cqt(aud,sr=sr, n_bins=720, bins_per_octave=120, hop_length=1024)
        D = librosa.stft(aud, n_fft=4096)
        # D = pad_along_axis(D,2631,axis=1)
        M, P = librosa.magphase(D[:400, :])

        data = M, np.real(D[:400, :]), np.imag(D[:400, :])
        data = np.stack(data, axis=0)
        # print(data.shape, file=sys.stderr)
        return data
        # data = librosa.amplitude_to_db(M)#.astype("float32")
    except Exception as err:
        print("load file %s fail," % file, err)
        return None


if __name__ == "__main__":
    gg = preprosess(fortest=False)

    # a = np.identity(5)
    # b = pad_along_axis(a, 7, axis=1)
    # print(a,a.shape)
    # print(b,b.shape)
