import os

import numpy as np
from matplotlib import pyplot as plt
import librosa.feature
import librosa


def plot_waveform(data, fs, title=None, show=True, save_path=None, fig=None, ax=None, eng=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    librosa.display.waveshow(y=data, sr=fs, ax=ax)
    ax.set(
        title=title,
    )
    if eng:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
    else:
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
    if show:
        plt.show()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path)


def plot_spectrogram(data, fs, title=None, show=True, save_path=None, fig=None, ax=None, eng=False):
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    win_len = 256
    hop_len = 128
    stft = librosa.stft(data, win_length=win_len, hop_length=hop_len)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    img = librosa.display.specshow(stft_db, sr=fs, ax=ax, x_axis='time', y_axis='linear', win_length=win_len, hop_length=hop_len)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set(
        title=title,
    )
    if eng:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
    else:
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Częstotliwość [Hz]")
    if show:
        plt.show()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path)


def plot_mel_spec(data, fs, title=None, show=True, save_path=None):
    pass


def get_feature_vector(data, fs):
    data = librosa.util.normalize(data)

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=fs))
    zero_cross = np.mean(librosa.feature.zero_crossing_rate(y=data))
    return np.array([mfcc, zero_cross])


def knn_audio():
    librosa.feature.mfcc()
    pass
