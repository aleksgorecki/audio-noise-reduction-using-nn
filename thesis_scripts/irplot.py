import os
from custom_model_evaluation import (
    get_best_checkpoint,
    evaluate_on_testset,
    prepare_batch,
)
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet.main import load_config
import numpy as np
import librosa
import matplotlib.pyplot as plt
from datasets.visual import plot_spectrogram
import soundfile as sf


def plot_waveform(
    data, fs, title=None, show=True, save_path=None, fig=None, ax=None, eng=False
):
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


ir_path = "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav"
ir = librosa.core.load(ir_path, sr=16000)[0]

ir = librosa.util.normalize(ir)
plot_waveform(data=ir, fs=16000)
plt.show()
