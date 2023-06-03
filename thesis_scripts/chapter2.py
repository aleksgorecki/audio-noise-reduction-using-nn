import librosa
import numpy as np
import matplotlib.pyplot as plt
from datasets.visual import *


cmajor = librosa.core.load("CMajor.wav", sr=16000)[0]
plot_waveform(cmajor, fs=16000)

plot_spectrogram(cmajor, fs=16000)