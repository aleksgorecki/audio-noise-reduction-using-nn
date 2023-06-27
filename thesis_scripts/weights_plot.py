from speech_denoising_wavenet.my_loss_functions import gaussian_spectrogram_weights
import numpy as np
import matplotlib.pyplot as plt

samplerate=16000

nfft = 2048
freq_bins = np.linspace(0, int(samplerate / 2), nfft // 2 + 1, dtype='float32')
center_freq = 1000
std = 3000

w = gaussian_spectrogram_weights(nfft, center_freq, std, samplerate)

fig, ax = plt.subplots()
#fig.tight_layout()
fig.set_size_inches(7, 4)
plt.plot(freq_bins, w)
plt.xlabel("częstotliwość [Hz]")
plt.ylabel("wartość wagi")
plt.savefig("/home/aleks/Desktop/weights.png", dpi=400)

