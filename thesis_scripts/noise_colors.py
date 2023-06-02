from datasets.noise_generators import *
import matplotlib.pyplot as plt


# blue = blue_noise(1000)
white = white_noise(1000000)
pink = pink_noise(1000000)
blue = blue_noise(1000000)
# plt.psd(white, Fs=20000, color="gray")
# plt.psd(pink, Fs=20000, color="pink")
# plt.psd(blue, Fs=20000, color="blue")
# plt.legend(["szum biały", "szum różowy", "szum niebieski"])
plt.magnitude_spectrum(white, Fs=16000, color="gray", alpha=0.7)
plt.magnitude_spectrum(pink, Fs=16000, color="pink", alpha=0.7)
plt.magnitude_spectrum(blue, Fs=16000, color="blue", alpha=0.7)
plt.ylim((0, 0.006))
plt.xlim(0, 8000)
plt.grid(False)
plt.legend(["szum biały", "szum różowy", "szum niebieski"])
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Energia sygnału")
plt.show()
