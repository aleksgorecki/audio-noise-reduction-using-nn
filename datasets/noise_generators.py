import numpy as np


def white_noise(length):
    return np.random.randn(length)


def gaussian_noise(length):
    return np.random.normal(size=length)


def pink_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = 1 / np.where(f == 0, float("inf"), np.sqrt(f))
    S = S / np.sqrt(np.mean(S**2))
    return np.fft.irfft(white * S, length)


def red_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = 1 / np.where(f == 0, float("inf"), f)
    S = S / np.sqrt(np.mean(S**2))
    return np.fft.irfft(white * S, length)


def blue_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = np.sqrt(f)
    S = S / np.sqrt(np.mean(S**2))
    return np.fft.irfft(white * S, length)


def violet_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = f
    S = S / np.sqrt(np.mean(S**2))
    return np.fft.irfft(white * S, length)
