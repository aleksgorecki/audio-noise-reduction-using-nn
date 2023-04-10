import tensorflow as tf
import numpy as np


def gaussian_spectrogram_weights(num_freq_bins, center_freq_hz, samplerate):
    freq_std = 500
    freq_bins = np.linspace(0, int(samplerate/2), num_freq_bins)
    weights = np.exp(-(freq_bins - center_freq_hz) ** 2 / (2 * freq_std ** 2))
    weights /= np.max(weights)
    return weights


def spectrogram_loss(y_true, y_pred):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=512, frame_step=256, fft_length=512))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=512, frame_step=256, fft_length=512))

    loss = tf.reduce_mean(tf.abs(spec_true - spec_pred))
    return loss


def weighted_spectrogram_loss(y_true, y_pred, weights):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=512, frame_step=256, fft_length=512))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=512, frame_step=256, fft_length=512))

    loss = tf.reduce_mean(tf.abs(spec_true * weights - spec_pred * weights))
    return loss


def mel_spectrogram_loss(y_true, y_pred):
    n_mels = 128
    sr = 16000
    nfft = 512

    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=512, frame_step=256, fft_length=nfft))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=512, frame_step=256, fft_length=nfft))

    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=nfft // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=0.0,
        upper_edge_hertz=nfft // 2
    )
    mel_spec_true = tf.matmul(tf.square(spec_true), mel_filterbank)
    mel_spec_pred = tf.matmul(tf.square(spec_pred), mel_filterbank)

    loss = tf.reduce_mean(tf.abs(mel_spec_true - mel_spec_pred))
    return loss


def phase_spectrum_loss(y_true, y_pred):
    spec_true = tf.signal.stft(y_true, frame_length=1024, frame_step=512)
    spec_pred = tf.signal.stft(y_pred, frame_length=1024, frame_step=512)

    phase_true = tf.math.angle(spec_true)
    phase_pred = tf.math.angle(spec_pred)

    loss = tf.reduce_mean(tf.abs(phase_true - phase_pred))
    return loss


def spectral_convergence_loss(y_true, y_pred):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=1024, frame_step=512))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=1024, frame_step=512))

    spec_conv = tf.norm(spec_true - spec_pred) / tf.norm(spec_true)

    return spec_conv


def spectral_loss(y_true, y_pred):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=1024, frame_step=512))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=1024, frame_step=512))

    return tf.reduce_mean(tf.abs(spec_true - spec_pred))
