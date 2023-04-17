import tensorflow as tf
import numpy as np


def gaussian_spectrogram_weights(nfft, center_freq_hz, samplerate):
    freq_std = 500
    freq_bins = np.linspace(0, int(samplerate/2), nfft//2 + 1, dtype='float32')
    weights = np.exp(-(freq_bins - center_freq_hz) ** 2 / (2 * freq_std ** 2), dtype='float32')
    weights /= np.max(weights)
    return weights


WEIGHTS = gaussian_spectrogram_weights(1024, 2000, 16000)


def spectrogram_loss(y_true, y_pred, frame_length, frame_step, fft_length):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))

    loss = tf.reduce_mean(tf.abs(spec_true - spec_pred))
    return loss


def weighted_spectrogram_loss(y_true, y_pred, weights, frame_length, frame_step, fft_length):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))

    loss = tf.reduce_mean(tf.abs((spec_true  - spec_pred) * tf.constant(weights)))
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


def spectral_convergence_loss(y_true, y_pred, frame_length, frame_step, fft_length):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length))

    spec_conv = tf.norm(spec_true - spec_pred) / tf.norm(spec_true)

    return spec_conv


def spectral_loss(y_true, y_pred):
    spec_true = tf.abs(tf.signal.stft(y_true, frame_length=1024, frame_step=512))
    spec_pred = tf.abs(tf.signal.stft(y_pred, frame_length=1024, frame_step=512))

    return tf.reduce_mean(tf.abs(spec_true - spec_pred))


def tf_rms_loss(y_true, y_pred):
    rms_true = tf.sqrt(tf.reduce_mean(tf.square(y_true), axis=-1))
    rms_pred = tf.sqrt(tf.reduce_mean(tf.square(y_pred), axis=-1))
    return tf.abs(rms_true - rms_pred)

def segmented_rms_loss(y_true, y_pred):

    seg_true = tf.split(y_true, 10)
    seg_pred = tf.split(y_pred, 10)

    rms_true = tf.sqrt(tf.reduce_mean(tf.square(seg_true), axis=1))
    rms_pred = tf.sqrt(tf.reduce_mean(tf.square(seg_pred), axis=1))
    return tf.reduce_mean(tf.abs(rms_true - rms_pred), axis=-1)
