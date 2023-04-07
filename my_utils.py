import librosa
from scipy.signal import wiener
import numpy as np
import os
import soundfile as sf
import pathlib
import csv
from typing import List
from matplotlib import pyplot as plt
import os
import random
import pathlib


def apply_wiener_filter_to_file(input_file: str, output_file: str) -> None:
    audio_tuple = librosa.core.load(path=input_file, sr=None)
    filtered = wiener(audio_tuple[0])
    sf.write(output_file, data=filtered, format='WAV', samplerate=int(audio_tuple[1]))


def apply_wiener_filter_to_dir(input_dir: str, output_dir: str) -> None:
    files = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for af in files:
        output_name = pathlib.Path(af).stem + '.wav'
        apply_wiener_filter_to_file(
            f"{input_dir}/{af}", output_file=f"{output_dir}/{output_name}")


def plot_wavenet_history(history_csv_path: str):
    pass


def parse_file_read_log(log_path: str):
    with open(log_path, "r") as f:
        epoch = 0
        train_epochs = {}
        test_epochs = {}
        current_train = []
        current_test = []
        for line in f:
            if line.startswith("="):
                continue
            elif line.startswith("EPOCH"):
                train_epochs.update({epoch: current_train})
                test_epochs.update({epoch: current_test})
                epoch += 1
                current_test = []
                current_train = []
            elif "clean" in line:
                continue
            elif "test" in line:
                current_test.append(line.split("/")[len(line.split("/")) - 1])
            else:
                current_train.append(line.split("/")[len(line.split("/")) - 1])

        result = dict(
            (i.strip('\n'), train_epochs[0].count(i)) for i in train_epochs[0])
        # for key in result.keys():
        #     print(f"{str(key)}: {str(result[key])}")
        keys2 = dict((i.strip('\n'), train_epochs[1].count(
            i)) for i in train_epochs[1]).keys()
        for key in result.keys():
            if key in keys2:
                print(key)

        for key in keys2:
            if key in result.keys():
                print(key)

        # print(len(train_epochs[0]))
        # print(len(test_epochs[0]))


def rms(x):
    return np.sqrt(np.mean(np.square(x), axis=-1))


def normalize(x):
    max_peak = np.max(np.abs(x))
    return x / max_peak


def pink_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = 1 / np.where(f == 0, float('inf'), np.sqrt(f))
    S = S / np.sqrt(np.mean(S ** 2))
    return np.fft.irfft(white * S, length)


def red_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = 1 / np.where(f == 0, float('inf'), f)
    S = S / np.sqrt(np.mean(S ** 2))
    return np.fft.irfft(white * S, length)


def blue_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = np.sqrt(f)
    S = S / np.sqrt(np.mean(S ** 2))
    return np.fft.irfft(white * S, length)

def violet_noise(length):
    white = np.fft.rfft(np.random.randn(length))
    f = np.fft.rfftfreq(length)
    S = f
    S = S / np.sqrt(np.mean(S ** 2))
    return np.fft.irfft(white * S, length)

def low_pass_filter():
    pass


def mix_with_snr(signal, noise, snr):
    signal_energy = np.mean(signal[0] ** 2)
    noise_energy = np.mean(noise ** 2)
    noise_gain = np.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
    a = np.sqrt(1 / (1 + noise_gain ** 2))
    b = np.sqrt(noise_gain ** 2 / (1 + noise_gain ** 2))
    generated = a * signal[0] + b * noise
    return generated


def white_noise(length):
    return np.random.randn(length)


def create_white_noise_dataset(vctk_corpus_path: str, sn_ratios: List[float], speakers: List[str],
                               original_clips_per_speaker: int, output_dir_path: str, setname: str) -> None:
    os.makedirs(f"{output_dir_path}/noisy_{setname}_wav", exist_ok=True)
    os.makedirs(f"{output_dir_path}/clean_{setname}_wav", exist_ok=True)
    vctk_corpus_path = os.path.join(vctk_corpus_path, "wav48_silence_trimmed")
    for speaker in speakers:
        speaker_dir = os.path.join(vctk_corpus_path, speaker)
        speaker_files = [os.path.join(speaker_dir, x) for x in os.listdir(speaker_dir) if "mic1" in x]
        random.shuffle(speaker_files)
        files_to_use = speaker_files[0:original_clips_per_speaker - 1]
        for file in files_to_use:
            signal = librosa.load(file, sr=16000)
            for snr in sn_ratios:
                # noise = np.random.normal(0, 1, len(signal[0]))
                noise = red_noise(len(signal[0]))
                generated = mix_with_snr(signal, noise, snr)
                new_name = pathlib.Path(file).stem + '.wav'
                save_path = os.path.join(f"{output_dir_path}/noisy_{setname}_wav", new_name)
                sf.write(save_path, data=generated, samplerate=16000, format='WAV')
                sf.write(os.path.join(f"{output_dir_path}/clean_{setname}_wav", new_name), data=signal[0],
                         samplerate=16000, format='WAV')


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    plt.loglog(f, np.abs(np.fft.rfft(s)))
    plt.show()

if __name__ == "__main__":
    # # apply_wiener_filter_to_file(input_file="/home/aleks/magister/audio-noise-reduction-using-nn/speech-denoising-wavenet/data/NSDTSEA/noisy_testset_wav/p232_154.wav", output_file="/home/aleks/Desktop/p232_154_wiener.wav")
    # # parse_file_read_log(
    # #     "/home/aleks/magister/audio-noise-reduction-using-nn/speech-denoising-wavenet/file_read_log.txt")
    # create_white_noise_dataset("/home/aleks/magister/datasets/VCTK-Corpus-0.92", sn_ratios=[10000, 15, 10, 5, 0],
    #                            speakers=os.listdir(
    #                                "/home/aleks/magister/datasets/VCTK-Corpus-0.92/wav48_silence_trimmed")[0:30],
    #                            original_clips_per_speaker=35,
    #                            output_dir_path="/home/aleks/magister/datasets/brown_NSDTSEA",
    #                            setname="trainset")
    # create_white_noise_dataset("/home/aleks/magister/datasets/VCTK-Corpus-0.92", sn_ratios=[60, 17.5, 12.5, 7.5, 2.5],
    #                            speakers=os.listdir(
    #                                "/home/aleks/magister/datasets/VCTK-Corpus-0.92/wav48_silence_trimmed")[30:33],
    #                            original_clips_per_speaker=40,
    #                            output_dir_path="/home/aleks/magister/datasets/brown_NSDTSEA",
    #                            setname="testset")
    signal = librosa.load("speech-denoising-wavenet/data/NSDTSEA/clean_trainset_wav/p226_043.wav", sr=16000)
    noise = red_noise(len(signal[0]))
    #plot_spectrum(noise)
    output = mix_with_snr(signal, noise, 10)
    sf.write(f"output.wav", data=output, samplerate=16000, format='WAV')
