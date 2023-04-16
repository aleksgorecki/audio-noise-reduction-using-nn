import librosa
from scipy.signal import wiener
from scipy.signal import lfilter, butter, filtfilt
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
import shutil


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


# def low_pass_filter(signal, cutoff_f, sr):
#     cutoff_index = int(cutoff_f * len(signal) / sr)
#     spectrum = np.fft.fft(signal)
#     spectrum[cutoff_index:] = 0
#     filtered = np.fft.ifft(spectrum)
#     return np.real(filtered)
#
#
# def high_pass_filter(signal, cutoff_f, sr):
#     cutoff_index = int(cutoff_f * len(signal) / sr)
#     spectrum = np.fft.fft(signal)
#     spectrum[0:cutoff_index] = 0
#     filtered = np.fft.ifft(spectrum)
#     return np.real(filtered)


def butter_lowpass(signal, cutoff, sr, order):
    b, a = butter(order, cutoff, fs=sr, btype='low', analog=False, output='ba')
    filtered = filtfilt(b, a, signal)
    return filtered


def butter_highpass(signal, cutoff, sr, order):
    b, a = butter(order, cutoff, fs=sr, btype='high', analog=False, output='ba')
    filtered = filtfilt(b, a, signal)
    return filtered


def butter_bandpass(data, low, high, sr, order=5):
    b, a = butter(order, [low, high], fs=sr, btype='band', analog=False, output='ba')
    filtered = filtfilt(b, a, data)
    return filtered


# def mix_with_snr(signal, noise, snr):
#     signal_energy = np.mean(signal[0] ** 2)
#     noise_energy = np.mean(noise ** 2)
#     noise_gain = np.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
#     a = np.sqrt(1 / (1 + noise_gain ** 2))
#     b = np.sqrt(noise_gain ** 2 / (1 + noise_gain ** 2))
#     generated = a * signal[0] + b * noise
#     return generated

def mix_with_snr(signal, noise, snr):
    rms_sig = rms(signal)
    rms_noise = rms(noise)
    rms_target = rms_sig / (10 ** (snr / 20))
    generated = signal[0] + (noise * (rms_target / rms_noise))
    return generated


def shrink_dataset(input_dir, new_sample_number, output_dir):
    all_files = os.listdir(input_dir)
    all_files = [os.path.join(input_dir, x) for x in all_files]
    clean_all_files = [str(x).replace('noisy', 'clean') for x in all_files]
    indices_to_copy = np.random.randint(0, len(all_files), new_sample_number)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(str(output_dir).replace('noisy', 'clean'), exist_ok=True)
    for idx in indices_to_copy:
        shutil.copy(all_files[idx], os.path.join(output_dir, pathlib.Path(all_files[idx]).name))
        shutil.copy(clean_all_files[idx],
                    os.path.join(str(output_dir).replace('noisy', 'clean'), pathlib.Path(all_files[idx]).name))


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
                noise = np.random.normal(0, 1, len(signal[0]))
                noise = butter_lowpass(noise, 3000, 16000, 5)
                generated = mix_with_snr(signal[0], noise, snr)
                new_name = pathlib.Path(file).stem + '.wav'
                save_path = os.path.join(f"{output_dir_path}/noisy_{setname}_wav", new_name)
                sf.write(save_path, data=generated, samplerate=16000, format='WAV')
                sf.write(os.path.join(f"{output_dir_path}/clean_{setname}_wav", new_name), data=signal[0],
                         samplerate=16000, format='WAV')


def plot_spectrum(s):
    f = np.fft.rfftfreq(len(s))
    plt.loglog(f, np.abs(np.fft.rfft(s)))
    plt.show()


def eval_model_on_dir(config_path, checkpoint_path, input_dir):
    pass


if __name__ == "__main__":
    pass
    # # apply_wiener_filter_to_file(input_file="/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/data/NSDTSEA/noisy_testset_wav/p232_154.wav", output_file="/home/aleks/Desktop/p232_154_wiener.wav")
    # # parse_file_read_log(
    # #     "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/file_read_log.txt")
    # create_white_noise_dataset("/home/aleks/magister/datasets/VCTK-Corpus-0.92", sn_ratios=[10000, 50, 30, 20, 10],
    #                            speakers=os.listdir(
    #                                "/home/aleks/magister/datasets/VCTK-Corpus-0.92/wav48_silence_trimmed")[0:30],
    #                            original_clips_per_speaker=35,
    #                            output_dir_path="/home/aleks/magister/datasets/lowpass_gaussian_NSDTSEA",
    #                            setname="trainset")
    # create_white_noise_dataset("/home/aleks/magister/datasets/VCTK-Corpus-0.92", sn_ratios=[60, 40, 20, 10],
    #                            speakers=os.listdir(
    #                                "/home/aleks/magister/datasets/VCTK-Corpus-0.92/wav48_silence_trimmed")[30:37],
    #                            original_clips_per_speaker=30,
    #                            output_dir_path="/home/aleks/magister/datasets/lowpass_gaussian_NSDTSEA",
    #                            setname="testset")
    # signal = librosa.load("speech_denoising_wavenet/data/NSDTSEA/clean_trainset_wav/p226_043.wav", sr=16000)
    # noise = red_noise(len(signal[0]))
    # # plot_spectrum(noise)
    # output = mix_with_snr(signal, noise, 10)
    # sf.write(f"output.wav", data=output, samplerate=16000, format='WAV')
    # subsets = ['noisy_trainset_wav']
    # for subset in subsets:
    #     shrink_dataset(f"../datasets/NSDTSEA/{subset}", 1200, f"../datasets/NSDTSEA_shrink/{subset}")
    # subsets = ['noisy_testset_wav']
    # for subset in subsets:
    #     shrink_dataset(f"../datasets/NSDTSEA/{subset}", 120, f"../datasets/NSDTSEA_shrink/{subset}")
    # signal = librosa.load("speech_denoising_wavenet/data/NSDTSEA/clean_trainset_wav/p226_043.wav", sr=16000)
    # noise = white_noise(len(signal[0]))
    # noise = butter_bandpass(noise, 2000, 4000, 16000, 5)
    # sf.write("./filtered", data=noise, samplerate=16000, format='WAV')
    # files = os.listdir("../datasets/NSDTSEA/noisy_trainset_wav")
    # files = [f"../datasets/NSDTSEA/noisy_trainset_wav/{x}" for x in files]
    # rmss = []
    # for file in files:
    #     audio = librosa.load(file, sr=16000)
    #     rmss.append((rms(audio[0])))
    # print(min(rmss))
    # print(files[np.argmin(rmss)])
    # print(files[np.argmax(rmss)])
