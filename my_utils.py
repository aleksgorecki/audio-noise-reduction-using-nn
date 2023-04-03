import librosa
from scipy.signal import wiener
import numpy as np
import os
import soundfile as sf
import pathlib
import csv


def apply_wiener_filter_to_file(input_file: str, output_file: str) -> None:
    audio_tuple = librosa.core.load(path=input_file, sr=None)
    filtered = wiener(audio_tuple[0])
    sf.write(output_file, data=filtered, format='WAV', samplerate=audio_tuple[1])


def apply_wiener_filter_to_dir(input_dir: str, output_dir: str) -> None:
    files = os.listdir(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    for af in files:
        output_name = (pathlib.Path(af).stem) + '.wav'
        apply_wiener_filter_to_file(f"{input_dir}/{af}", output_file=f"{output_dir}/{output_name}")


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
                train_epochs.update({ epoch: current_train })
                test_epochs.update({ epoch: current_test })
                epoch += 1
                current_test = []
                current_train = []
            elif "clean" in line:
                continue
            elif "test" in line:
                current_test.append(line.split("/")[len(line.split("/")) - 1])
            else:
                current_train.append(line.split("/")[len(line.split("/")) - 1])

    
        result = dict((i.strip('\n'), train_epochs[0].count(i)) for i in train_epochs[0])
        # for key in result.keys():
        #     print(f"{str(key)}: {str(result[key])}")
        occurences = []
        keys2 = dict((i.strip('\n'), train_epochs[1].count(i)) for i in train_epochs[1]).keys()
        for key in result.keys():
            if key in keys2:
                print(key)

        for key in keys2:
            if key in result.keys():
                print(key)

        # print(len(train_epochs[0]))
        # print(len(test_epochs[0]))




if __name__ == "__main__":
    #apply_wiener_filter_to_file(input_file="/home/aleks/magister/audio-noise-reduction-using-nn/speech-denoising-wavenet/data/NSDTSEA/noisy_testset_wav/p232_154.wav", output_file="/home/aleks/Desktop/p232_154_wiener.wav")
    parse_file_read_log("/home/aleks/magister/audio-noise-reduction-using-nn/speech-denoising-wavenet/file_read_log.txt")