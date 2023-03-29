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


if __name__ == "__main__":
    apply_wiener_filter_to_file(input_file="/home/aleks/magister/audio-noise-reduction-using-nn/speech-denoising-wavenet/sessions/0012_10000/samples/samples_2/test_audio2_denoised.wav", output_file="/home/aleks/Desktop/sample_denoised_wiener_after.wav")
    pass