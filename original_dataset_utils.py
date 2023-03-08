import librosa
import soundfile as sf
from typing import Tuple, List
from numpy.typing import NDArray
import numpy as np
import os


def list_dir_with_full_path(dir: str) -> List[str]:
    return [os.path.join(dir, file) for file in os.listdir(dir)]


def join_audio_files(files: List[str], sr=None) -> Tuple[NDArray, float]:
    joined = np.array([])
    if sr is None:
        sr = librosa.get_samplerate(files[0])
    for file in files:
        audio = librosa.load(file, sr=sr, mono=True)
        joined = np.concatenate([joined, audio[0]])
    return joined, sr


def cut_audio_into_segments(data: NDArray, sr: float, seg_len_s: float) -> List:
    seg_len_samples = sr * seg_len_s
    segments = np.array_split(data, np.arange(seg_len_samples, len(data), seg_len_samples))

    return [x for x in segments if x.size == seg_len_samples]


def save_audio_list_as_files(audios: List[NDArray], sr: float, output_dir: str, file_name_prefix: str):
    for idx, audio in enumerate(audios):
        file_name = f"{output_dir}/{file_name_prefix}_{idx}.wav"
        sf.write(file_name, data=audio, samplerate=int(sr), format='WAV')


if __name__ == "__main__":
    joined = join_audio_files(list(filter(lambda filename: "mic1" in filename, list_dir_with_full_path(
        "../datasets/VCTK-Corpus-0.92/wav48_silence_trimmed/p225"))), sr=16000)
    audios = cut_audio_into_segments(joined[0], joined[1], seg_len_s=3)
    os.makedirs("./cut", exist_ok=True)
    save_audio_list_as_files(audios, joined[1], "./cut", "p225")
