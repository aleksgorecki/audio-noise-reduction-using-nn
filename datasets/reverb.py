import os
import pathlib

import librosa
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import soundfile as sf
import numpy as np
import speechmetrics
import pandas


def add_reverb(sig, reverb_ir):
    if (len(sig) > len(reverb_ir)):
        ir_padded = np.concatenate([reverb_ir, np.zeros(len(sig) - len(reverb_ir), dtype=reverb_ir.dtype)])
    else:
        ir_padded = reverb_ir[:len(sig)]
    signal_reverbed = np.convolve(sig, ir_padded, mode="full")
    return signal_reverbed


def add_reverb_to_dataset(input_path, output_path, ir_path):
    clips = os.listdir(input_path)
    d = pathlib.Path(input_path).name


    os.makedirs(output_path, exist_ok=True)
    for i, clip in enumerate(clips):
        clip = os.path.join(os.path.join(input_path, clip))
        ir = ir_path
        ir = os.path.join(ir_path, ir)

        audio = librosa.core.load(clip, sr=16000)[0]
        ir_clip = librosa.core.load(ir, sr=16000)[0]

        reverbed = add_reverb(audio, ir_clip)
        reverbed = reverbed[0:len(audio)]

        output_clip = os.path.join(output_path, pathlib.Path(clip).name)
        sf.write(output_clip, reverbed, samplerate=16000)



if __name__ == "__main__":
    irs = [
    "h256_Stairwell_1txts.wav",
    "h255_MITCampus_StudentLounge_1txts.wav",
    "h252_Auditorium_1txts.wav"
    ]
    for i, split in enumerate(["train", "test", "val"]):
        ir = irs[i]
        add_reverb_to_dataset(f"/home/aleks/magister/datasets/final_datasets/general/vctk_demand/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/final_datasets/reverb/vctk_demand_reverb/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/Audio/Audio/{ir}")

        add_reverb_to_dataset(f"/home/aleks/magister/datasets/final_datasets/general/vctk_fma/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/final_datasets/reverb/vctk_fma_reverb/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/Audio/Audio/{ir}")

        add_reverb_to_dataset(f"/home/aleks/magister/datasets/final_datasets/general/vctk_esc50/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/final_datasets/reverb/vctk_esc50_reverb/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/Audio/Audio/{ir}")

        add_reverb_to_dataset(f"/home/aleks/magister/datasets/final_datasets/general/vctk_art/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/final_datasets/reverb/vctk_art_reverb/noisy_{split}set_wav",
                              f"/home/aleks/magister/datasets/Audio/Audio/{ir}")


    # clip_path = "/home/aleks/magister/datasets/VCTK-Corpus-0.92/wav48_silence_trimmed/p226/p226_002_mic1.flac"
    # ir_path = "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav"
    # audio = librosa.core.load(clip_path, sr=16000)[0]
    # ir_clip = librosa.core.load(ir_path, sr=16000)[0]
    # s = add_reverb(audio, ir_clip)
    # sf.write("/home/aleks/Desktop/testreverb.wav", s, samplerate=16000)

    #h252_Auditorium_1txts.wav
    #h255_MITCampus_StudentLounge_1txts.wav
    #h256_Stairwell_1txts.wav