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
    clips = os.listdir(os.path.join(input_path, "clips"))
    meta = pd.read_csv(os.path.join(input_path, "metadata.csv"))


    os.makedirs(os.path.join(output_path, "clips"), exist_ok=True)
    for i, clip in enumerate(clips):
        clip = os.path.join(os.path.join(input_path, "clips", clip))
        ir = ir_path
        ir = os.path.join(ir_path, ir)

        audio = librosa.core.load(clip, sr=16000)[0]
        ir_clip = librosa.core.load(ir, sr=16000)[0]

        reverbed = add_reverb(audio, ir_clip)
        reverbed = reverbed[0:len(audio)]

        output_clip = os.path.join(output_path, "clips", pathlib.Path(clip).name)
        sf.write(output_clip, reverbed, samplerate=16000)

    meta.to_csv(os.path.join(output_path, "metadata.csv"))



if __name__ == "__main__":
    add_reverb_to_dataset("/home/aleks/magister/datasets/inter/vctk_intermediate",
                          "/home/aleks/magister/datasets/inter/vctk_reverb_intermediate",
                          "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav")
    add_reverb_to_dataset("/home/aleks/magister/datasets/inter/cv_intermediate",
                          "/home/aleks/magister/datasets/inter/cv_reverb_intermediate",
                          "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav")

    add_reverb_to_dataset("/home/aleks/magister/datasets/inter/demand_intermediate",
                          "/home/aleks/magister/datasets/inter/demand_reverb_intermediate",
                          "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav")
    add_reverb_to_dataset("/home/aleks/magister/datasets/inter/esc50_intermediate",
                          "/home/aleks/magister/datasets/inter/esc50_reverb_intermediate",
                          "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav")
    add_reverb_to_dataset("/home/aleks/magister/datasets/inter/fma_intermediate",
                          "/home/aleks/magister/datasets/inter/fma_reverb_intermediate",
                          "/home/aleks/magister/datasets/Audio/Audio/h256_Stairwell_1txts.wav")