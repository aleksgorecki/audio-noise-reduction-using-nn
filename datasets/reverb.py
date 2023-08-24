import os
import pathlib
import librosa
import soundfile as sf
import numpy as np


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
