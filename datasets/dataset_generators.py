import pandas as pd
import os
import librosa
import numpy as np
from typing import Sized, List
import zipfile
import io
from datasets.dataset_constants import DEMAND_NOISE_CLASSES, VCTK_speakers, ESC50_CATEGORIES
import soundfile as sf
import my_utils


def generate_final_subset(noise_inter: str, clean_inter: str, output_dataset: str, split_name: str,  speakers_num: int, clips_per_speaker: int, SNRs: List[int]):

    split_dir_name = f"noisy_{split_name}set_wav"
    split_dir_name_clean = f"clean_{split_name}set_wav"
    os.makedirs(os.path.join(output_dataset, split_dir_name), exist_ok=True)
    os.makedirs(os.path.join(output_dataset, split_dir_name_clean), exist_ok=True)
    noise_meta = pd.read_csv(os.path.join(noise_inter, "metadata.csv"))
    clean_meta = pd.read_csv(os.path.join(clean_inter, "metadata.csv"))
    noise_meta = noise_meta[noise_meta["split"] == split_name]
    clean_meta = clean_meta[clean_meta["split"] == split_name]

    grouped_by_speaker = clean_meta.groupby("speaker")
    grouped_by_speaker = [group[1] for group in list(grouped_by_speaker)[:speakers_num]]

    output_meta = pd.DataFrame(columns=["clip", "speaker", "noise_category", "snr"])
    for speaker_df in grouped_by_speaker:
        clips = speaker_df.sample(n=clips_per_speaker)
        noise_clips = noise_meta.sample(n=clips_per_speaker).to_dict("records")
        for i, row in enumerate(clips.iterrows()):
            speaker = row[1]["speaker"]
            noise_file = noise_clips[i]["clip"]
            clean_file = row[1]["clip"]
            noise_category = noise_clips[i]["category"]
            snr = SNRs[i % len(SNRs)]

            noise_path = os.path.join(noise_inter, "clips", noise_file)
            clean_path = os.path.join(clean_inter, "clips", clean_file)

            noise_clip = librosa.core.load(noise_path, sr=16000)[0]
            clean_clip = librosa.core.load(clean_path, sr=16000)[0]

            if len(noise_clip) > len(clean_clip):
                random_offset = np.random.randint(0, len(noise_clip) - len(clean_clip))
                noise_clip = noise_clip[random_offset:random_offset + len(clean_clip)]
            else:
                noise_clip = np.concatenate([noise_clip, np.zeros(len(clean_clip) - len(noise_clip), dtype=noise_clip.dtype)])

            mixed = my_utils.mix_with_snr(clean_clip, noise_clip, snr)

            output_filename = f"{speaker}_{noise_category}_{snr}db.wav"
            output_path = os.path.join(output_dataset, split_dir_name, output_filename)

            new_record = pd.DataFrame(data=[{"clip": output_filename, "speaker": speaker, "noise_category": noise_category, "snr": snr}])
            output_meta = pd.concat((output_meta, new_record), ignore_index=True)

            sf.write(output_path, data=mixed, samplerate=16000)
            sf.write(os.path.join(output_dataset, split_dir_name_clean, output_filename), data=clean_clip, samplerate=16000)

    output_meta.to_csv(os.path.join(output_dataset, f"metadata_{split_name}.csv"), index=False)



if __name__ == "__main__":
    # noise_path = "/home/aleks/magister/datasets/inter/esc50_intermediate"
    # clean_path = "/home/aleks/magister/datasets/inter/vctk_intermediate"
    # output_path = "/home/aleks/magister/datasets/final_datasets/alpha/"
    # split_name = "train"
    # speakers_num = 30
    # clips_per_speaker = 100
    # SNRs = [0, 5, 10, 15, 20]
    # generate_final_subset(noise_path, clean_path, output_path, split_name, speakers_num, clips_per_speaker, SNRs)
    #
    # noise_path = "/home/aleks/magister/datasets/inter/esc50_intermediate"
    # clean_path = "/home/aleks/magister/datasets/inter/vctk_intermediate"
    # output_path = "/home/aleks/magister/datasets/final_datasets/alpha/"
    # split_name = "test"
    # speakers_num = 10
    # clips_per_speaker = 100
    # SNRs = [-4, 4, 8, 12, 18]
    # generate_final_subset(noise_path, clean_path, output_path, split_name, speakers_num, clips_per_speaker, SNRs)

    noise_path = "/home/aleks/magister/datasets/inter/fma_intermediate"
    clean_path = "/home/aleks/magister/datasets/inter/vctk_intermediate"
    output_path = "/home/aleks/magister/datasets/final_datasets/vctk_fma/"
    split_name = "train"
    speakers_num = 30
    clips_per_speaker = 100
    SNRs = [0, 5, 10, 15, 20]
    generate_final_subset(noise_path, clean_path, output_path, split_name, speakers_num, clips_per_speaker, SNRs)

    noise_path = "/home/aleks/magister/datasets/inter/fma_intermediate"
    clean_path = "/home/aleks/magister/datasets/inter/vctk_intermediate"
    output_path = "/home/aleks/magister/datasets/final_datasets/vctk_fma/"
    split_name = "test"
    speakers_num = 10
    clips_per_speaker = 100
    SNRs = [-4, 4, 8, 12, 18]
    generate_final_subset(noise_path, clean_path, output_path, split_name, speakers_num, clips_per_speaker, SNRs)