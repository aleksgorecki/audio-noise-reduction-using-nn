import pandas as pd
import os
import librosa
import numpy as np
from typing import Sized
import zipfile
import io
from datasets.dataset_constants import DEMAND_NOISE_CLASSES, VCTK_speakers, ESC50_CATEGORIES
import soundfile as sf


def split_iterable(collection: Sized, val_ratio: float, test_ratio: float):
    test = collection[-int(len(collection) * test_ratio):]
    val = collection[-int(len(collection) * val_ratio + len(test)):-len(test)]
    train = collection[0:-(len(val) + len(test))]

    return train, val, test


def demand_to_intermediate_form(demand_zipped_dir: str, output_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15):
    output_meta = pd.DataFrame(columns=["clip", "category", "split", "top_category"])
    split_names = ["train", "val", "test"]
    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    for noise_class in DEMAND_NOISE_CLASSES:
        nc_path = f"{demand_zipped_dir}/{noise_class}_48k.zip"
        with zipfile.ZipFile(nc_path, "r") as archive:
            ch01 = io.BytesIO(archive.read(f"{noise_class}/ch01.wav"))
            ch01_audio = librosa.core.load(ch01, sr=16000, mono=True)[0]

            train = ch01_audio[0: - int(len(ch01_audio) * val_ratio + len(ch01_audio) * test_ratio)]
            val = ch01_audio[len(train): int(len(train) + len(ch01_audio) * val_ratio)]
            test = ch01_audio[len(train) + len(val):]

            for i, subset in enumerate([train, val, test]):
                num_clips = int((len(subset) / (4 * 16000)))
                clips = np.array_split(subset, num_clips)
                clips = [x for x in clips if len(x) >= int(4 * 16000)]
                for i_clip, clip in enumerate(clips):
                    name = os.path.join(f"{noise_class}_{split_names[i]}_{i_clip}.wav")
                    record = pd.DataFrame(data=[{"clip": name, "category": noise_class, "split": split_names[i]}])
                    output_meta = pd.concat((output_meta, record), ignore_index=True)
                    sf.write(os.path.join(clips_dir, name), clip, samplerate=16000)
    output_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

def vctk_to_intermediate_form(vctk_original_path: str, output_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15, clips_per_speaker: int = 30):
    clips_path = os.path.join(vctk_original_path, "wav48_silence_trimmed")
    output_clips_path = os.path.join(output_dir, "clips")
    os.makedirs(output_clips_path, exist_ok=True)

    speakers = list(filter(lambda x: len(os.listdir(os.path.join(clips_path, x))) > 600, VCTK_speakers.copy()))
    np.random.shuffle(speakers)

    test = speakers[-int(len(speakers) * test_ratio):]
    val = speakers[-int(len(speakers) * val_ratio + len(test)):-len(test)]
    train = speakers[0:-(len(val) + len(test))]

    split_names = ["train", "val", "test"]
    output_meta = pd.DataFrame(columns=["clip", "speaker", "split"])
    for i, split in enumerate((train, val, test)):
        for speaker in split:
            speaker_path = os.path.join(clips_path, speaker)
            viable_clips = list(filter(lambda x: "mic2" not in x, os.listdir(speaker_path)))
            chosen_clips = np.random.choice(viable_clips, clips_per_speaker, replace=False)
            for clip in chosen_clips:
                record = pd.DataFrame(data=[{"clip": clip.replace(".flac", ".wav"), "speaker": speaker, "split": split_names[i]}])
                output_meta = pd.concat((output_meta, record), ignore_index=True)
                audio = librosa.core.load(os.path.join(speaker_path, clip), sr=16000, mono=True)[0]
                audio = librosa.util.normalize(audio)
                audio, _ = librosa.effects.trim(audio, top_db=10, frame_length=256, hop_length=64)
                if len(audio) / 16000 > 4.0:
                    center = len(audio) // 2
                    low = center - int((4 * 16000 // 2))
                    high = center + int((4 * 16000 // 2))
                    audio = audio[low:high]
                sf.write(os.path.join(output_clips_path, clip.replace(".flac", ".wav")), audio, samplerate=16000)
    output_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def cv_to_intermediate_form(cv_original_path: str, output_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15, clips_per_speaker: int = 30):
    metadata = pd.read_csv(os.path.join(cv_original_path, "pl", "validated.tsv"), sep='\t')
    metadata = metadata[["client_id", "path"]]
    clips_dir = os.path.join(cv_original_path, "pl", "clips")
    output_clips_path = os.path.join(output_dir, "clips")
    os.makedirs(output_clips_path, exist_ok=True)

    filtered = metadata[metadata.groupby("client_id").transform('size') > 200]
    grouped = filtered.groupby("client_id")

    output_meta = pd.DataFrame(columns=["clip", "speaker", "split"])
    train, val, test = split_iterable(list(grouped.groups.keys()), val_ratio, test_ratio)

    split_names = ("train", "val", "test")
    for i, split in enumerate((train, val, test)):
        for speaker in split:
            clips = grouped.get_group(speaker)["path"].tolist()[0:clips_per_speaker]
            for clip in clips:
                record = pd.DataFrame(data=[{"clip": clip.replace(".mp3", ".wav"), "speaker": speaker, "split": split_names[i]}])
                output_meta = pd.concat((output_meta, record), ignore_index=True)
                audio = librosa.core.load(os.path.join(clips_dir, clip), sr=16000, mono=True)[0]
                audio, _ = librosa.effects.trim(audio)
                if len(audio) / 16000 > 4.0:
                    center = len(audio) // 2
                    low = center - int((4 * 16000 // 2))
                    high = center + int((4 * 16000 // 2))
                    audio = audio[low:high]
                sf.write(os.path.join(output_clips_path, clip.replace(".mp3", ".wav")), audio, samplerate=16000)

    output_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def fma_to_intermediate_form(fma_original_path: str, metadata_path: str, output_dir: str, val_ratio: float = 0.15, test_ratio: float = 0.15, clips_per_class: int = 100, clips_per_track: int = 30):
    # metadata = pd.read_csv(metadata_path, usecols=[0, 32, 40])
    # metadata = metadata.dropna(subset=[metadata.columns[2]])

    metadata = pd.read_csv(metadata_path, usecols=[0, 32, 40])
    metadata.columns = ["track_id", "subset", "genre"]
    metadata.drop(metadata.head(2).index, inplace=True)
    metadata = metadata.dropna()
    metadata = metadata.loc[metadata["subset"] == "small"]
    grouped = metadata.groupby("genre")

    output_clips_path = os.path.join(output_dir, "clips")
    os.makedirs(output_clips_path, exist_ok=True)

    output_meta = pd.DataFrame(columns=["clip", "category", "split"])
    split_names = ("train", "val", "test")
    for genre in grouped.groups.keys():
        clips = grouped.get_group(genre)["track_id"].tolist()[0:clips_per_class]
        train, val, test = split_iterable(clips, val_ratio, test_ratio)
        for i, split in enumerate((train, val, test)):
            for track_id in split:
                zfilled_id = str(track_id).zfill(6)
                dir_name = zfilled_id[:3]
                audio = librosa.core.load(os.path.join(fma_original_path, dir_name, zfilled_id + ".mp3"), sr=16000, mono=True)[0]
                if len(audio) / 16000 > 4.0:
                    center = len(audio) // 2
                    low = center - int((3 * 16000 // 2))
                    high = center + int((3 * 16000 // 2))
                    audio = audio[low:high]
                record = pd.DataFrame(data=[{"clip": zfilled_id + ".wav", "category": genre, "split": split_names[i]}])
                output_meta = pd.concat((output_meta, record), ignore_index=True)
                sf.write(os.path.join(output_clips_path, zfilled_id + ".wav"), audio, samplerate=16000)
    output_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


def esc50_to_intermediate(esc_original_path: str, output_dir: str):
    metadata = pd.read_csv(os.path.join(esc_original_path, "meta", "esc50.csv"))
    train_meta = metadata.loc[metadata["fold"].isin((1, 2, 3))]
    val_meta = metadata.loc[metadata["fold"] == 4]
    test_meta = metadata.loc[metadata["fold"] == 5]

    inverse_top_category_dict = dict()
    for key in ESC50_CATEGORIES.keys():
        for category in ESC50_CATEGORIES[key]:
            inverse_top_category_dict.update({category: key})

    output_clips_path = os.path.join(output_dir, "clips")
    os.makedirs(output_clips_path, exist_ok=True)

    output_meta = pd.DataFrame(columns=["clip", "category", "split", "top_category"])
    split_names = ("train", "val", "test")
    for i, split in enumerate((train_meta, val_meta, test_meta)):
        for row_index, record in split.iterrows():
            clip = record["filename"]
            category = record["category"]
            top_category = inverse_top_category_dict[category]
            new_record = pd.DataFrame(data=[{"clip": clip, "category": category, "split": split_names[i], "top_category": top_category}])
            output_meta = pd.concat((output_meta, new_record), ignore_index=True)
            audio = librosa.core.load(os.path.join(esc_original_path, "audio", clip), sr=16000, mono=True)[0]
            sf.write(os.path.join(output_clips_path, clip), audio, samplerate=16000)

    output_meta.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)


if __name__ == "__main__":
    pass
    #demand_to_intermediate_form("../../demand", "../../demand_intermediate")
    # cv_to_intermediate_form("../../cv-corpus-12.0-2022-12-07", "../../cv_intermediate")
    # esc50_to_intermediate("../../ESC-50-master", "../../esc50_intermediate")
    # vctk_to_intermediate_form("../../VCTK-Corpus", "../../vctk_intermediate")
    vctk_to_intermediate_form("/home/aleks/magister/datasets/VCTK-Corpus-0.92/", "/home/aleks/magister/datasets/inter/vctk_intermediate/", clips_per_speaker=200)
    #cv_to_intermediate_form("/home/aleks/magister/datasets/cv-corpus-13.0-2023-03-09", "/home/aleks/magister/datasets/inter/cv_intermediate", clips_per_speaker=200)
    # fma_to_intermediate_form("/home/aleks/magister/datasets/fma_small", "/home/aleks/magister/datasets/fma_metadata/tracks.csv", "/home/aleks/magister/datasets/fma_intermediate", clips_per_class=100)
    #esc50_to_intermediate("/home/aleks/magister/datasets/ESC-50-master", "/home/aleks/magister/datasets/esc50_intermediate")