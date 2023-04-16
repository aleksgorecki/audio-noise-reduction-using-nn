import sys
import pypesq
import pystoi
import os
import librosa
import numpy as np
import json
import tensorflow as tf
import soundfile as sf
import speechmetrics
sys.path.append("speech_denoising_wavenet")
from speech_denoising_wavenet.models import DenoisingWavenet


my_speechmetrics = speechmetrics.load("", window=2)


def load_example(noisy_path, clean_path, sr):
    noisy = librosa.load(noisy_path, sr=sr)
    clean = librosa.load(clean_path, sr=sr)
    return noisy[0], clean[0]


def slice_example(example, length):
    if example.shape[0] == length:
        return example

    slice_points = list(range(length, example.shape[0], length))
    slices = np.split(example, slice_points)
    if slices[-1].shape[0] != length:
        slices.pop()

    return slices


def prepare_batch(example, model: DenoisingWavenet):
    target_field_len = int(model.target_field_length)
    input_len = int(model.input_length)
    num_output_samples = int(example.shape[0] - (model.receptive_field_length - 1))

    batch = np.zeros(shape=(int(num_output_samples / target_field_len), int(input_len)))
    for i in range(0, int(num_output_samples / target_field_len)):
        target_field_window_start = i * target_field_len
        if int(target_field_window_start + target_field_len) > num_output_samples:
            batch[i][0:num_output_samples - target_field_window_start] = example[target_field_window_start: num_output_samples]
            continue
        batch[i] = example[target_field_window_start: target_field_window_start + input_len]
    # for i, target_field_window_start in enumerate(list(range(0, num_output_samples, target_field_len))):
    #     if int(target_field_window_start + input_len) > num_output_samples:
    #         batch[i][0:example.shape[0] - target_field_window_start] = example[target_field_window_start:]
    #         continue
    #     batch[i] = example[target_field_window_start: target_field_window_start + input_len]
    return batch


def calculate_mean_metrics(metrics_dicts):

    keys = metrics_dicts[0].keys()
    ret_dict = dict()
    for key in keys:
        ret_dict.update({
            key: np.mean([np.mean(x[key]) for x in metrics_dicts])
        })
    return ret_dict


def evaluate_example(example_noisy, example_clean, model: DenoisingWavenet):
    # sliced_clean = slice_example(example_clean, int(model.input_length))

    noisy_batch = prepare_batch(example_noisy, model)
    predicted_batch = model.model.predict(noisy_batch, verbose=0)[0]
    predicted_vector = predicted_batch.flatten()
    example_clean = example_clean[: predicted_vector.shape[0]]

    example_metrics = {
        "pesq": pypesq.pesq(ref=example_clean, deg=predicted_vector, fs=model.config["dataset"]["sample_rate"]),
        "stoi": pystoi.stoi(x=example_clean, y=predicted_vector, fs_sig=model.config["dataset"]["sample_rate"])
    }

    speechmetrics_res = my_speechmetrics(predicted_vector, example_clean, rate=int(model.config["dataset"]["sample_rate"]))
    example_metrics.update(speechmetrics_res)

    # example_metrics = dict({
    #     "pesq": list(),
    #     "stoi": list()
    # })
    # for i, predicted_example in enumerate(predicted_batch):
    #     example_metrics["pesq"].append(
    #         pypesq.pesq(ref=sliced_clean[i], deg=predicted_example, fs=model.config["dataset"]["sample_rate"])
    #     )
    #     example_metrics["stoi"].append(
    #         pystoi.stoi(x=sliced_clean[i], y=predicted_example, fs_sig=model.config["dataset"]["sample_rate"])
    #     )

    return example_metrics


def evaluate_on_testset(main_set, model: DenoisingWavenet):

    noisy_files = os.listdir(os.path.join(main_set, "noisy_testset_wav"))

    calculated_metrics = list()

    for i, file in enumerate(noisy_files):
        noisy_file = os.path.join(main_set, "noisy_testset_wav", file)
        clean_file = os.path.join(main_set, "clean_testset_wav", file)
        noisy, clean = load_example(noisy_file, clean_file, sr=model.config["dataset"]["sample_rate"])
        example_metrics = evaluate_example(noisy, clean, model)
        calculated_metrics.append(example_metrics)

    mean_metrics = calculate_mean_metrics(calculated_metrics)
    return mean_metrics


if __name__ == "__main__":
    os.chdir("/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet")

    dataset = "data/NSDTSEA/"

    config_path = "sessions/default_5_l1l2_weighted_spec/config.json"
    checkpoint_path = "sessions/default_5_l1l2_weighted_spec/checkpoints/checkpoint.00040-0.258.hdf5"

    with open(config_path, "r") as f:
        config = json.load(f)
        model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
        metrics = evaluate_on_testset(dataset, model)
        print(metrics)

    config_path = "sessions/default_9_combined_sc_rms_loss/config.json"
    checkpoint_path = "sessions/default_9_combined_sc_rms_loss/checkpoints/checkpoint.00048-0.762.hdf5"

    with open(config_path, "r") as f:
        config = json.load(f)
        model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
        metrics = evaluate_on_testset(dataset, model)
        print(metrics)
