import os
import librosa
import numpy as np
import speechmetrics
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet import util
import tqdm
import scipy
import pandas as pd
import pathlib


METRICS = ["mosnet", "pesq", "sisdr"]
my_speechmetrics = speechmetrics.load(METRICS, window=2.0)


def mse(a, b):
    return np.mean(np.square(a - b))


def mae(a, b):
    return np.mean(np.abs(a - b))


def get_latest_checkpoint(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    latest_checkpoint = max(checkpoints, key=lambda x: int(x[11:16]))
    return os.path.join(checkpoints_dir, latest_checkpoint)


def get_best_checkpoint(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    best_checkpoint = min(checkpoints, key=lambda x: float(x[17:22]))
    return os.path.join(checkpoints_dir, best_checkpoint)


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


def evaluate_example_original(example_clean, example_noisy, model: DenoisingWavenet):
    if len(example_noisy) < model.receptive_field_length:
        raise ValueError("Input is not long enough to be used with this model.")

    batch_size = int(model.input_length)
    padding_length = int(model.receptive_field_length / 2)
    example_noisy = np.concatenate(
        [example_noisy, np.zeros(padding_length, dtype=example_noisy.dtype)]
    )
    num_output_samples = example_noisy.shape[0] - (model.receptive_field_length - 1)
    num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
    num_batches = int(np.ceil(num_fragments / batch_size))

    condition_input = util.binary_encode(int(0), 29)[0]
    denoised_output = []
    noise_output = []
    num_pad_values = 0
    fragment_i = 0
    batches = []
    for batch_i in tqdm.tqdm(range(0, num_batches)):
        if batch_i == num_batches - 1:
            batch_size = num_fragments - batch_i * batch_size

        condition_batch = np.array(
            [
                condition_input,
            ]
            * batch_size,
            dtype="uint8",
        )
        input_batch = np.zeros((batch_size, int(model.input_length)))

        for batch_fragment_i in range(0, batch_size):
            if fragment_i + model.target_field_length > num_output_samples:
                remainder = example_noisy[fragment_i:]
                current_fragment = np.zeros((int(model.input_length),))
                current_fragment[: remainder.shape[0]] = remainder
                num_pad_values = int(model.input_length - remainder.shape[0])
            else:
                current_fragment = example_noisy[
                    fragment_i : fragment_i + int(model.input_length)
                ]

            input_batch[batch_fragment_i, :] = current_fragment
            fragment_i += model.target_field_length
        batches.append(input_batch)

    batches = np.array(batches[0])
    predicted_batch = model.model.predict(
        {"data_input": batches, "condition_input": condition_batch}, verbose=0
    )[0]

    for row in predicted_batch:
        if type(row) is list:
            denoised_output_fragment = row[0]

        denoised_output_fragment = row[
            :, model.target_padding : model.target_padding + model.target_field_length
        ]
        denoised_output_fragment = denoised_output_fragment.flatten().tolist()

        if type(row) is float:
            denoised_output_fragment = [denoised_output_fragment]

        denoised_output = denoised_output + denoised_output_fragment

        denoised_output = np.array(denoised_output)

        if num_pad_values != 0:
            denoised_output = denoised_output[:-num_pad_values]

    predicted_vector = denoised_output

    speechmetrics_res = my_speechmetrics(
        predicted_vector, example_clean, rate=int(16000)
    )
    example_metrics = speechmetrics_res
    example_metrics.update({"mse": mse(predicted_vector, example_clean)})
    example_metrics.update({"mae": mae(predicted_vector, example_clean)})
    example_metrics.update({"mse_in": mse(example_noisy, example_clean)})
    example_metrics.update({"mae_in": mae(example_noisy, example_clean)})

    return example_metrics


def prepare_batch(example, model: DenoisingWavenet):
    target_field_len = int(model.target_field_length)
    input_len = int(model.input_length)
    num_output_samples = int(example.shape[0] - (model.receptive_field_length - 1))

    batch = np.zeros(shape=(int(num_output_samples / target_field_len), int(input_len)))
    for i in range(0, int(num_output_samples / target_field_len)):
        target_field_window_start = i * target_field_len
        if int(target_field_window_start + target_field_len) > num_output_samples:
            batch[i][0 : num_output_samples - target_field_window_start] = example[
                target_field_window_start:num_output_samples
            ]
            continue
        batch[i] = example[
            target_field_window_start : target_field_window_start + input_len
        ]
    if not bool(model.config["model"].get("no_conditioning")):
        condition_input = util.binary_encode(int(0), 29)[0]
        condition_batch = np.array(
            [
                condition_input,
            ]
            * len(batch),
            dtype="uint8",
        )
    else:
        condition_batch = None

    return batch, condition_batch


def calculate_mean_metrics(metrics_dicts):
    keys = metrics_dicts[0].keys()
    ret_dict = dict()
    for key in keys:
        ret_dict.update({key: np.mean([np.mean(x[key]) for x in metrics_dicts])})
    return ret_dict


def evaluate_example(
    example_noisy, example_clean, model: DenoisingWavenet, normalize, trim=True
):
    noisy_batch, condition_batch = prepare_batch(example_noisy, model)
    if condition_batch is not None:
        predicted_batch = model.model.predict(
            {"data_input": noisy_batch, "condition_input": condition_batch}, verbose=0
        )[0]
    else:
        predicted_batch = model.model.predict(noisy_batch, verbose=0)[0]

    predicted_batch = predicted_batch[
        :, model.target_padding : model.target_padding + model.target_field_length
    ]

    predicted_vector = predicted_batch.flatten()
    example_clean = example_clean[
        model.half_receptive_field_length : model.half_receptive_field_length
        + len(predicted_vector)
    ]
    example_noisy = example_noisy[
        model.half_receptive_field_length : model.half_receptive_field_length
        + len(predicted_vector)
    ]

    speechmetrics_res = my_speechmetrics(predicted_vector, example_clean, rate=16000)
    example_metrics = speechmetrics_res
    example_metrics.update({"mse": mse(predicted_vector, example_clean)})
    example_metrics.update({"mae": mae(predicted_vector, example_clean)})
    example_metrics.update({"mse_in": mse(example_noisy, example_clean)})
    example_metrics.update({"mae_in": mae(example_noisy, example_clean)})

    noise_in_denoised_output = predicted_vector - example_clean
    rms_clean = util.rms(example_clean)
    rms_noise_out = util.rms(noise_in_denoised_output)
    rms_noise_in = util.rms(example_noisy - example_clean)

    new_snr_db = np.round(util.snr_db(rms_clean, rms_noise_out))
    initial_snr_db = np.round(util.snr_db(rms_clean, rms_noise_in))

    example_metrics.update({"snr_calc": new_snr_db})

    ref_metrics = my_speechmetrics(example_noisy, example_clean, rate=16000)
    ref_metrics.update({"mse": mse(example_noisy, example_clean)})
    ref_metrics.update({"mae": mae(example_noisy, example_clean)})
    ref_metrics.update({"snr_calc": initial_snr_db})
    return example_metrics, ref_metrics


def evaluate_example_wiener(example_noisy, example_clean):
    predicted_vector = scipy.signal.wiener(example_noisy).flatten()
    example_clean = example_clean[: predicted_vector.shape[0]]

    speechmetrics_res = my_speechmetrics(
        predicted_vector, example_clean, rate=int(16000)
    )
    example_metrics = speechmetrics_res

    return example_metrics


def evaluate_on_testset(
    main_set, model: DenoisingWavenet, max_files=None, normalize=False
):
    noisy_meta = pd.read_csv(os.path.join(main_set, "metadata_val.csv"))
    if max_files is not None:
        noisy_meta = noisy_meta.sample(n=max_files)

    calculated_metrics = list()
    calculated_ref = list()

    output_meta = pd.DataFrame(
        columns=[
            "clip",
            "noise_category",
            "snr",
            "mae",
            "mse",
            "mae_in",
            "mse_in",
            "mosnet",
            "srmr",
            "bsseval",
            "nb_pesq",
            "pesq",
            "sisdr",
            "stoi",
        ]
    )
    ref_metrics_meta = pd.DataFrame(
        columns=[
            "clip",
            "noise_category",
            "snr",
            "mae",
            "mse",
            "mae_in",
            "mse_in",
            "mosnet",
            "srmr",
            "bsseval",
            "nb_pesq",
            "pesq",
            "sisdr",
            "stoi",
        ]
    )
    for row in tqdm.tqdm(noisy_meta.iterrows()):
        file = row[1]["clip"]
        noisy_file = os.path.join(main_set, "noisy_valset_wav", file)
        clean_file = os.path.join(main_set, "clean_valset_wav", file)
        noisy, clean = load_example(
            noisy_file, clean_file, sr=model.config["dataset"]["sample_rate"]
        )

        if len(noisy) < model.input_length:
            continue

        example_metrics, ref_metrics = evaluate_example(
            noisy, clean, model, normalize=normalize
        )

        calculated_metrics.append(example_metrics)
        calculated_ref.append(ref_metrics)
        for key in example_metrics.keys():
            if type(example_metrics[key]) != float:
                example_metrics[key] = example_metrics[key].item()

        for key in ref_metrics.keys():
            if type(ref_metrics[key]) != float:
                ref_metrics[key] = ref_metrics[key].item()

        new_record = pd.DataFrame(
            data=[
                {
                    "clip": file,
                    "noise_category": row[1]["noise_category"],
                    "snr": row[1]["snr"],
                    "mae": example_metrics.get("mae"),
                    "mse": example_metrics.get("mse"),
                    "mae_in": example_metrics.get("mae_in"),
                    "mse_in": example_metrics.get("mse_in"),
                    "mosnet": example_metrics.get("mosnet"),
                    "srmr": example_metrics.get("srmr"),
                    "isr": example_metrics.get("isr"),
                    "sar": example_metrics.get("sar"),
                    "sdr": example_metrics.get("sdr"),
                    "sisdr": example_metrics.get("sisdr"),
                    "pesq": example_metrics.get("pesq"),
                    "nb_pesq": example_metrics.get("nb_pesq"),
                    "stoi": example_metrics.get("stoi"),
                }
            ]
        )
        output_meta = pd.concat((output_meta, new_record), ignore_index=True)

        ref_record = pd.DataFrame(
            data=[
                {
                    "clip": file,
                    "noise_category": row[1]["noise_category"],
                    "snr": row[1]["snr"],
                    "mae": ref_metrics.get("mae"),
                    "mse": ref_metrics.get("mse"),
                    "mae_in": ref_metrics.get("mae_in"),
                    "mse_in": ref_metrics.get("mse_in"),
                    "mosnet": ref_metrics.get("mosnet"),
                    "srmr": ref_metrics.get("srmr"),
                    "isr": ref_metrics.get("isr"),
                    "sar": ref_metrics.get("sar"),
                    "sdr": ref_metrics.get("sdr"),
                    "sisdr": ref_metrics.get("sisdr"),
                    "pesq": ref_metrics.get("pesq"),
                    "nb_pesq": ref_metrics.get("nb_pesq"),
                    "stoi": ref_metrics.get("stoi"),
                }
            ]
        )
        ref_metrics_meta = pd.concat((ref_metrics_meta, ref_record), ignore_index=True)

    evals_path = os.path.join(
        model.config["training"]["path"], "evals", pathlib.Path(main_set).name
    )
    os.makedirs(evals_path, exist_ok=True)
    output_meta.to_csv(os.path.join(evals_path, "pred.csv"))
    ref_metrics_meta.to_csv(os.path.join(evals_path, "ref.csv"))
    mean_metrics = calculate_mean_metrics(calculated_metrics)
    mean_metrics_ref = calculate_mean_metrics(calculated_ref)
    return mean_metrics, mean_metrics_ref


def evaluate_on_testset_wiener(main_set, max_files=None):
    noisy_files = os.listdir(os.path.join(main_set, "noisy_testset_wav"))
    if max_files is not None:
        noisy_files = noisy_files[:max_files]

    calculated_metrics = list()
    for file in tqdm.tqdm(noisy_files):
        noisy_file = os.path.join(main_set, "noisy_testset_wav", file)
        clean_file = os.path.join(main_set, "clean_testset_wav", file)
        noisy, clean = load_example(noisy_file, clean_file, sr=16000)
        example_metrics = evaluate_example_wiener(noisy, clean)
        calculated_metrics.append(example_metrics)

    mean_metrics = calculate_mean_metrics(calculated_metrics)
    return mean_metrics


def predict_example(
    example_noisy, example_clean, model: DenoisingWavenet, calc_metrics
):
    noisy_batch, condition_batch = prepare_batch(example_noisy, model)
    if condition_batch is not None:
        predicted_batch = model.model.predict(
            {"data_input": noisy_batch, "condition_input": condition_batch}, verbose=0
        )[0]
    else:
        predicted_batch = model.model.predict(noisy_batch, verbose=0)[0]
    predicted_batch = predicted_batch[
        :, model.target_padding : model.target_padding + model.target_field_length
    ]
    predicted_vector = predicted_batch.flatten()
    example_clean = example_clean[: predicted_vector.shape[0]]

    clean_batches, _ = prepare_batch(example_clean, model)
    clean_batches = clean_batches[:, model.get_padded_target_field_indices()]
    example_clean = clean_batches.flatten()

    example_metrics = None
    if calc_metrics:
        speechmetrics_res = my_speechmetrics(
            predicted_vector,
            example_clean,
            rate=int(model.config["dataset"]["sample_rate"]),
        )
        example_metrics = speechmetrics_res

    return predicted_vector, example_metrics, example_clean
