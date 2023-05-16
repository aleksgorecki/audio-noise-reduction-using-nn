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
# sys.path.append("speech_denoising_wavenet")
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet import util
import time
import tqdm
import scipy


METRICS = ["mosnet", "srmr", "bsseval", "nb_pesq", "pesq", "sisdr", "stoi"]
my_speechmetrics = speechmetrics.load(METRICS, window=None)


def get_latest_checkpoint(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    latest_checkpoint = max(checkpoints, key=lambda x: int(x[11:16]))
    return os.path.join(checkpoints_dir, latest_checkpoint)


def get_best_checkpoint(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    best_checkpoint = min(checkpoints, key=lambda x: float(x[17:22]))
    return os.path.join(checkpoints_dir, best_checkpoint)


def append_examples_metadata():
    pass


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


# def assemble_batch(model, input, condition_input, batch_size):
#     if len(input['noisy']) < model.receptive_field_length:
#         raise ValueError('Input is not long enough to be used with this model.')
#
#     padding_length = int(model.receptive_field_length / 2)
#     # input['noisy'] = np.concatenate((input['noisy'], np.zeros(padding_length, dtype=input['noisy'].dtype)))
#     num_output_samples = input['noisy'].shape[0] - (model.receptive_field_length - 1)
#     num_fragments = int(np.ceil(num_output_samples / model.target_field_length))
#     num_batches = int(np.ceil(num_fragments / batch_size))
#
#     denoised_output = []
#     noise_output = []
#     num_pad_values = 0
#     fragment_i = 0
#     for batch_i in tqdm.tqdm(range(0, num_batches)):
#
#         if batch_i == num_batches - 1:  # If its the last batch'
#             batch_size = num_fragments - batch_i * batch_size
#
#         condition_batch = np.array([condition_input, ] * batch_size, dtype='uint8')
#         input_batch = np.zeros((batch_size, int(model.input_length)))
#
#         # Assemble batch
#         for batch_fragment_i in range(0, batch_size):
#
#             if fragment_i + model.target_field_length > num_output_samples:
#                 remainder = input['noisy'][fragment_i:]
#                 current_fragment = np.zeros((int(model.input_length),))
#                 current_fragment[:remainder.shape[0]] = remainder
#                 num_pad_values = int(model.input_length - remainder.shape[0])
#             else:
#                 current_fragment = input['noisy'][fragment_i:fragment_i + int(model.input_length)]
#
#             input_batch[batch_fragment_i, :] = current_fragment
#             fragment_i += model.target_field_length
#
#         denoised_output_fragments = model.denoise_batch({'data_input': input_batch, 'condition_input': condition_batch})
#
#         if type(denoised_output_fragments) is list:
#             noise_output_fragment = denoised_output_fragments[1]
#             denoised_output_fragment = denoised_output_fragments[0]
#
#         denoised_output_fragment = denoised_output_fragment[:,
#                                    model.target_padding: model.target_padding + model.target_field_length]
#         denoised_output_fragment = denoised_output_fragment.flatten().tolist()
#
#         if noise_output_fragment is not None:
#             noise_output_fragment = noise_output_fragment[:,
#                                     model.target_padding: model.target_padding + model.target_field_length]
#             noise_output_fragment = noise_output_fragment.flatten().tolist()
#
#         if type(denoised_output_fragments) is float:
#             denoised_output_fragment = [denoised_output_fragment]
#         if type(noise_output_fragment) is float:
#             noise_output_fragment = [noise_output_fragment]
#
#         denoised_output = denoised_output + denoised_output_fragment
#         noise_output = noise_output + noise_output_fragment
#
#     denoised_output = np.array(denoised_output)
#     noise_output = np.array(noise_output)
#
#     if num_pad_values != 0:
#         denoised_output = denoised_output[:-num_pad_values]
#         noise_output = noise_output[:-num_pad_values]



def prepare_batch(example, model: DenoisingWavenet):

    #example = np.concatenate((np.zeros(model.half_receptive_field_length, dtype=example.dtype), example, np.zeros(model.half_receptive_field_length, dtype=example.dtype)))

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
    if not bool(model.config["model"].get("no_conditioning")):
        condition_input = util.binary_encode(int(0), 29)[0]
        condition_batch = np.array([condition_input, ] * len(batch), dtype='uint8')
    else:
        condition_batch = None
    return batch, condition_batch
    # for i, target_field_window_start in enumerate(list(range(0, num_output_samples, target_field_len))):
    #     if int(target_field_window_start + input_len) > num_output_samples:
    #         batch[i][0:example.shape[0] - target_field_window_start] = example[target_field_window_start:]
    #         continue
    #     batch[i] = example[target_field_window_start: target_field_window_start + input_len]



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

    noisy_batch, condition_batch = prepare_batch(example_noisy, model)
    if condition_batch is not None:
        predicted_batch = model.model.predict({"data_input": noisy_batch, "condition_input": condition_batch}, verbose=0)[0]
    else:
        predicted_batch = model.model.predict(noisy_batch, verbose=0)[0]
    #predicted_vector = predicted_batch.flatten()

    predicted_vector = predicted_batch.flatten()
    clean_batches, _ = prepare_batch(example_clean, model)
    clean_batches = clean_batches[:, model.get_padded_target_field_indices()]
    example_clean = clean_batches.flatten()

    sf.write("./test_clean.wav", data=example_clean, samplerate=16000)
    sf.write("./test_noisy.wav", data=example_noisy, samplerate=16000)
    sf.write("./test_batches.wav", data=noisy_batch.flatten(), samplerate=16000)
    sf.write("./test_denoised.wav", data=predicted_vector, samplerate=16000)

    # example_clean = example_clean[len(example_clean) - len(predicted_vector):]
    sf.write("./test_clean_modified.wav", data=example_clean, samplerate=16000)

    speechmetrics_res = my_speechmetrics(predicted_vector, example_clean, rate=int(16000))
    example_metrics = speechmetrics_res

    return example_metrics


def evaluate_example_wiener(example_noisy, example_clean):
    # sliced_clean = slice_example(example_clean, int(model.input_length))
    # for batch in noisy_batch:
        # _, _, spec = scipy.signal.stft(batch, 16000)
        # spec_denoised = scipy.signal.wiener(np.abs(spec))
        # denoised.append(scipy.signal.istft(spec_denoised, 16000))
    predicted_vector = scipy.signal.wiener(example_noisy).flatten()
    example_clean = example_clean[: predicted_vector.shape[0]]

    # sf.write("./test_noisy.wav", data=example_noisy, samplerate=16000)
    # sf.write("./test_denoised.wav", data=predicted_vector, samplerate=16000)
    # sf.write("./test_clean.wav", data=example_clean, samplerate=16000)

    speechmetrics_res = my_speechmetrics(predicted_vector, example_clean, rate=int(16000))
    example_metrics = speechmetrics_res

    return example_metrics


def evaluate_on_testset(main_set, model: DenoisingWavenet, max_files=None):

    noisy_files = os.listdir(os.path.join(main_set, "noisy_testset_wav"))
    if max_files is not None:
        noisy_files = noisy_files[:max_files]

    calculated_metrics = list()

    for file in tqdm.tqdm(noisy_files):
        noisy_file = os.path.join(main_set, "noisy_testset_wav", file)
        clean_file = os.path.join(main_set, "clean_testset_wav", file)
        noisy, clean = load_example(noisy_file, clean_file, sr=model.config["dataset"]["sample_rate"])
        example_metrics = evaluate_example(noisy, clean, model)
        calculated_metrics.append(example_metrics)


    mean_metrics = calculate_mean_metrics(calculated_metrics)
    return mean_metrics


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


def predict_example(example_noisy, example_clean, model: DenoisingWavenet, calc_metrics):
    noisy_batch, condition_batch = prepare_batch(example_noisy, model)
    if condition_batch is not None:
        predicted_batch = model.model.predict({"data_input": noisy_batch, "condition_input": condition_batch}, verbose=0)[0]
    else:
        predicted_batch = model.model.predict(noisy_batch, verbose=0)[0]
    predicted_vector = predicted_batch.flatten()
    example_clean = example_clean[: predicted_vector.shape[0]]

    clean_batches, _ = prepare_batch(example_clean, model)
    clean_batches = clean_batches[:, model.get_padded_target_field_indices()]
    example_clean = clean_batches.flatten()

    example_metrics = None
    if calc_metrics:
        speechmetrics_res = my_speechmetrics(predicted_vector, example_clean, rate=int(model.config["dataset"]["sample_rate"]))
        example_metrics = speechmetrics_res

    return predicted_vector, example_metrics, example_clean


if __name__ == "__main__":
    #os.chdir("/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet")
    dataset = "speech_denoising_wavenet/data/NSDTSEA/"

    # config_path = "sessions/001/config.json"
    # checkpoint_path = "sessions/001/checkpoints/checkpoint.00144.hdf5"
    #
    # with open(config_path, "r") as f:
    #     config = json.load(f)
    #     model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
    #     metrics = evaluate_on_testset(dataset, model)
    #     print(metrics)

    config_path = "speech_denoising_wavenet/sessions/default_5/config.json"
    checkpoint_path = "speech_denoising_wavenet/sessions/default_5/checkpoints/checkpoint.00044-0.230.hdf5"

    # start = time.time()
    # metrics = evaluate_on_testset_wiener(dataset, max_files=100)
    # end = time.time()
    # print(end - start, " s")
    # print(metrics)

    with open(config_path, "r") as f:
        config = json.load(f)

        config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
        model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
        start = time.time()
        metrics = evaluate_on_testset(dataset, model, max_files=100)
        end = time.time()
        print(end - start, " s")
        print(metrics)

    config_path = "speech_denoising_wavenet/sessions/default_5_sdr/config.json"
    checkpoint_path = "speech_denoising_wavenet/sessions/default_5_sdr/checkpoints/checkpoint.00096--16.008.hdf5"

    # start = time.time()
    # metrics = evaluate_on_testset_wiener(dataset, max_files=100)
    # end = time.time()
    # print(end - start, " s")
    # print(metrics)

    with open(config_path, "r") as f:
        config = json.load(f)

        config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
        model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
        start = time.time()
        metrics = evaluate_on_testset(dataset, model, max_files=100)
        end = time.time()
        print(end - start, " s")
        print(metrics)

    # config_path = "speech_denoising_wavenet/sessions/001/config.json"
    # checkpoint_path = "speech_denoising_wavenet/sessions/001/checkpoints/checkpoint.00144.hdf5"
    #
    # with open(config_path, "r") as f:
    #     config = json.load(f)
    #
    #     config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
    #     model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
    #     start = time.time()
    #     metrics = evaluate_on_testset(dataset, model, max_files=100)
    #     end = time.time()
    #     print(end - start, " s")
    #     print(metrics)
    #
    #
    # config_path = "speech_denoising_wavenet/sessions/default_9_combined_spectrogram_weighted/config.json"
    # checkpoint_path = "speech_denoising_wavenet/sessions/default_9_combined_spectrogram_weighted/checkpoints/checkpoint.00068-0.282.hdf5"
    #
    # with open(config_path, "r") as f:
    #     config = json.load(f)
    #
    #     config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
    #     model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
    #     start = time.time()
    #     metrics = evaluate_on_testset(dataset, model, max_files=100)
    #     end = time.time()
    #     print(end - start, " s")
    #     print(metrics)
    #
    # config_path = "speech_denoising_wavenet/sessions/default_5_new_config/config.json"
    # checkpoint_path = "speech_denoising_wavenet/sessions/default_5_new_config/checkpoints/checkpoint.00045-0.016.hdf5"
    #
    # with open(config_path, "r") as f:
    #     config = json.load(f)
    #
    #     config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
    #     model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
    #     start = time.time()
    #     metrics = evaluate_on_testset(dataset, model, max_files=100)
    #     end = time.time()
    #     print(end - start, " s")
    #     print(metrics)
    #
    # start = time.time()
    # metrics = evaluate_on_testset_wiener(dataset)
    # end = time.time()
    # print(end - start, " s")
    # print(metrics)

    # config_path = "speech_denoising_wavenet/sessions/default_5/config.json"
    # checkpoint_path = "speech_denoising_wavenet/sessions/default_5/checkpoints/checkpoint.00044-0.230.hdf5"
    #
    # with open(config_path, "r") as f:
    #     config = json.load(f)
    #
    #     config["training"]["path"] = os.path.join("speech_denoising_wavenet", config["training"]["path"])
    #
    #     model = DenoisingWavenet(config, load_checkpoint=checkpoint_path)
    #     metrics = evaluate_on_testset(dataset, model)
    #     print(metrics)
