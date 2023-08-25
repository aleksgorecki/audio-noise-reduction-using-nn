import os
from model_management.evaluation import (
    get_best_checkpoint,
    prepare_batch,
)
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet.main import load_config
import librosa
import matplotlib.pyplot as plt
import soundfile as sf


def plot_waveform(
    data, fs, title=None, show=True, save_path=None, fig=None, ax=None, eng=False
):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    librosa.display.waveshow(y=data, sr=fs, ax=ax)
    ax.set(
        title=title,
    )
    if eng:
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
    else:
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
    if show:
        plt.show()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(save_path)


model_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/general/vctk_art/"
file_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/data/final_datasets/reverb/vctk_art_reverb/noisy_valset_wav/p226_white_15db_63.wav"
clean_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/data/final_datasets/reverb/vctk_art_reverb/clean_valset_wav/p226_white_15db_63.wav"
config = load_config(os.path.join(model_path, "config.json"))
config["training"]["path"] = os.path.join(
    "../speech_denoising_wavenet", config["training"]["path"]
)
model = DenoisingWavenet(
    config, load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints"))
)

example_noisy = librosa.core.load(file_path, sr=16000)[0]
example_clean = librosa.core.load(clean_path, sr=16000)[0]

example_clean = librosa.util.normalize(example_clean)
example_noisy = librosa.util.normalize(example_noisy)

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


predicted_vector = librosa.util.normalize(predicted_vector)

plot_waveform(data=example_noisy, fs=16000, show=False)
plot_waveform(data=example_clean, fs=16000, show=False)
plot_waveform(data=predicted_vector, fs=16000, show=False)

sf.write(
    "/home/aleks/Desktop/reverbdenoised.wav", samplerate=16000, data=predicted_vector
)


model_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/experiments/databased/reverb/vctk_art_reverb"
file_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/data/final_datasets/reverb/vctk_art_reverb/noisy_valset_wav/p226_white_15db_63.wav"
clean_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/data/final_datasets/reverb/vctk_art_reverb/clean_valset_wav/p226_white_15db_63.wav"
config = load_config(os.path.join(model_path, "config.json"))
config["training"]["path"] = os.path.join(
    "../speech_denoising_wavenet", config["training"]["path"]
)
model = DenoisingWavenet(
    config, load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints"))
)

example_noisy = librosa.core.load(file_path, sr=16000)[0]
example_clean = librosa.core.load(clean_path, sr=16000)[0]

example_clean = librosa.util.normalize(example_clean)
example_noisy = librosa.util.normalize(example_noisy)

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


predicted_vector = librosa.util.normalize(predicted_vector)

plot_waveform(data=example_noisy, fs=16000, show=False)
plot_waveform(data=example_clean, fs=16000, show=False)
plot_waveform(data=predicted_vector, fs=16000, show=False)

sf.write(
    "/home/aleks/Desktop/reverbdenoised.wav", samplerate=16000, data=predicted_vector
)


plt.show()
