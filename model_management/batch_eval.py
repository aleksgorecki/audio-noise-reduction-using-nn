import os

import pandas

from model_management.evaluation import (
    get_best_checkpoint,
    evaluate_on_testset,
    prepare_batch,
)
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet.main import load_config
import time
import numpy as np


def eval_and_dump(dataset_path, model_path):
    config = load_config(os.path.join(model_path, "config.json"))
    config["training"]["path"] = os.path.join(
        "../speech_denoising_wavenet", config["training"]["path"]
    )
    model = DenoisingWavenet(
        config,
        load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints")),
    )
    metrics, ref = evaluate_on_testset(dataset_path, model, max_files=None)
    print("ref: ")
    print(ref)
    print("pred: ")
    print(metrics)


def eval_time_abstract(dataset_path, model_path):
    config = load_config(os.path.join(model_path, "config.json"))
    config["training"]["path"] = os.path.join(
        "../speech_denoising_wavenet", config["training"]["path"]
    )
    model = DenoisingWavenet(
        config,
        load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints")),
    )
    times = []
    for i in range(0, 100, 1):
        sig = np.random.random(16000 * 3)
        batch = prepare_batch(sig, model)[0]
        t1 = time.time()
        model.model.predict(batch, verbose=False)
        t2 = time.time()
        times.append(t2 - t1)
    df = pandas.DataFrame(data={"pred_times": times})
    df.to_csv(os.path.join(model.config["training"]["path"], "pred_times.csv"))


def general_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
        )
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/general/vctk_{dataset}"
        )
        eval_and_dump(dataset_path, model_session_path)


def language_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/lang/cv_{dataset}"
        )
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/databased/lang/cv_{dataset}"
        )
        eval_and_dump(dataset_path, model_session_path)
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/lang/cv_{dataset}"
        )
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/general/vctk_{dataset}"
        )
        eval_and_dump(dataset_path, model_session_path)
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
        )
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/databased/lang/cv_{dataset}"
        )
        eval_and_dump(dataset_path, model_session_path)


def reverb_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = f"../speech_denoising_wavenet/data/final_datasets/reverb/vctk_{dataset}_reverb"
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/general/vctk_{dataset}"
        )
        eval_and_dump(dataset_path, model_session_path)
    for dataset in ["demand", "esc50", "fma", "art"]:
        dataset_path = f"../speech_denoising_wavenet/data/final_datasets/reverb/vctk_{dataset}_reverb"
        model_session_path = f"../speech_denoising_wavenet/experiments/databased/reverb/vctk_{dataset}_reverb"
        eval_and_dump(dataset_path, model_session_path)


def approx_eval():
    for dataset in ["demand", "esc50", "fma"]:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
        )
        model_session_path = f"../speech_denoising_wavenet/experiments/general/vctk_art"
        eval_and_dump(dataset_path, model_session_path)


def depth_eval():
    for dataset in ["fma", "art"]:
        for i in [1, 3, 5]:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = (
                f"../speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}"
            )
            eval_and_dump(dataset_path, model_session_path)


def dropout_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = f"../speech_denoising_wavenet/experiments/arch/dropout/vctk_{dataset}_{rate}"
            eval_and_dump(dataset_path, model_session_path)


def loss_eval():
    losses = [
        "l1",
        "l2",
        "sdr",
        "spectrogram",
        "spectral_convergence",
        "weighted_spectrogram",
    ]
    for dataset in ["demand", "esc50", "fma", "art"]:
        for loss in losses:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = f"../speech_denoising_wavenet/experiments/hiper/loss/vctk_{dataset}_{loss}"
            eval_and_dump(dataset_path, model_session_path)


def loss_eval_pesq():
    losses = [
        "l1",
        "l2",
        "sdr",
        "spectrogram",
        "spectral_convergence",
        "weighted_spectrogram",
    ]
    for loss in losses:
        dataset_path = (
            f"../speech_denoising_wavenet/data/final_datasets/general/vctk_art_long"
        )
        model_session_path = (
            f"../speech_denoising_wavenet/experiments/hiper/loss/vctk_art_{loss}"
        )
        eval_and_dump(dataset_path, model_session_path)


def opt_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        for opt in ["adam", "rmsprop", "sgd"]:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = f"../speech_denoising_wavenet/experiments/hiper/optim/vctk_{dataset}_{opt}"
            eval_and_dump(dataset_path, model_session_path)


def lr_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        for lr in [0.01, 0.001, 0.0001, 0.00001]:
            dataset_path = (
                f"../speech_denoising_wavenet/data/final_datasets/general/v"
                f"ctk_{dataset}"
            )
            model_session_path = (
                f"../speech_denoising_wavenet/experiments/hiper/lr/vctk_{dataset}_{lr}"
            )
            eval_and_dump(dataset_path, model_session_path)


def batch_eval():
    for dataset in ["demand", "esc50", "fma", "art"]:
        for batch in [2, 5, 10]:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = f"../speech_denoising_wavenet/experiments/hiper/batch/vctk_{dataset}_{batch}"
            eval_and_dump(dataset_path, model_session_path)


def eval_time():
    for dataset in ["demand", "esc50", "fma", "art"]:
        for i in [1, 3, 5, 7, 9]:
            dataset_path = f"../speech_denoising_wavenet/data/final_datasets/general/vctk_{dataset}"
            model_session_path = (
                f"../speech_denoising_wavenet/experiments/arch/depth/vctk_{dataset}_{i}"
            )
            eval_time_abstract(dataset_path, model_session_path)
