import json

from speech_denoising_wavenet.main import training, get_command_line_arguments, load_config
import os
import copy
import tensorflow as tf
import gc
import subprocess


def reset_tf_keras():
    tf.keras.backend.clear_session()
    _ = gc.collect()


base_config = load_config("base_config.json")

os.chdir("../speech_denoising_wavenet")

cla = get_command_line_arguments()
# config = load_config(cla.config)
# training(config, cla)
# print("done")

config = copy.copy(base_config)


def dilations_train():
    for i in [1, 3, 5, 7, 9]:
        config["model"]["dilations"] = i
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/arch/depth/vctk_{dataset}_{i}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {i} done")


def loss_train(config):
    losses = ["l1", "l2", "sdr", "spectrogram", "spectral_convergence", "weighted_spectrogram"]

    def set_weights_to_0(config):
        for loss in losses:
            for out in ["out_1", "out_2"]:
                if loss == "l1":
                    config["training"]["loss"][out]["l1"] = 0
                    config["training"]["loss"][out]["weight"] = 0
                elif loss == "l2":
                    config["training"]["loss"][out]["l2"] = 0
                    config["training"]["loss"][out]["weight"] = 0
                else:
                    config["training"]["loss"][out][loss]["weight"] = 0
        return config

    def enable_loss(config, loss):
        config = set_weights_to_0(config)
        for out in ["out_1", "out_2"]:
            if loss == "l1":
                config["training"]["loss"][out]["l1"] = 1
                config["training"]["loss"][out]["weight"] = 1
            elif loss == "l2":
                config["training"]["loss"][out]["l2"] = 1
                config["training"]["loss"][out]["weight"] = 1
            else:
                config["training"]["loss"][out][loss]["weight"] = 1

        return config

    for loss in losses:
        config = enable_loss(config, loss)
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/hiper/loss/vctk_{dataset}_{loss}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {loss} done")


def opt_train(config):
    for opt in ["adam", "rmsprop", "sgd"]:
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["optimizer"]["type"] = opt
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/hiper/optim/vctk_{dataset}_{opt}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {opt} done")


def lr_train(config):
    for lr in [0.0001, 0.001, 0.01]:
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["optimizer"]["lr"] = lr
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/hiper/lr/vctk_{dataset}_{lr}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {lr} done")


def batch_train(config):
    for batch in [2, 5, 10]:
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["training"]["batch_size"] = batch
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/hiper/lr/vctk_{dataset}_{batch}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {batch} done")


def dropout_train(config):
    for rate in [0.1, 0.3, 0.5]:
        for dataset in ["demand", "esc50", "fma", "art"]:
            config["model"]["dropout"]["use"] = True
            config["model"]["dropout"]["rate"] = rate
            config["dataset"]["path"] = f"data/final_datasets/general/vctk_{dataset}"
            config["training"]["path"] = f"experiments/arch/dropout/vctk_{dataset}_{rate}"
            with open("./temp_config.json", "w") as f:
                json.dump(obj=config, fp=f)
            subprocess.call(["python", "main.py", "--config", "temp_config.json"])
            print(f"{dataset} {rate} done")



def reverb_train(config):
    for dataset in ["demand", "esc50", "fma", "art"]:
        config["dataset"]["path"] = f"data/final_datasets/reverb/vctk_{dataset}_reverb"
        config["training"]["path"] = f"experiments/databased/reverb/vctk_{dataset}_reverb"
        with open("./temp_config.json", "w") as f:
            json.dump(obj=config, fp=f)
        subprocess.call(["python", "main.py", "--config", "temp_config.json"])
        print(f"{dataset} done")

reverb_train(base_config)
#
# config["dataset"]["path"] = "data/final_datasets/general/vctk_demand"
# config["training"]["path"] = "experiments/general/vctk_demand"
#

# training(config, cla)
# print("done")
#
# config["dataset"]["path"] = "data/final_datasets/general/vctk_esc50"
# config["training"]["path"] = "experiments/general/vctk_esc50"
#
# training(config, cla)
# print("done")

# config["dataset"]["path"] = "data/final_datasets/general/vctk_art"
# config["training"]["path"] = "experiments/general/vctk_art"
#
# training(config, cla)
# print("done")
#
# config["dataset"]["path"] = "data/final_datasets/general/vctk_fma"
# config["training"]["path"] = "experiments/general/vctk_fma"
#
# training(config, cla)
# print("done")
#


# config["dataset"]["path"] = "data/final_datasets/lang/cv_demand"
# config["training"]["path"] = "experiments/databased/lang/cv_demand"
#
# training(config, cla)
# print("done")
# tf.keras.backend.clear_session()


# config["dataset"]["path"] = "data/final_datasets/lang/cv_esc50"
# config["training"]["path"] = "experiments/databased/lang/cv_esc50"
#
# training(config, cla)
# print("done")
# reset_tf_keras()

# config["dataset"]["path"] = "data/final_datasets/lang/cv_art"
# config["training"]["path"] = "experiments/databased/lang/cv_art"
#
# training(config, cla)
# print("done")
# reset_tf_keras()

# config["dataset"]["path"] = "data/final_datasets/lang/cv_fma"
# config["training"]["path"] = "experiments/databased/lang/cv_fma"
#
# training(config, cla)
# print("done")
# reset_tf_keras()
