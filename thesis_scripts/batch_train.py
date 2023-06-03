from speech_denoising_wavenet.main import training, get_command_line_arguments, load_config
import os
import copy


base_config = load_config("base_config.json")

os.chdir("../speech_denoising_wavenet")

cla = get_command_line_arguments()
# config = load_config(cla.config)
# training(config, cla)
# print("done")

config = copy.copy(base_config)

config["dataset"]["path"] = "data/final_datasets/general/vctk_esc50"
config["training"]["path"] = "experiments/general/vctk_esc50"

training(config, cla)
print("done")


config = copy.copy(base_config)

config["dataset"]["path"] = "data/final_datasets/general/vctk_fma"
config["training"]["path"] = "experiments/general/vctk_fma"

training(config, cla)
print("done")


config = copy.copy(base_config)

config["dataset"]["path"] = "data/final_datasets/general/vctk_art"
config["training"]["path"] = "experiments/general/vctk_art"

training(config, cla)
print("done")
