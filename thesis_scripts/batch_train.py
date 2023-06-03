from speech_denoising_wavenet.main import training, get_command_line_arguments, load_config
import os


base_config = load_config("base_config.json")

os.chdir("../speech_denoising_wavenet")

cla = get_command_line_arguments()
config = load_config(cla.config)
training(config, cla)
print("done")
