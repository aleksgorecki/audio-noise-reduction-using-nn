import os
from custom_model_evaluation import get_best_checkpoint, evaluate_on_testset
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet.main import load_config
import matplotlib.pyplot as plt
import visualkeras
import tensorflow as tf

model_path = "../speech_denoising_wavenet/experiments/arch/dropout/vctk_demand_0.1"
config = load_config(os.path.join(model_path, "config.json"))
config["training"]["path"] = os.path.join("../speech_denoising_wavenet", config["training"]["path"])
model = DenoisingWavenet(config, load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints")))
tf.keras.utils.plot_model(
    model.model,
    to_file='model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False
)