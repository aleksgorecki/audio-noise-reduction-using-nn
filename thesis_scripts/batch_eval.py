import os
from custom_model_evaluation import get_best_checkpoint, evaluate_on_testset
from speech_denoising_wavenet.models import DenoisingWavenet
from speech_denoising_wavenet.main import load_config


dataset_path = "../speech_denoising_wavenet/data/final_datasets/general/vctk_demand"
model_session_path = "../speech_denoising_wavenet/experiments/general/vctk_demand"


def eval_and_dump(dataset_path, model_path):
    config = load_config(os.path.join(model_path, "config.json"))
    config["training"]["path"] = os.path.join("../speech_denoising_wavenet", config["training"]["path"])
    model = DenoisingWavenet(config, load_checkpoint=get_best_checkpoint(os.path.join(model_path, "checkpoints")))
    metrics, ref = evaluate_on_testset(dataset_path, model, max_files=None)
    print("ref: ")
    print(ref)
    print("pred: ")
    print(metrics)


if __name__ == "__main__":
    eval_and_dump(dataset_path, model_session_path)
