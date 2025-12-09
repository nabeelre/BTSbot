import os
import json
import torch
import btsbot
from huggingface_hub import snapshot_download

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def validate_model_params(architecture: str, multi_modal: bool, pretrain: str):
    if architecture == "convnext":
        architecture = "convnext-pico"
    elif architecture == "maxvit":
        architecture = "maxvit-tiny"
    else:
        raise ValueError(f"Invalid architecture: {architecture}")

    if pretrain == "imagenet":
        pretrain = "in1k"
    elif pretrain not in ["galaxyzoo", "randinit"]:
        raise ValueError(f"Invalid pre-training regimen: {pretrain}")

    return architecture, multi_modal, pretrain


def get_HF_model_link(architecture: str, multi_modal: bool, pretrain: str) -> str:
    architecture, multi_modal, pretrain = validate_model_params(architecture, multi_modal, pretrain)
    return "nabeelr/BTSbot-" + architecture + "-" + pretrain + ("-metadata" if multi_modal else "")


def get_local_model_dir(architecture: str, multi_modal: bool, pretrain: str) -> str:
    architecture, multi_modal, pretrain = validate_model_params(architecture, multi_modal, pretrain)
    model_name = "BTSbot-" + architecture + "-" + pretrain + ("-metadata" if multi_modal else "")
    return os.path.join("models", model_name)


def download_HF_model(architecture: str, multi_modal: bool, pretrain: str):
    HF_link = get_HF_model_link(architecture, multi_modal, pretrain)

    print(f"Fetching model from HuggingFace Hub: {HF_link}")
    model_name = HF_link.split("/")[-1]
    model_dir = os.path.join("models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=HF_link,
        local_dir=model_dir
    )
    print(f"Model downloaded to {model_dir}")
    return


def load_HF_model(architecture: str, multi_modal: bool, pretrain: str):
    model_dir = get_local_model_dir(architecture, multi_modal, pretrain)

    required_files = ["pytorch_model.bin", "train_config.json"]
    if not all(os.path.isfile(os.path.join(model_dir, f)) for f in required_files):
        print("Model files not present; downloading model...")
        download_HF_model(architecture, multi_modal, pretrain)

    config_path = os.path.join(model_dir, "train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    model_type = getattr(btsbot.architectures, model_name)
    model = model_type(config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "pytorch_model.bin"),
            map_location=torch.device('cpu')
        )
    )

    return model
