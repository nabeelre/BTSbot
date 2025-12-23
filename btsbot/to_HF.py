from btsbot.from_HF import get_HF_model_link
from btsbot import architectures
from huggingface_hub import HfApi
import argparse
import torch
import json
import os


def prep_config(model_dir: str):
    report_path = os.path.join(model_dir, "report.json")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with open(report_path, "r") as f:
        report = json.load(f)

    config = report["train_config"]

    config_path = os.path.join(model_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return config


def prep_model(model_dir: str, config: dict):
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_name = config["model_name"]
    model_type = getattr(architectures, model_name)
    model = model_type(config)
    model.load_state_dict(
        torch.load(
            model_path, map_location=torch.device('cpu')
        )
    )

    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

    return


def create_gitattributes(model_dir: str):
    gitattributes_path = os.path.join(model_dir, ".gitattributes")
    with open(gitattributes_path, "w") as f:
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.bin filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    return


def create_model_card(model_dir: str, arch: str, multi_modal: bool, pretrain: str):
    model_card = f"""---
library_name: pytorch
tags:
- vision
- image-classification
- pytorch
license: mit
base_model: {get_HF_basemodel(arch, pretrain)}
---

# BTSbot

This is a {arch} fine-tuned for classifying images from the
Zwicky Transient Facility (ZTF) observatory.
[Rehemtulla et al. 2024](https://arxiv.org/abs/2401.15167) originally introduced
`BTSbot` and its classification task, and
[Rehemtulla et al. 2025](https://arxiv.org/abs/2512.11957) performed
architecture and pre-training benchmarking on this `BTSbot` image classification task.

**Base Model**:
[{get_HF_basemodel(arch, pretrain)}](https://huggingface.co/{get_HF_basemodel(arch, pretrain)})

## Usage

Easily install the btsbot package and load this model with:

```python
git clone https://github.com/nabeelre/BTSbot.git
cd BTSbot
pip install -e .

import btsbot
model = btsbot.load_HF_model(
    architecture="{arch}", multi_modal={multi_modal}, pretrain="{pretrain}"
)
```

Also see
[`BTSbot/btsbot/inference_example.py`](https://github.com/nabeelre/BTSbot/blob/main/btsbot/inference_example.py).

## Citation

If you use this model, please cite:
""" + """
```bibtex
@ARTICLE{Rehemtulla+2025,
       author = {{Rehemtulla}, Nabeel and {Miller}, Adam A. and {Walmsley}, Mike
                 and {Shah}, Ved G. and {Jegou du Laz}, Theophile and
                 {Coughlin}, Michael W. and {Sasli}, Argyro and
                 {Bloom}, Joshua and {Fremling}, Christoffer and
                 {Graham}, Matthew J. and {Groom}, Steven L. and {Hale}, David and
                 {Mahabal}, Ashish A. and {Perley}, Daniel A. and
                 {Purdum}, Josiah and {Rusholme}, Ben and {Sollerman}, Jesper and
                 {Kasliwal}, Mansi M.},
        title = "{Pre-training vision models for the classification of alerts from
                  wide-field time-domain surveys}",
      journal = {arXiv e-prints},
     keywords = {Instrumentation and Methods for Astrophysics,
                 Computer Vision and Pattern Recognition},
         year = 2025,
        month = dec,
          eid = {arXiv:2512.11957},
        pages = {arXiv:2512.11957},
          doi = {10.48550/arXiv.2512.11957},
archivePrefix = {arXiv},
       eprint = {2512.11957},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025arXiv251211957R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```""" + """

## License

This model is released under the MIT License.

## Repository

For more information, see the [BTSbot GitHub repository](https://github.com/nabeelre/BTSbot).
"""

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card.strip() + "\n")


def config_to_params(config: dict):
    multi_modal = config["model_name"] == "frozen_fusion"
    image_config = config["image_model_config"] if multi_modal else config

    if "maxvit" in image_config['model_kind']:
        architecture = "maxvit"
    elif "convnext" in image_config['model_kind']:
        architecture = "convnext"
    else:
        raise ValueError("Couldn't understand architecture")

    if "mwalmsley" in image_config["model_kind"]:
        pretrain = "galaxyzoo"
    elif not image_config.get("pretrained", True):
        pretrain = "randinit"
    elif "in1k" in image_config["model_kind"]:
        pretrain = "imagenet"
    else:
        raise ValueError("Couldn't understand pre-training regimen")

    return (architecture, multi_modal, pretrain)


def get_HF_basemodel(arch: str, pretrain: str):
    if arch == "maxvit":
        if pretrain == "galaxyzoo":
            return "mwalmsley/baseline-encoder-regression-maxvit_tiny"
        elif pretrain in ["imagenet", "randinit"]:
            return "timm/maxvit_tiny_rw_224.sw_in1k"
    elif arch == "convnext":
        if pretrain == "galaxyzoo":
            return "mwalmsley/zoobot-encoder-convnext_pico"
        elif pretrain in ["imagenet", "randinit"]:
            return "timm/convnext_pico.d1_in1k"

    raise ValueError(f"Invalid architecture: {arch} or pre-training regimen: {pretrain}")


def upload_model_to_hf(model_dir: str):
    config_path = os.path.join(model_dir, "train_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    arch, multi_modal, pretrain = config_to_params(config)

    hf_link = get_HF_model_link(arch, multi_modal, pretrain)

    print("Parsed model at", model_dir, "to")
    print(f"{arch} pre-trained on {pretrain} and {'not ' if not multi_modal else ''}multi-modal")
    print("Uploading model to HuggingFace repository:", hf_link)

    # Create HF repo if needed
    api = HfApi()
    try:
        api.create_repo(repo_id=hf_link, repo_type="model", exist_ok=True)
        print(f"Repository {hf_link} is ready")
    except Exception as e:
        print(f"Note: Repository creation/verification: {e}")

    files_to_upload = [
        ("pytorch_model.bin", os.path.join(model_dir, "pytorch_model.bin")),
        ("train_config.json", os.path.join(model_dir, "train_config.json")),
        ("README.md", os.path.join(model_dir, "README.md"))
    ]

    for filename, filepath in files_to_upload:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required file not found: {filepath}")
        print(f"Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=hf_link,
            repo_type="model"
        )

    print("Successfully uploaded model to HuggingFace Hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare a BTSbot model for uploading to the HuggingFace Hub"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory containing best_model.pth and report.json"
    )

    args = parser.parse_args()
    print("Attempting to upload model to HuggingFace Hub from:", args.model_dir)

    config = prep_config(args.model_dir)
    arch, multi_modal, pretrain = config_to_params(config)
    prep_model(args.model_dir, config)
    # create_gitattributes(args.model_dir)
    create_model_card(args.model_dir, arch, multi_modal, pretrain)
    upload_model_to_hf(args.model_dir)
