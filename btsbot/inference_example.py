#!/usr/bin/env python3
import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import btsbot
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and use a BTSbot PyTorch model from HuggingFace Hub"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=["convnext", "maxvit"],
        help="Name of the model architecture to use (e.g., 'convnext', 'maxvit')"
    )
    parser.add_argument(
        "--multi_modal",
        action="store_true",
        help="Set this flag if the model is multi-modal"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Name of the pre-training regimen used (e.g., 'imagenet', 'galaxyzoo', 'none')"
    )

    args = parser.parse_args()

    if args.architecture == "convnext":
        arch = "convnext-pico"
    elif args.architecture == "maxvit":
        arch = "maxvit-tiny"
    else:
        raise ValueError("Invalid architecture")

    if args.pretrain == "imagenet":
        pretrain = "in1k"
    elif args.pretrain == "galaxyzoo":
        pretrain = "galaxyzoo"
    elif args.pretrain == "none":
        pretrain = "randinit"
    else:
        raise ValueError("Invalid pre-training regimen")

    if args.multi_modal:
        multi_modal = True
    else:
        multi_modal = False

    link = "nabeelr/BTSbot-" + arch + "-" + pretrain + ("-metadata" if multi_modal else "")
    print(f"Fetching model from HuggingFace Hub: {link}")
    return link, multi_modal


def download_model(link):
    model_name = link.split("/")[-1]
    model_dir = "models" + os.sep + model_name
    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        repo_id=link,
        local_dir=model_dir
    )
    print(f"Model downloaded to {model_dir}")

    return model_dir


def load_model(model_path):
    config_path = model_path + os.sep + "train_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model_name = config["model_name"]
    model_type = getattr(btsbot.architectures, model_name)
    model = model_type(config).to(device)
    model.load_state_dict(
        torch.load(
            model_path + os.sep + "pytorch_model.bin",
            map_location=torch.device('cpu')
        )
    )
    model = model.to(device).eval()

    return model


def run_inference(model, multi_modal):
    cand = pd.read_csv("example_data/usage_candidates.csv", index_col=None)
    labels_tensor = torch.tensor(cand['label'].values, dtype=torch.long)

    metadata_tensor = None
    if multi_modal:
        metadata_cols = [
            "sgscore1", "distpsnr1", "sgscore2", "distpsnr2", "fwhm", "magpsf",
            "sigmapsf", "chipsf", "ra", "dec", "diffmaglim", "ndethist", "nmtchps",
            "age", "days_since_peak", "days_to_peak", "peakmag_so_far", "new_drb",
            "ncovhist", "nnotdet", "chinr", "sharpnr", "scorr", "sky", "maxmag_so_far"
        ]
        metadata_values = cand[metadata_cols].values.astype(np.float32)
        metadata_tensor = torch.tensor(metadata_values)

    triplets_np = np.load("example_data/usage_triplets.npy", mmap_mode='r').astype(np.float32)
    triplets_np = np.transpose(triplets_np, (0, 3, 1, 2))
    triplets_tensor = torch.from_numpy(np.ascontiguousarray(triplets_np))

    dataloader = DataLoader(
        dataset=btsbot.FlexibleDataset(
            images=triplets_tensor,
            metadata=metadata_tensor,
            labels=labels_tensor,
        ),
        batch_size=64, shuffle=False, num_workers=4, pin_memory=(device != 'mps')
    )

    with torch.no_grad():
        batch = next(iter(dataloader))
        if multi_modal:
            images_batch, meta_batch, labels_batch = batch
            images_batch = images_batch.to(device, non_blocking=True)
            meta_batch = meta_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True).cpu().numpy()
            logits = model(image_input=images_batch, metadata_input=meta_batch).to(device)
        else:
            images_batch, labels_batch = batch
            images_batch = images_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True).cpu().numpy()
            logits = model(input_data=images_batch).to(device)

        raw_preds = torch.sigmoid(logits).round().squeeze().cpu().numpy().astype(int)

    print(raw_preds)
    print(labels_batch)
    return


if __name__ == "__main__":
    link, multi_modal = parse_args()

    model_dir = download_model(link)

    model = load_model(model_dir)

    run_inference(model, multi_modal)
