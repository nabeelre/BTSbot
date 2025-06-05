import umap.umap_ as umap
import pandas as pd
import numpy as np

import architectures_torch as architectures
from torch_utils import FlexibleDataset
from torch.utils.data import DataLoader
from os import path
import torch
import json
import tqdm

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

standard_metadata_cols = [
    "sgscore1", "distpsnr1", "sgscore2", "distpsnr2",
    "fwhm", "magpsf", "sigmapsf", "chipsf", "ra",
    "dec", "diffmaglim", "ndethist", "nmtchps", "age",
    "days_since_peak", "days_to_peak", "peakmag_so_far",
    "new_drb", "ncovhist", "nnotdet", "chinr", "sharpnr",
    "scorr", "sky", "maxmag_so_far"
]


def get_torch_embedding(model_dir, cand_path, trips_path=None, batch_size=1024,
                        metadata_cols=None, validate_model=True, config=None,
                        umap_seed=2):
    # Check for multiple GPUs
    multi_gpu = False
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        multi_gpu = True

    need_triplets = trips_path is not None
    need_metadata = metadata_cols is not None

    cand = pd.read_csv(cand_path, index_col=None)
    labels_tensor = torch.tensor(cand['label'].values, dtype=torch.long)

    triplets_tensor = None
    if need_triplets:
        triplets_np = np.load(trips_path).astype(np.float32)
        triplets_np = np.transpose(triplets_np, (0, 3, 1, 2))
        triplets_tensor = torch.from_numpy(triplets_np.copy())

    metadata_tensor = None
    if need_metadata:
        metadata_values = cand[metadata_cols].values.astype(np.float32)
        metadata_tensor = torch.tensor(metadata_values)

    if multi_gpu:
        batch_size = batch_size // torch.cuda.device_count()

    dataloader = DataLoader(
        dataset=FlexibleDataset(
            images=triplets_tensor,
            metadata=metadata_tensor,
            labels=labels_tensor,
        ), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device != 'mps')
    )

    if config is None:
        with open(model_dir + "report.json") as report:
            config = json.load(report)['train_config']

    try:
        if config['model_name'] == "SwinV2_t":
            config['model_name'] = "SwinV2"
        model_type = getattr(architectures, config['model_name'])
    except AttributeError:
        print(f"Could not find model of name {config['model_name']}")
        exit(0)
    model = model_type(config).to(device)
    model.load_state_dict(
        torch.load(
            path.join(model_dir, "best_model.pth"),
            map_location=torch.device('cpu')
        )
    )
    model = model.to(device).eval()

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    all_embs = []
    all_raw_preds = []

    with torch.no_grad():
        if need_triplets and need_metadata:
            if validate_model:
                for batch in tqdm.tqdm(dataloader):
                    images_batch, meta_batch, _ = batch
                    images_batch = images_batch.to(device, non_blocking=True)
                    meta_batch = meta_batch.to(device, non_blocking=True)
                    raw_preds = torch.sigmoid(
                        model(images_batch, meta_batch).cpu()
                    )
                    all_raw_preds.append(raw_preds)

            if multi_gpu:
                # If using multiple GPUs, model is wrapped in DataParallel
                # access the model with .module
                emb_model = model.module
            else:
                emb_model = model

            # Remove final layer(s) of the model to get embeddings
            if config['model_name'] == "mm_SwinV2":
                pass
                # emb_model.combined_head = nn.Sequential(
                #     nn.Linear(
                #         emb_model.combined_head[0].in_features,
                #         emb_model.combined_head[0].out_features
                #     ),
                #     nn.ReLU()
                # )
            elif config['model_name'] == "mm_MaxViT":
                print(emb_model.combined_head)
                emb_model.combined_head = emb_model.combined_head[:4]
                print(emb_model.combined_head)

            emb_model = emb_model.to(device).eval()
            if multi_gpu:
                emb_model = torch.nn.DataParallel(emb_model)

            for batch in tqdm.tqdm(dataloader):
                images_batch, meta_batch, _ = batch
                images_batch = images_batch.to(device, non_blocking=True)
                meta_batch = meta_batch.to(device, non_blocking=True)
                embs = emb_model(
                    images_batch, meta_batch
                )

                all_embs.append(embs.cpu())
        elif need_triplets:
            if validate_model:
                for batch in tqdm.tqdm(dataloader):
                    images_batch, _ = batch
                    images_batch = images_batch.to(device, non_blocking=True)
                    raw_preds = torch.sigmoid(
                        model(images_batch).cpu()
                    )
                    all_raw_preds.append(raw_preds)

            if multi_gpu:
                emb_model = model.module
            else:
                emb_model = model

            if config['model_name'] == "SwinV2":
                pass
                # emb_model.swin.head = nn.Sequential(
                #     nn.Linear(
                #         emb_model.swin.head[0].in_features,
                #         emb_model.swin.head[0].out_features
                #     ),
                #     nn.Linear(
                #         emb_model.swin.head[1].in_features,
                #         emb_model.swin.head[1].out_features
                #     ),
                #     nn.ReLU()
                # )
            elif config['model_name'] == "MaxViT":
                print(emb_model.maxvit.head)
                emb_model.maxvit.head = emb_model.maxvit.head[:5]
                print(emb_model.maxvit.head)
            elif config['model_name'] == "ConvNeXt":
                print(emb_model.convnext.head)
                emb_model.convnext.head = emb_model.convnext.head[:7]
                print(emb_model.convnext.head)

            emb_model = emb_model.to(device).eval()
            if multi_gpu:
                emb_model = torch.nn.DataParallel(emb_model)

            for batch in tqdm.tqdm(dataloader):
                images_batch, _ = batch
                images_batch = images_batch.to(device, non_blocking=True)
                embs = emb_model(images_batch)

                all_embs.append(embs.cpu())
        elif need_metadata:
            pass

    if validate_model:
        raw_preds_np = torch.cat(all_raw_preds, dim=0).squeeze().numpy()
        labels_np = labels_tensor.cpu().numpy()
        accuracy = (raw_preds_np.round() == labels_np).sum() / len(labels_np)
        print(f"Accuracy: {accuracy:.4f}")

        cand['raw_preds'] = raw_preds_np

    embs = torch.cat(all_embs, dim=0).squeeze().numpy()
    print("shape of embeddings", np.shape(embs))

    umap_model = umap.UMAP(random_state=umap_seed)
    umap_emb = umap_model.fit_transform(embs)

    # cand["umap_emb_1"] = umap_emb[:, 0]
    # cand["umap_emb_2"] = umap_emb[:, 1]

    return umap_emb


if __name__ == "__main__":
    mm_maxvit_emb = get_torch_embedding(
        model_dir="models/mm_MaxViT_v10_N100_cuda/dutiful-sweep-11/",
        cand_path="data/test_cand_v10_N100.csv",
        trips_path="data/test_triplets_v10_N100.npy",
        metadata_cols=standard_metadata_cols,
        validate_model=False,
        batch_size=128,
    )

    mm_maxvit_emb_df = pd.DataFrame(mm_maxvit_emb, columns=["umap_emb_1", "umap_emb_2"])

    mm_maxvit_emb_df.to_csv(
        "embeddings/mm_MaxViT_v10_N100_dutiful-sweep-11.csv",
        index=False
    )
